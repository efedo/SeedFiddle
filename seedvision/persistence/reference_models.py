"""Byte-bound transport assets for immutable saved classifier recipes."""
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile

MAX_MODEL_ASSETS = 64
MAX_MODEL_BYTES = 8 * 1024 * 1024
MAX_TOTAL_MODEL_BYTES = 128 * 1024 * 1024
_OPERATIONS = {'external.plantcv.kmeans_centres': 'native_kmeans_recipe',
               'external.plantcv.naive_bayes_pdf': 'native_naive_bayes_recipe'}


def _digest(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def selection_sha256(value):
    if not isinstance(value, Mapping):
        raise ValueError('Model transport requires a selected model mapping.')
    return sha256(json.dumps(dict(value), sort_keys=True, separators=(',', ':'),
                             ensure_ascii=True, allow_nan=False).encode('utf-8')).hexdigest()


@dataclass(frozen=True, slots=True)
class ReferenceModelAsset:
    recipe_id: str
    recipe_version: str
    stage_id: str
    selection_sha256: str
    source_sha256: str
    source_bytes: int
    path: str

    def __post_init__(self):
        if any(type(value) is not str or not value or value != value.strip() or len(value) > 256
               for value in (self.recipe_id, self.recipe_version, self.stage_id)):
            raise ValueError('Model asset requires bounded recipe/stage identity.')
        if not _digest(self.selection_sha256) or not _digest(self.source_sha256):
            raise ValueError('Model asset requires lowercase selection/source SHA-256.')
        if type(self.source_bytes) is not int or not 1 <= self.source_bytes <= MAX_MODEL_BYTES:
            raise ValueError('Model asset byte length exceeds its bound.')
        path = self.path
        if (type(path) is not str or not path or len(path) > 4096 or '\\' in path or ':' in path
                or '\0' in path or PurePosixPath(path).is_absolute() or '..' in PurePosixPath(path).parts
                or str(PurePosixPath(path)) != path or path == '.'):
            raise ValueError('Model asset path must be canonical project-relative text.')

    @property
    def key(self):
        return self.recipe_id, self.recipe_version, self.stage_id, self.selection_sha256

    def to_dict(self):
        return asdict(self)


def _requirements(attachments, native_recipes=()):
    from seedvision.persistence.reference_graphs import validate_reference_graphs
    from seedvision.persistence.native_recipes import validate_native_recipes
    recipes = tuple(attachment.recipe for attachment in validate_reference_graphs(attachments))
    recipes += validate_native_recipes(native_recipes)
    rows, seen = [], set()
    for recipe in recipes:
        for stage in recipe.stages:
            if stage.operation_id not in _OPERATIONS:
                continue
            selection = recipe.parameters_for_runner(stage.stage_id).get('model_source')
            if selection is None:
                continue
            if stage.operation_id == 'external.plantcv.naive_bayes_pdf' and 'training_provenance' in selection:
                # A malformed new training record is a metadata error even when
                # unavailable historical model assets are otherwise permitted.
                _factory(stage).validate_model_source_metadata(selection)
            digest = selection_sha256(selection)
            if (not _digest(selection.get('source_sha256')) or type(selection.get('source_bytes')) is not int
                    or not 1 <= selection['source_bytes'] <= MAX_MODEL_BYTES):
                raise ValueError('Selected model has no bounded byte identity.')
            key = recipe.recipe_id, recipe.recipe_version, stage.stage_id, digest
            if key in seen:
                continue
            seen.add(key)
            rows.append((recipe, stage, selection, digest))
            if len(rows) > MAX_MODEL_ASSETS:
                raise ValueError('Selected model asset count exceeds its bound.')
    total = sum(size for _, size in {(value['source_sha256'], value['source_bytes'])
                                     for _, _, value, _ in rows})
    if total > MAX_TOTAL_MODEL_BYTES:
        raise ValueError('Selected model assets exceed their total byte bound.')
    return rows


def validate_reference_model_assets(assets, attachments, native_recipes=()):
    if type(assets) is not tuple or len(assets) > MAX_MODEL_ASSETS:
        raise ValueError('Project model assets require a bounded immutable tuple.')
    requirements = {(recipe.recipe_id, recipe.recipe_version, stage.stage_id, digest): selection
                    for recipe, stage, selection, digest in _requirements(attachments, native_recipes)}
    seen, paths = set(), {}
    for asset in assets:
        if not isinstance(asset, ReferenceModelAsset):
            raise ValueError('Project model asset has an invalid record type.')
        # Revalidate directly constructed records at every persistence boundary.
        ReferenceModelAsset(**asset.to_dict())
        if asset.key in seen or asset.key not in requirements:
            raise ValueError('Duplicate, stale or unselected project model binding.')
        selected = requirements[asset.key]
        if (asset.source_sha256 != selected['source_sha256'] or asset.source_bytes != selected['source_bytes']
                or asset.path in paths and paths[asset.path] != (asset.source_sha256, asset.source_bytes)):
            raise ValueError('Project model asset differs from its exact selected byte identity.')
        seen.add(asset.key)
        paths[asset.path] = asset.source_sha256, asset.source_bytes
    return assets


def reference_model_assets_from_payload(value, attachments, native_recipes=()):
    if type(value) is not list or len(value) > MAX_MODEL_ASSETS:
        raise ValueError('Project model assets must be a bounded list.')
    return validate_reference_model_assets(tuple(ReferenceModelAsset(**row) for row in value), attachments, native_recipes)


def resolve_asset_path(root, relative):
    root = Path(root).resolve()
    path = (root/relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Project model asset escapes its workspace root.')
    return path


def model_asset_paths(root, recipe, assets):
    """Resolve transport only; the executor rereads and verifies before caching."""
    selected = {stage.stage_id: recipe.parameters_for_runner(stage.stage_id).get('model_source')
                for stage in recipe.stages if stage.operation_id in _OPERATIONS}
    result = {}
    for asset in assets:
        if (asset.recipe_id, asset.recipe_version) != (recipe.recipe_id, recipe.recipe_version):
            continue
        value = selected.get(asset.stage_id)
        if value is None or selection_sha256(value) != asset.selection_sha256:
            # Another retained historical/working selection may share this stage.
            continue
        if asset.stage_id in result:
            raise ValueError('Duplicate project model transport stage.')
        result[asset.stage_id] = str(resolve_asset_path(root, asset.path))
    return result


def _factory(stage):
    from importlib import import_module
    return import_module('seedvision.benchmarking.'+_OPERATIONS[stage.operation_id])


def verified_asset_bytes(stage, selection, path):
    factory = _factory(stage)
    factory.load_selected_model(selection, resolved_path=str(Path(path).resolve(strict=True)))
    with Path(path).open('rb') as stream:
        raw = stream.read(selection['source_bytes']+1)
    if len(raw) != selection['source_bytes'] or sha256(raw).hexdigest() != selection['source_sha256']:
        raise ValueError('Project model asset bytes differ from their selected identity.')
    return raw


def capture_reference_model_assets(root, attachments, existing=(), *, allow_unavailable=False, native_recipes=()):
    """Publish immutable bounded model objects before publishing a project master."""
    requirements = _requirements(attachments, native_recipes)  # Before reads/copies.
    if (type(existing) is not tuple or len(existing) > MAX_MODEL_ASSETS
            or any(not isinstance(asset, ReferenceModelAsset) for asset in existing)):
        raise ValueError('Existing model assets must be a bounded immutable tuple.')
    for asset in existing:
        ReferenceModelAsset(**asset.to_dict())
    existing_by_key = {asset.key: asset for asset in existing}
    if len(existing_by_key) != len(existing):
        raise ValueError('Existing project model bindings must be unique.')
    result = []
    for recipe, stage, selection, digest in requirements:
        key = recipe.recipe_id, recipe.recipe_version, stage.stage_id, digest
        previous = existing_by_key.get(key)
        if previous is not None and (previous.source_sha256 != selection['source_sha256']
                                      or previous.source_bytes != selection['source_bytes']):
            raise ValueError('Existing project model asset differs from the selected bytes.')
        try:
            if previous is None:
                from seedvision.model_io.asset_paths import resolve_model_path
                source = Path(resolve_model_path(selection['path']))
            else:
                source = resolve_asset_path(root, previous.path)
            raw = verified_asset_bytes(stage, selection, source)
        except (ValueError, OSError):
            if not allow_unavailable:
                raise
            # Saving historical metadata is still possible. Preserve a bad
            # pinned transport so neither capture nor execution falls back.
            if previous is not None:
                result.append(previous)
            continue
        relative = 'native-models/'+selection['source_sha256']+'.model'
        destination = resolve_asset_path(root, relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            verified_asset_bytes(stage, selection, destination)
        else:
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(prefix='.model-', suffix='.incomplete',
                                                 dir=destination.parent, delete=False) as stream:
                    temporary = Path(stream.name)
                    stream.write(raw)
                    stream.flush()
                    os.fsync(stream.fileno())
                # Bound the copied bytes again before the immutable address publishes.
                verified_asset_bytes(stage, selection, temporary)
                os.replace(temporary, destination)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        result.append(ReferenceModelAsset(recipe.recipe_id, recipe.recipe_version, stage.stage_id,
            digest, selection['source_sha256'], selection['source_bytes'], relative))
    return validate_reference_model_assets(tuple(result), attachments, native_recipes)


def verify_reference_model_assets(root, assets, attachments, *, require_complete=False, native_recipes=()):
    assets = validate_reference_model_assets(assets, attachments, native_recipes)
    requirements = {(recipe.recipe_id, recipe.recipe_version, stage.stage_id, digest): (stage, selection)
                    for recipe, stage, selection, digest in _requirements(attachments, native_recipes)}
    if require_complete and {asset.key for asset in assets} != set(requirements):
        raise ValueError('Project classifier model closure is incomplete.')
    for asset in assets:
        stage, selection = requirements[asset.key]
        verified_asset_bytes(stage, selection, resolve_asset_path(root, asset.path))
    return assets


def unbound_model_selections(assets, attachments, native_recipes=()):
    bound = {asset.key for asset in validate_reference_model_assets(assets, attachments, native_recipes)}
    return tuple((recipe, stage, selection) for recipe, stage, selection, digest in _requirements(attachments, native_recipes)
        if (recipe.recipe_id, recipe.recipe_version, stage.stage_id, digest) not in bound)


def verify_original_model_selection(stage, selection):
    # No transport was bound. Preserve the original loader's native-path and
    # full byte/table checks rather than interpreting a foreign path locally.
    return _factory(stage).load_selected_model(selection)
