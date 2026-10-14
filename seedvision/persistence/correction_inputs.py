"""Exact per-image correction selections and bounded project input transport.

Origin records describe the four selected inputs. Relative paths describe their
portable copies; moving a project never changes those origin records.
"""
from dataclasses import dataclass, replace
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile

from seedvision.benchmarking.checkpoints import CoordinateFrame
from seedvision.model_io.asset_paths import validate_model_origin, resolve_model_path


PORTS = ('target_rgb', 'target_mask', 'source_rgb', 'source_mask')
RECIPE_ID = 'plantcv_color_card_correct_color'
MAX_CORRECTION_BINDINGS = 256
MAX_INPUT_BYTES = 256 * 1024 * 1024
MAX_TOTAL_INPUT_BYTES = 1024 * 1024 * 1024
COPY_CHUNK_BYTES = 1024 * 1024
PROJECT_RUNTIME_KEYS = ('correction_project_adapter', 'correction_binding_validation', 'correction_project_metadata')


def _text(value, name, maximum=256):
    if type(value) is not str or not value or value != value.strip() or '\0' in value or len(value) > maximum:
        raise ValueError(f'Correction {name} must be bounded nonempty text.')
    return value


def _digest(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('Correction identity requires lowercase SHA-256.')
    return value


def _relative(value):
    if (type(value) is not str or not value or len(value) > 4096 or '\\' in value or ':' in value
            or '\0' in value or PurePosixPath(value).is_absolute() or '..' in PurePosixPath(value).parts
            or str(PurePosixPath(value)) != value or value == '.'):
        raise ValueError('Correction asset path must be canonical project-relative text.')
    return value


@dataclass(frozen=True, slots=True)
class CorrectionInputAsset:
    role: str
    origin_path: str
    source_sha256: str
    source_bytes: int
    frame: CoordinateFrame
    decode: str
    path: str

    def __post_init__(self):
        if self.role not in PORTS:
            raise ValueError('Unknown correction input role.')
        validate_model_origin(self.origin_path)
        _digest(self.source_sha256)
        if type(self.source_bytes) is not int or not 1 <= self.source_bytes <= MAX_INPUT_BYTES:
            raise ValueError('Correction input exceeds its byte bound.')
        if not isinstance(self.frame, CoordinateFrame):
            raise ValueError('Correction input requires a decoded coordinate frame.')
        CoordinateFrame.from_dict(self.frame.to_dict())
        _text(self.decode, 'decode record', 8192)
        _relative(self.path)

    def source_record(self):
        return {'path': self.origin_path, 'sha256': self.source_sha256,
                'bytes': self.source_bytes, 'frame': self.frame.to_dict(), 'decode': self.decode}

    def to_dict(self):
        return {'role': self.role, 'origin_path': self.origin_path,
                'source_sha256': self.source_sha256, 'source_bytes': self.source_bytes,
                'frame': self.frame.to_dict(), 'decode': self.decode, 'path': self.path}

    @classmethod
    def from_dict(cls, value):
        if type(value) is not dict or set(value) != {
                'role', 'origin_path', 'source_sha256', 'source_bytes', 'frame', 'decode', 'path'}:
            raise ValueError('Correction asset has missing or unknown fields.')
        return cls(**(value | {'frame': CoordinateFrame.from_dict(value['frame'])}))


@dataclass(frozen=True, slots=True)
class ProjectCorrectionBinding:
    image_id: str
    recipe_id: str
    recipe_version: str
    recipe_sha256: str
    inputs: tuple[CorrectionInputAsset, ...]

    def __post_init__(self):
        _text(self.image_id, 'image identifier')
        _text(self.recipe_id, 'recipe identifier')
        _text(self.recipe_version, 'recipe version')
        _digest(self.recipe_sha256)
        if self.recipe_id != RECIPE_ID:
            raise ValueError('Correction binding requires the correction recipe.')
        if (type(self.inputs) is not tuple or len(self.inputs) != len(PORTS)
                or any(not isinstance(item, CorrectionInputAsset) for item in self.inputs)
                or tuple(item.role for item in self.inputs) != PORTS):
            raise ValueError('Correction binding requires four ordered, distinct input roles.')
        for item in self.inputs:
            CorrectionInputAsset.from_dict(item.to_dict())
        # Reuse the file runner's paired-frame and aggregate-pixel contract.
        from seedvision.benchmarking.native_correction_inputs import validate_correction_input_sources
        validate_correction_input_sources(self.input_sources)

    @property
    def key(self):
        return self.image_id, self.recipe_id, self.recipe_version, self.recipe_sha256

    @property
    def input_sources(self):
        return {item.role: item.source_record() for item in self.inputs}

    @property
    def selection_sha256(self):
        return sha256(json.dumps(self.input_sources, sort_keys=True, separators=(',', ':'),
                                 ensure_ascii=True, allow_nan=False).encode('utf-8')).hexdigest()

    def to_dict(self):
        return {'image_id': self.image_id, 'recipe_id': self.recipe_id,
                'recipe_version': self.recipe_version, 'recipe_sha256': self.recipe_sha256,
                'inputs': [item.to_dict() for item in self.inputs]}

    @classmethod
    def from_dict(cls, value):
        if type(value) is not dict or set(value) != {
                'image_id', 'recipe_id', 'recipe_version', 'recipe_sha256', 'inputs'}:
            raise ValueError('Correction binding has missing or unknown fields.')
        if type(value['inputs']) is not list or len(value['inputs']) != len(PORTS):
            raise ValueError('Correction binding inputs must be a four-item list.')
        return cls(**(value | {'inputs': tuple(CorrectionInputAsset.from_dict(item) for item in value['inputs'])}))


def _recipes(attachments, native_recipes):
    from seedvision.persistence.reference_graphs import validate_reference_graphs
    from seedvision.persistence.native_recipes import validate_native_recipes
    from seedvision.benchmarking.native_correction_recipe import _validate_recipe
    result = {}
    recipes = tuple(item.recipe for item in validate_reference_graphs(attachments))
    recipes += validate_native_recipes(native_recipes)
    for recipe in recipes:
        if recipe.recipe_id == RECIPE_ID:
            _validate_recipe(recipe)
            result[(recipe.recipe_id, recipe.recipe_version, recipe.identity.recipe_sha256)] = recipe
    return result


def validate_correction_binding_records(bindings):
    if type(bindings) is not tuple or len(bindings) > MAX_CORRECTION_BINDINGS:
        raise ValueError('Project correction bindings require a bounded immutable tuple.')
    seen, paths, contents = set(), {}, set()
    for binding in bindings:
        if not isinstance(binding, ProjectCorrectionBinding):
            raise ValueError('Invalid project correction binding record.')
        ProjectCorrectionBinding.from_dict(binding.to_dict())
        if binding.key in seen:
            raise ValueError('Duplicate project correction binding.')
        seen.add(binding.key)
        for item in binding.inputs:
            content = item.source_sha256, item.source_bytes
            if item.path in paths and paths[item.path] != content:
                raise ValueError('Correction transport path has conflicting byte identities.')
            paths[item.path] = content
            contents.add(content)
    if sum(size for _, size in contents) > MAX_TOTAL_INPUT_BYTES:
        raise ValueError('Project correction assets exceed their aggregate byte bound.')
    return bindings


def validate_correction_bindings(bindings, images, attachments=(), native_recipes=()):
    bindings = validate_correction_binding_records(bindings)
    if not bindings:
        return bindings
    selected = _recipes(attachments, native_recipes)
    images_by_id = {image.identifier: image for image in images}
    for binding in bindings:
        if binding.key[1:] not in selected:
            raise ValueError('Stale or unselected project correction binding.')
        image = images_by_id.get(binding.image_id)
        source = binding.inputs[PORTS.index('source_rgb')]
        if (image is None or source.source_sha256 != image.source.sha256
                or source.frame.shape != tuple(image.source_shape)):
            raise ValueError('Correction source RGB must match its exact project image identity and frame.')
    return bindings


def correction_bindings_from_payload(value, images, attachments=(), native_recipes=()):
    if type(value) is not list or len(value) > MAX_CORRECTION_BINDINGS:
        raise ValueError('Project correction bindings must be a bounded JSON list.')
    return validate_correction_bindings(tuple(ProjectCorrectionBinding.from_dict(row) for row in value),
                                        images, attachments, native_recipes)


def binding_for_result(recipe, image_id, result):
    from seedvision.benchmarking.native_correction_recipe import NativeCorrectionResult, _validate_recipe
    _validate_recipe(recipe)
    if (not isinstance(result, NativeCorrectionResult)
            or result.recipe_identity.analytical_dict() != recipe.identity.analytical_dict()):
        raise ValueError('Correction binding requires the selected recipe\'s actual four-file result.')
    from seedvision.benchmarking.native_correction_inputs import validate_correction_input_sources
    sources = validate_correction_input_sources(result.input_sources)
    inputs = tuple(CorrectionInputAsset(role, row['path'], row['sha256'], row['bytes'],
        CoordinateFrame.from_dict(row['frame']), row['decode'], 'native-correction/'+row['sha256']+'.image')
        for role in PORTS for row in (sources[role],))
    return ProjectCorrectionBinding(image_id, recipe.recipe_id, recipe.recipe_version,
                                    recipe.identity.recipe_sha256, inputs)


def resolve_correction_asset_path(root, relative):
    _relative(relative)
    root = Path(root).resolve()
    path = (root/relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Project correction asset escapes its workspace root.')
    return path


def _cancel(cancel):
    if cancel is not None and cancel():
        raise RuntimeError('Project correction asset operation cancelled.')


def copy_verified_input(source, asset, destination=None, *, cancel=None):
    """Stream at most the selected byte length plus one; publish nothing here."""
    if cancel is not None and not callable(cancel):
        raise TypeError('cancel must be callable or None.')
    _cancel(cancel)
    source = Path(source)
    if source.stat().st_size != asset.source_bytes or not source.is_file():
        raise ValueError('Correction asset size/type differs from its selected bytes.')
    digest, total = sha256(), 0
    with source.open('rb') as stream:
        while total <= asset.source_bytes:
            _cancel(cancel)
            block = stream.read(min(COPY_CHUNK_BYTES, asset.source_bytes+1-total))
            if not block:
                break
            total += len(block)
            if total > asset.source_bytes:
                raise ValueError('Correction asset grew beyond its selected byte bound.')
            digest.update(block)
            if destination is not None:
                destination.write(block)
    _cancel(cancel)
    if (total != asset.source_bytes or digest.hexdigest() != asset.source_sha256
            or source.stat().st_size != asset.source_bytes):
        raise ValueError('Correction asset differs from its selected byte identity.')
    if cancel is not None:
        # The final caller callback can change same-sized source bytes. Audit
        # them again without any callbacks before returning the copied hash.
        copy_verified_input(source, asset)
    return digest.hexdigest()


def capture_correction_bindings(root, bindings, images, attachments=(), native_recipes=(), *,
                                existing=None, cancel=None):
    """Verify/publish all role copies before a project master can reference them.

    A retained selection reads only pinned transport, never its original path.
    New/reselected inputs read their origin. Unreferenced immutable objects from
    an interrupted multi-file capture are harmless; the master stays unchanged.
    Omitting existing preserves every declared pin. Pass the prior tuple
    explicitly (empty for a wholly new selection) to capture new origin bytes.
    """
    bindings = validate_correction_bindings(bindings, images, attachments, native_recipes)
    if existing is None:
        existing = bindings
    if type(existing) is not tuple or len(existing) > MAX_CORRECTION_BINDINGS:
        raise ValueError('Existing correction bindings require a bounded tuple.')
    previous = {}
    for binding in existing:
        if not isinstance(binding, ProjectCorrectionBinding):
            raise ValueError('Invalid existing correction binding.')
        ProjectCorrectionBinding.from_dict(binding.to_dict())
        if binding.key in previous:
            raise ValueError('Duplicate existing correction binding.')
        previous[binding.key] = binding
    result = []
    for binding in bindings:
        old = previous.get(binding.key)
        retained = old is not None and old.selection_sha256 == binding.selection_sha256
        inputs = []
        for index, item in enumerate(binding.inputs):
            _cancel(cancel)
            source = (resolve_correction_asset_path(root, old.inputs[index].path) if retained
                      else Path(resolve_model_path(item.origin_path)))
            relative = 'native-correction/'+item.source_sha256+'.image'
            target = resolve_correction_asset_path(root, relative)
            # Verify a bound source even when a deduplicated destination exists.
            copy_verified_input(source, item, cancel=cancel)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                copy_verified_input(target, item, cancel=cancel)
            else:
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.correction-',
                                                     suffix='.incomplete', delete=False) as stream:
                        temporary = Path(stream.name)
                        copy_verified_input(source, item, stream, cancel=cancel)
                        stream.flush()
                        os.fsync(stream.fileno())
                    copy_verified_input(temporary, item, cancel=cancel)
                    copy_verified_input(source, item, cancel=cancel)
                    _cancel(cancel)
                    os.replace(temporary, target)
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
            inputs.append(replace(item, path=relative))
        result.append(replace(binding, inputs=tuple(inputs)))
    result = validate_correction_bindings(tuple(result), images, attachments, native_recipes)
    _cancel(cancel)
    # No caller callbacks after the final bounded transport audit.
    verify_correction_bindings(root, result, images, attachments, native_recipes)
    return result


def verify_correction_bindings(root, bindings, images, attachments=(), native_recipes=(), *, cancel=None):
    bindings = validate_correction_bindings(bindings, images, attachments, native_recipes)
    for binding in bindings:
        for asset in binding.inputs:
            copy_verified_input(resolve_correction_asset_path(root, asset.path), asset, cancel=cancel)
    return bindings


def select_correction_binding(current, selected):
    if type(current) is not tuple or not isinstance(selected, ProjectCorrectionBinding):
        raise ValueError('Correction selections require immutable binding records.')
    found = any(item.key == selected.key for item in current)
    return tuple(selected if item.key == selected.key else item for item in current) + (() if found else (selected,))


def retain_selected_correction_bindings(bindings, attachments=(), native_recipes=()):
    selected = _recipes(attachments, native_recipes)
    return tuple(binding for binding in bindings if binding.key[1:] in selected)


@dataclass(frozen=True, slots=True)
class CorrectionRunOrigins:
    """Saved selection plus its exact logical project-image locator.

    The run's separate input_sources retain actual opened paths and current
    decode facts. This compact origin record never asserts originals were read.
    """
    binding: ProjectCorrectionBinding
    project_source_location: str
    project_source_path: str

    def __post_init__(self):
        if not isinstance(self.binding, ProjectCorrectionBinding):
            raise ValueError('Correction run origins require a project binding.')
        ProjectCorrectionBinding.from_dict(self.binding.to_dict())
        from seedvision.persistence.project_analysis import _validated_source_locator, _image_identifier
        _text(self.project_source_path, 'project image locator', 4096)
        location, path = _validated_source_locator(self.project_source_location, self.project_source_path)
        if self.binding.image_id != _image_identifier(location, path):
            raise ValueError('Correction origin image ID differs from its exact project locator.')

    def to_dict(self):
        return {'binding': self.binding.to_dict(), 'project_source_location': self.project_source_location,
                'project_source_path': self.project_source_path}

    @classmethod
    def from_dict(cls, value):
        if type(value) is not dict or set(value) != {'binding', 'project_source_location', 'project_source_path'}:
            raise ValueError('Correction run origins have missing or unknown fields.')
        return cls(**(value | {'binding': ProjectCorrectionBinding.from_dict(value['binding'])}))


def validate_correction_run_origins(recipe, origins, actual_sources):
    from seedvision.benchmarking.native_correction_recipe import _validate_recipe
    from seedvision.benchmarking.native_correction_inputs import validate_correction_input_sources
    _validate_recipe(recipe)
    if not isinstance(origins, CorrectionRunOrigins):
        raise ValueError('Correction run origins require an immutable origin record.')
    CorrectionRunOrigins.from_dict(origins.to_dict())
    binding = origins.binding
    if (binding.recipe_id, binding.recipe_version, binding.recipe_sha256) != (
            recipe.recipe_id, recipe.recipe_version, recipe.identity.recipe_sha256):
        raise ValueError('Correction run origins differ from the selected analytical recipe.')
    sources = validate_correction_input_sources(actual_sources)
    for asset in binding.inputs:
        actual = sources[asset.role]
        if (asset.source_sha256 != actual['sha256'] or asset.source_bytes != actual['bytes']
                or asset.frame.to_dict() != actual['frame']):
            raise ValueError('Correction opened bytes/frames differ from the exact saved origin selection.')
    return origins
