"""Bounded standalone pipeline and layout metadata, independent of project saves."""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
import unicodedata

from seedvision.persistence.analysis_settings import (
    AnalysisSettingsProfile, analysis_settings_profile_from_payload,
    analysis_settings_profile_to_payload,
)
from seedvision.persistence.correction_inputs import (
    ProjectCorrectionBinding, validate_correction_binding_records, MAX_CORRECTION_BINDINGS,
)
from seedvision.persistence.reference_models import (
    ReferenceModelAsset, MAX_MODEL_ASSETS, MAX_TOTAL_MODEL_BYTES,
)
from seedvision.pipeline.workflows import PipelineWorkflow

PIPELINE_EXTENSION = '.seedfiddle-pipeline.json'
LAYOUT_EXTENSION = '.seedfiddle-layout.json'
MAX_DOCUMENT_BYTES = 24 * 1024 * 1024
MAX_LAYOUT_NODES = 4096


class PipelineDocumentError(ValueError):
    """The requested standalone document cannot be safely read or written."""


def _name_key(value):
    return ' '.join(unicodedata.normalize('NFKC', value).split()).casefold()


def _text(value, name, maximum=256):
    if type(value) is not str or not value.strip() or value != value.strip() or len(value) > maximum or '\0' in value:
        raise PipelineDocumentError(f'{name} must be bounded, nonempty text.')
    return value


def _pipeline_name(value):
    _text(value, 'Pipeline name')
    if (value in ('.', '..') or value.endswith('.')
            or any(character in '/\\:*?"<>|' or ord(character) < 32 or ord(character) == 127
                   for character in value)):
        raise PipelineDocumentError('Pipeline names cannot contain path separators, control characters or filename punctuation.')
    return value


def _keys(value, fields):
    if type(value) is not dict or set(value) != set(fields):
        raise PipelineDocumentError('Document has missing or unknown fields.')


def _backend_values(default, overrides, known):
    if type(default) is not str or default not in ('native', 'plantcv'):
        raise PipelineDocumentError('Backend default must be native or plantcv.')
    if (not isinstance(overrides, (dict, MappingProxyType)) or len(overrides) > MAX_LAYOUT_NODES
            or any(type(key) is not str or key not in known or type(value) is not str
                   or value not in ('native', 'plantcv') for key, value in overrides.items())):
        raise PipelineDocumentError('Backend overrides must name known nodes and backends.')


@dataclass(frozen=True)
class PipelineLayoutDocument:
    positions: tuple = ()
    bundle_cables: bool = False
    route_around_nodes: bool = False
    selected_node: str | None = None
    selected_output: str = ''
    image_visible: bool = True
    pipeline_visible: bool = True

    def __post_init__(self):
        if type(self.positions) is not tuple or len(self.positions) > MAX_LAYOUT_NODES:
            raise PipelineDocumentError('Layout positions must be a bounded immutable tuple.')
        seen = set()
        for row in self.positions:
            if (type(row) is not tuple or len(row) != 3 or type(row[0]) is not str
                    or not row[0] or len(row[0]) > 512 or '\0' in row[0] or row[0] in seen
                    or any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 1e9 for v in row[1:])):
                raise PipelineDocumentError('Invalid or duplicate stable node layout position.')
            seen.add(row[0])
        for value in (self.bundle_cables, self.route_around_nodes, self.image_visible, self.pipeline_visible):
            if type(value) is not bool:
                raise PipelineDocumentError('Layout presentation flags must be Boolean.')
        if self.selected_node is not None:
            _text(self.selected_node, 'Selected node', 512)
        if type(self.selected_output) is not str or len(self.selected_output) > 256 or '\0' in self.selected_output:
            raise PipelineDocumentError('Selected output must be bounded text.')

    def to_dict(self):
        return {'format': 'seedfiddle-layout', 'version': 1,
                'positions': [list(row) for row in self.positions],
                'bundle_cables': self.bundle_cables, 'route_around_nodes': self.route_around_nodes,
                'selected_node': self.selected_node, 'selected_output': self.selected_output,
                'image_visible': self.image_visible, 'pipeline_visible': self.pipeline_visible}

    @classmethod
    def from_dict(cls, value):
        _keys(value, cls().to_dict())
        if value['format'] != 'seedfiddle-layout' or type(value['version']) is not int or value['version'] != 1:
            raise PipelineDocumentError('Unsupported layout document format or version.')
        if type(value['positions']) is not list:
            raise PipelineDocumentError('Layout positions must be a JSON array.')
        return cls(**{key: tuple(tuple(row) if type(row) is list else row for row in values)
                      if key == 'positions' else values for key, values in value.items()
                      if key not in ('format', 'version')})

    @classmethod
    def from_graph(cls, graph, **presentation):
        nodes = {**graph.nodes, **getattr(graph, 'unused_nodes', {})}
        return cls(tuple((key, float(node.x), float(node.y)) for key, node in nodes.items()), **presentation)

    def apply(self, graph):
        """Move matching stable IDs only; never touch calculation state or caches."""
        nodes = {**graph.nodes, **getattr(graph, 'unused_nodes', {})}
        missing = []
        for identifier, x, y in self.positions:
            node = nodes.get(identifier)
            if node is None:
                missing.append(identifier)
            else:
                node.x, node.y = float(x), float(y)
        return tuple(missing)


@dataclass(frozen=True)
class PipelineDocument:
    name: str
    analysis_settings: AnalysisSettingsProfile
    pipeline_workflow: PipelineWorkflow | None = None
    node_instances: tuple = ()
    layout: PipelineLayoutDocument = field(default_factory=PipelineLayoutDocument)
    reference_model_assets: tuple = ()
    correction_bindings: tuple = ()
    asset_root: str | None = None
    backend_default: str = 'native'
    node_backends: object = field(default_factory=dict)

    def __post_init__(self):
        _pipeline_name(self.name)
        if not isinstance(self.analysis_settings, AnalysisSettingsProfile):
            raise PipelineDocumentError('Pipeline requires default analytical settings.')
        analysis_settings_profile_to_payload(self.analysis_settings)
        if self.pipeline_workflow is not None and not isinstance(self.pipeline_workflow, PipelineWorkflow):
            raise PipelineDocumentError('Pipeline primary workflow is invalid.')
        from seedvision.pipeline.catalogue_nodes import validate_node_instances, instance_nodes, display_id
        validate_node_instances(self.node_instances)
        known = {node.identifier for node in self.analysis_settings.nodes}
        if self.pipeline_workflow is not None:
            known.update(stage.stage_id for stage in self.pipeline_workflow.recipe.stages)
        for instance in self.node_instances:
            known.update(display_id(instance, identifier) for identifier in instance_nodes(instance))
        if not isinstance(self.layout, PipelineLayoutDocument):
            raise PipelineDocumentError('Pipeline requires validated layout metadata.')
        if any(row[0] not in known for row in self.layout.positions) or self.layout.selected_node not in known | {None}:
            raise PipelineDocumentError('Pipeline layout refers to an unknown stable node ID.')
        _backend_values(self.backend_default, self.node_backends, known)
        object.__setattr__(self, 'node_backends', MappingProxyType(dict(self.node_backends)))
        if self.asset_root is not None:
            _text(self.asset_root, 'Asset root', 4096)
            if not Path(self.asset_root).is_absolute():
                raise PipelineDocumentError('Asset root must be an absolute workspace locator.')
        _model_records(self.reference_model_assets)
        validate_correction_binding_records(self.correction_bindings)

    def to_dict(self):
        from seedvision.pipeline.catalogue_nodes import node_instances_to_payload
        return {'format': 'seedfiddle-pipeline', 'version': 1, 'name': self.name,
                'analysis_settings': analysis_settings_profile_to_payload(self.analysis_settings),
                'pipeline_workflow': None if self.pipeline_workflow is None else self.pipeline_workflow.to_dict(),
                'node_instances': node_instances_to_payload(self.node_instances), 'layout': self.layout.to_dict(),
                'reference_model_assets': [asset.to_dict() for asset in self.reference_model_assets],
                'correction_bindings': [binding.to_dict() for binding in self.correction_bindings],
                'asset_root': self.asset_root, 'backend_default': self.backend_default,
                'node_backends': dict(self.node_backends)}

    @classmethod
    def from_dict(cls, value):
        fields = {'format', 'version', 'name', 'analysis_settings', 'pipeline_workflow', 'node_instances',
                  'layout', 'reference_model_assets', 'correction_bindings', 'asset_root', 'backend_default', 'node_backends'}
        _keys(value, fields)
        if value['format'] != 'seedfiddle-pipeline' or type(value['version']) is not int or value['version'] != 1:
            raise PipelineDocumentError('Unsupported pipeline document format or version.')
        for name in ('reference_model_assets', 'correction_bindings'):
            if type(value[name]) is not list:
                raise PipelineDocumentError(f'{name} must be a bounded JSON array.')
        if len(value['reference_model_assets']) > MAX_MODEL_ASSETS:
            raise PipelineDocumentError('Too many model asset references.')
        if len(value['correction_bindings']) > MAX_CORRECTION_BINDINGS:
            raise PipelineDocumentError('Too many correction input references.')
        from seedvision.pipeline.catalogue_nodes import node_instances_from_payload
        return cls(value['name'], analysis_settings_profile_from_payload(value['analysis_settings']),
                   None if value['pipeline_workflow'] is None else PipelineWorkflow.from_dict(value['pipeline_workflow']),
                   node_instances_from_payload(value['node_instances']), PipelineLayoutDocument.from_dict(value['layout']),
                   tuple(_model_from_dict(row) for row in value['reference_model_assets']),
                   tuple(ProjectCorrectionBinding.from_dict(row) for row in value['correction_bindings']),
                   value['asset_root'], value['backend_default'], value['node_backends'])

    def merge_asset_records(self, existing_models=(), existing_bindings=()):
        """Retain unresolved project references; reject conflicting identities."""
        models = _merge_records(existing_models, self.reference_model_assets)
        bindings = _merge_records(existing_bindings, self.correction_bindings)
        _model_records(models)
        validate_correction_binding_records(bindings)
        return models, bindings

    def asset_statuses(self, root=None, *, verify=False):
        """Inspect transported assets in an explicit workspace, or their saved origin."""
        location = self.asset_root if root is None else root
        base = Path(location).resolve() if location is not None else None
        context_base = Path(self.asset_root).resolve() if self.asset_root else base
        references = [('model', asset.key, asset.path, asset.source_sha256, asset.source_bytes)
                      for asset in self.reference_model_assets]
        references.extend(('correction', binding.key + (asset.role,), asset.path, asset.source_sha256, asset.source_bytes)
                          for binding in self.correction_bindings for asset in binding.inputs)
        rows = []
        for kind, key, relative, digest, size in references:
            path = None if base is None or relative is None else (base / relative).resolve()
            status = 'unresolved'
            if path is not None:
                if not path.is_relative_to(base):
                    status = 'outside_asset_root'
                elif not path.is_file():
                    status = 'missing'
                elif not verify:
                    status = 'unchecked'
                else:
                    try:
                        if path.stat().st_size != size:
                            status = 'fingerprint_mismatch'
                        else:
                            hasher = sha256()
                            with path.open('rb') as stream:
                                for block in iter(lambda: stream.read(1024 * 1024), b''):
                                    hasher.update(block)
                            status = 'available' if hasher.hexdigest() == digest else 'fingerprint_mismatch'
                    except OSError:
                        status = 'unreadable'
            rows.append({'kind': kind, 'key': key, 'path': path, 'status': status})
        workflows = ([self.pipeline_workflow] if self.pipeline_workflow is not None else [])
        workflows.extend(instance.workflow for instance in self.node_instances if instance.workflow is not None)
        for workflow in workflows:
            for role, filename in workflow.input_paths.items():
                path = Path(filename)
                if not path.is_absolute() and context_base is None:
                    rows.append({'kind': 'context_input', 'key': (workflow.recipe.recipe_id, role),
                                 'path': path, 'status': 'unresolved'})
                    continue
                path = (context_base / path).resolve() if not path.is_absolute() else path
                rows.append({'kind': 'context_input', 'key': (workflow.recipe.recipe_id, role),
                             'path': path, 'status': 'unchecked' if path.is_file() else 'missing'})
        return tuple(rows)


def _model_from_dict(value):
    _keys(value, ReferenceModelAsset.__dataclass_fields__)
    return ReferenceModelAsset(**value)


def _model_records(values):
    if type(values) is not tuple or len(values) > MAX_MODEL_ASSETS:
        raise PipelineDocumentError('Model assets require a bounded immutable tuple.')
    seen, paths = set(), {}
    for value in values:
        if not isinstance(value, ReferenceModelAsset):
            raise PipelineDocumentError('Invalid model asset reference.')
        _model_from_dict(value.to_dict())
        identity = (value.source_sha256, value.source_bytes)
        if value.key in seen or value.path in paths and paths[value.path] != identity:
            raise PipelineDocumentError('Duplicate or conflicting model transport reference.')
        seen.add(value.key)
        paths[value.path] = identity
    if sum(size for _, size in set(paths.values())) > MAX_TOTAL_MODEL_BYTES:
        raise PipelineDocumentError('Model assets exceed their aggregate byte bound.')


def _merge_records(existing, incoming):
    merged = {record.key: record for record in existing}
    for record in incoming:
        if record.key in merged and merged[record.key] != record:
            raise PipelineDocumentError('Imported asset conflicts with an existing project reference.')
        merged[record.key] = record
    return tuple(merged.values())


class PipelineDocumentStore:
    def __init__(self, root, *, factory_names=(), factory_paths=()):
        self.root = Path(root).resolve()
        self.factory_names = frozenset(_name_key(name) for name in ('Default seed analysis', 'Default', *factory_names))
        self.factory_paths = frozenset(Path(path).resolve() for path in factory_paths)

    def _destination(self, path, extension, name=None):
        destination = Path(path).resolve()
        if not destination.name.casefold().endswith(extension):
            raise PipelineDocumentError(f'Document must use the {extension} extension.')
        protected_roots = tuple(self.root / folder for folder in ('seedvision', 'config', 'images', '.git'))
        if destination in self.factory_paths or any(destination.is_relative_to(root) for root in protected_roots):
            raise PipelineDocumentError('Factory definitions and project source fixtures cannot be overwritten.')
        stem = destination.name[:-len(extension)]
        if name is not None and (_name_key(name) in self.factory_names or _name_key(stem) in self.factory_names):
            raise PipelineDocumentError('Save a factory-derived pipeline under a distinct custom name and filename.')
        return destination

    def save_pipeline(self, document, path):
        if not isinstance(document, PipelineDocument):
            raise PipelineDocumentError('Expected a complete pipeline document.')
        destination = self._destination(path, PIPELINE_EXTENSION, document.name)
        payload = document.to_dict()
        PipelineDocument.from_dict(payload)
        return _atomic_write(destination, payload)

    def load_pipeline(self, path):
        return PipelineDocument.from_dict(_read(path))

    def save_layout(self, document, path):
        if not isinstance(document, PipelineLayoutDocument):
            raise PipelineDocumentError('Expected a layout document.')
        return _atomic_write(self._destination(path, LAYOUT_EXTENSION), document.to_dict())

    def load_layout(self, path):
        return PipelineLayoutDocument.from_dict(_read(path))


def _read(path):
    from seedvision.benchmarking.recipes import _reject_duplicate_pairs, _reject_constant
    try:
        with Path(path).open('rb') as stream:
            raw = stream.read(MAX_DOCUMENT_BYTES + 1)
        if len(raw) > MAX_DOCUMENT_BYTES:
            raise PipelineDocumentError('Standalone document exceeds its byte bound.')
        return json.loads(raw, object_pairs_hook=_reject_duplicate_pairs, parse_constant=_reject_constant)
    except (OSError, UnicodeError, ValueError, RecursionError) as error:
        raise PipelineDocumentError(f'Could not read standalone document: {error}') from error


def _atomic_write(destination, payload):
    try:
        encoded = (json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + '\n').encode('utf-8')
    except (TypeError, ValueError, RecursionError) as error:
        raise PipelineDocumentError(f'Could not encode standalone document: {error}') from error
    if len(encoded) > MAX_DOCUMENT_BYTES:
        raise PipelineDocumentError('Standalone document exceeds its byte bound.')
    temporary = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, filename = tempfile.mkstemp(prefix='.' + destination.name + '.', suffix='.tmp', dir=destination.parent)
        temporary = Path(filename)
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except OSError as error:
        raise PipelineDocumentError(f'Could not atomically save standalone document: {error}') from error
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return destination
