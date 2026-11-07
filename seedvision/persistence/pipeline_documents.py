"""Bounded standalone pipeline and layout metadata, independent of project saves."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import tempfile
import unicodedata

from seedvision.persistence.analysis_settings import (
    AnalysisSettingsProfile, analysis_settings_profile_from_payload,
    analysis_settings_profile_to_payload,
)

PIPELINE_EXTENSION = '.seedfiddle-pipeline.json'
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
    node_instances: tuple = ()
    layout: PipelineLayoutDocument = field(default_factory=PipelineLayoutDocument)

    def __post_init__(self):
        _pipeline_name(self.name)
        if not isinstance(self.analysis_settings, AnalysisSettingsProfile):
            raise PipelineDocumentError('Pipeline requires default analytical settings.')
        analysis_settings_profile_to_payload(self.analysis_settings)
        from seedvision.pipeline.catalogue_nodes import validate_node_instances, instance_nodes, display_id
        validate_node_instances(self.node_instances)
        known = {node.identifier for node in self.analysis_settings.nodes}
        for instance in self.node_instances:
            known.update(display_id(instance, identifier) for identifier in instance_nodes(instance))
        if not isinstance(self.layout, PipelineLayoutDocument):
            raise PipelineDocumentError('Pipeline requires validated layout metadata.')
        if any(row[0] not in known for row in self.layout.positions) or self.layout.selected_node not in known | {None}:
            raise PipelineDocumentError('Pipeline layout refers to an unknown stable node ID.')

    def to_dict(self):
        from seedvision.pipeline.catalogue_nodes import node_instances_to_payload
        return {'format': 'seedfiddle-pipeline', 'version': 2, 'name': self.name,
                'analysis_settings': analysis_settings_profile_to_payload(self.analysis_settings),
                'node_instances': node_instances_to_payload(self.node_instances), 'layout': self.layout.to_dict()}

    @classmethod
    def from_dict(cls, value):
        if type(value) is not dict or value.get('format') != 'seedfiddle-pipeline':
            raise PipelineDocumentError('This is not a Seed Fiddle pipeline document.')
        if type(value.get('version')) is not int or value['version'] != 2:
            raise PipelineDocumentError(f'Unsupported pipeline version {value.get("version")!r}; '
                                        'this application reads only version 2.')
        _keys(value, {'format', 'version', 'name', 'analysis_settings', 'node_instances', 'layout'})
        from seedvision.pipeline.catalogue_nodes import node_instances_from_payload
        return cls(value['name'], analysis_settings_profile_from_payload(value['analysis_settings']),
                   node_instances_from_payload(value['node_instances']), PipelineLayoutDocument.from_dict(value['layout']))


class PipelineDocumentStore:
    def __init__(self, root, *, factory_names=(), factory_paths=()):
        self.root = Path(root).resolve()
        from seedvision.pipeline.default_pipelines import GRAPH_PIPELINES
        self.factory_names = frozenset(_name_key(name) for name in (
            'Default', *(pipeline.title for pipeline in GRAPH_PIPELINES), *factory_names))
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


def _reject_duplicate_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise PipelineDocumentError('Standalone document repeats a JSON field.')
        result[key] = value
    return result


def _reject_constant(value):
    raise PipelineDocumentError(f'Standalone document contains the non-finite number {value}.')


def _read(path):
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
