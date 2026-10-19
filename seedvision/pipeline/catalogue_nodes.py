"""Independent instances of registered calculations in the Pipeline canvas.

Canonical backend graphs keep their original node identities. Only their canvas
projection is namespaced; each instance owns settings, layout and execution scope.
"""
from dataclasses import dataclass, replace
from functools import lru_cache
from hashlib import sha256
import json
import math
import re
from uuid import uuid4

from seedvision.pipeline.model import PipelineGraph, PipelineConnection, build_default_pipeline
from seedvision.pipeline.workflows import (PipelineWorkflow, create_workflow, workflow_catalogue,
    workflow_graph, workflow_from_graph, validate_workflow)
from seedvision.persistence.analysis_settings import (AnalysisSettingsProfile,
    analysis_settings_profile_from_graph, analysis_settings_profile_to_payload,
    analysis_settings_profile_from_payload, apply_analysis_settings_profile)

MAX_NODE_INSTANCES = 64
MAX_NODE_INSTANCE_BYTES = 12 * 1024 * 1024
_PRIVATE_DEFAULT_ARGUMENTS = frozenset({'settings', 'calibration_settings', 'dish_settings',
    'layer_settings', 'advanced_settings', 'procedural_settings', 'unet_settings',
    'stardist_settings', 'trait_settings', 'enabled_nodes', 'node_cache', 'dirty_nodes',
    'cancellation_requested', 'background_colour_enabled'})


def instance_default_context(options):
    """Retain every image/review context input while keeping settings private."""
    return {key: value for key, value in options.items() if key not in _PRIVATE_DEFAULT_ARGUMENTS}


@dataclass(frozen=True)
class CatalogueEntry:
    key: str
    title: str
    category: str
    backend: str
    target_node: str
    recipe: object | None = None


def _default_catalogue_graph():
    graph = build_default_pipeline()
    for identifier in tuple(graph.unused_nodes):
        graph.restore_unused_node(identifier)
    return graph


def _contract_key(stage):
    value = stage.to_dict()
    for key in ('stage_id', 'title', 'description'):
        value.pop(key)
    for port in value['ports']:
        port.pop('label')
    for parameter in value['parameters']:
        parameter.pop('value')
        parameter.pop('description')
    for mapping in value['source_mappings']:
        mapping.pop('note')
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


@lru_cache(maxsize=1)
def catalogue_entries():
    graph = _default_catalogue_graph()
    entries = [CatalogueEntry('default:' + node.identifier, node.title, node.category,
                              'default', node.identifier) for node in graph.nodes.values()]
    seen = set()
    for recipe in workflow_catalogue():
        for stage in recipe.stages:
            key = _contract_key(stage)
            if key in seen:
                continue
            seen.add(key)
            category = ('SMPTS' if recipe.recipe_id.startswith('smpts_') else
                        'Native' if recipe.recipe_id.startswith('seedfiddle_') else 'PlantCV')
            entries.append(CatalogueEntry('native:' + key, stage.title, category,
                                          'native', stage.stage_id, recipe))
    return tuple(entries)


def catalogue_entry(key):
    if type(key) is not str:
        raise ValueError('Catalogue key must be text.')
    entry = next((entry for entry in catalogue_entries() if entry.key == key), None)
    if entry is None:
        raise ValueError('Unknown registered node catalogue entry.')
    return entry


@dataclass(frozen=True)
class NodeInstance:
    instance_id: str
    catalogue_key: str
    target_node: str
    workflow: PipelineWorkflow | None = None
    default_settings: AnalysisSettingsProfile | None = None
    layout: tuple = ()
    selected_node: str | None = None
    selected_output: str = ''

    def __post_init__(self):
        if type(self.instance_id) is not str or re.fullmatch(r'node_[0-9a-f]{32}', self.instance_id) is None:
            raise ValueError('Node instance requires its own bounded identity.')
        entry = catalogue_entry(self.catalogue_key)
        if self.target_node != entry.target_node:
            raise ValueError('Node instance target differs from its registered calculation.')
        if entry.backend == 'native':
            if not isinstance(self.workflow, PipelineWorkflow) or self.default_settings is not None:
                raise ValueError('Native node instance requires its own workflow.')
            validate_workflow(self.workflow)
            if self.workflow.source_recipe != entry.recipe:
                raise ValueError('Node instance workflow has a different registered source.')
            known_nodes = {stage.stage_id for stage in self.workflow.recipe.stages}
        else:
            if self.workflow is not None or not isinstance(self.default_settings, AnalysisSettingsProfile):
                raise ValueError('Default node instance requires a settings profile.')
            # JSON objects have no parameter order; normalize that irrelevant
            # ordering so save/reopen retains immutable descriptor equality.
            normalized = replace(self.default_settings, nodes=tuple(replace(node,
                parameters=tuple(sorted(node.parameters))) for node in self.default_settings.nodes))
            object.__setattr__(self, 'default_settings', normalized)
            graph = _default_catalogue_graph()
            apply_analysis_settings_profile(graph, self.default_settings)
            known_nodes = set(graph.nodes)
        if self.target_node not in known_nodes:
            raise ValueError('Node instance target must remain in its graph.')
        if self.selected_node is not None and (type(self.selected_node) is not str
                                               or self.selected_node not in instance_nodes(self)):
            raise ValueError('Selected node must belong to this calculation instance.')
        if (type(self.selected_output) is not str or len(self.selected_output) > 256
                or '\0' in self.selected_output):
            raise ValueError('Selected node output must be bounded text.')
        if type(self.layout) is not tuple or len(self.layout) > 4096:
            raise ValueError('Node instance layout must be a bounded immutable tuple.')
        seen = set()
        for row in self.layout:
            if (type(row) is not tuple or len(row) != 3 or row[0] not in known_nodes
                    or row[0] in seen or any(type(v) not in (int, float) or not math.isfinite(v)
                                           or abs(v) > 1e9 for v in row[1:])):
                raise ValueError('Invalid node instance canvas position.')
            seen.add(row[0])

    def to_dict(self):
        value = {'instance_id': self.instance_id, 'catalogue_key': self.catalogue_key,
                'target_node': self.target_node,
                'workflow': None if self.workflow is None else self.workflow.to_dict(),
                'default_settings': None if self.default_settings is None else
                    analysis_settings_profile_to_payload(self.default_settings),
                'layout': [list(row) for row in self.layout]}
        if self.selected_node is not None:
            value['selected_node'] = self.selected_node
        if self.selected_output:
            value['selected_output'] = self.selected_output
        return value

    @classmethod
    def from_dict(cls, value):
        fields = {'instance_id', 'catalogue_key', 'target_node', 'workflow', 'default_settings', 'layout'}
        if (type(value) is not dict or not fields <= set(value)
                or set(value) - fields - {'selected_node', 'selected_output'} or type(value['layout']) is not list):
            raise ValueError('Invalid node instance fields.')
        return cls(value['instance_id'], value['catalogue_key'], value['target_node'],
            None if value['workflow'] is None else PipelineWorkflow.from_dict(value['workflow']),
            None if value['default_settings'] is None else analysis_settings_profile_from_payload(value['default_settings']),
            tuple(tuple(row) if type(row) is list else row for row in value['layout']),
            value.get('selected_node'), value.get('selected_output', ''))


def create_instance(catalogue_key, existing_ids=()):
    entry = catalogue_entry(catalogue_key)
    identifier = 'node_' + uuid4().hex
    while identifier in existing_ids:
        identifier = 'node_' + uuid4().hex
    if entry.backend == 'native':
        return NodeInstance(identifier, catalogue_key, entry.target_node, create_workflow(entry.recipe))
    graph = _default_catalogue_graph()
    scope = {entry.target_node, *graph.upstream(entry.target_node)}
    pending = list(scope)
    while pending:
        for parent in graph.upstream(pending.pop()):
            if parent not in scope:
                scope.add(parent)
                pending.append(parent)
    for node in graph.nodes.values():
        node.enabled = node.identifier in scope
    return NodeInstance(identifier, catalogue_key, entry.target_node,
                        default_settings=analysis_settings_profile_from_graph(graph))


def default_graph(instance):
    if instance.default_settings is None:
        raise ValueError('This node instance uses the native backend.')
    graph = _default_catalogue_graph()
    apply_analysis_settings_profile(graph, instance.default_settings)
    return graph


def _backend_graph(instance):
    return workflow_graph(instance.workflow) if instance.workflow is not None else default_graph(instance)


def instance_nodes(instance):
    # Ownership follows the registered closure, not the currently wired path.
    # Rewiring around a filter must not strand that instance's visible card.
    entry = catalogue_entry(instance.catalogue_key)
    if entry.backend == 'native':
        recipe = instance.workflow.source_recipe
        incoming = {stage.stage_id: [] for stage in recipe.stages}
        for edge in recipe.dependencies:
            incoming[edge.target_stage].append(edge.source_stage)
        scope, pending = {instance.target_node}, [instance.target_node]
        while pending:
            for parent in incoming[pending.pop()]:
                if parent not in scope:
                    scope.add(parent)
                    pending.append(parent)
        return tuple(stage.stage_id for stage in recipe.stages if stage.stage_id in scope)
    graph = _default_catalogue_graph()
    scope, pending = {instance.target_node}, [instance.target_node]
    while pending:
        for parent in graph.upstream(pending.pop()):
            if parent not in scope:
                scope.add(parent)
                pending.append(parent)
    return tuple(identifier for identifier in graph.topological_order() if identifier in scope)


def display_id(instance, node_id):
    return instance.instance_id + '::' + node_id


def canonical_id(instance, identifier):
    prefix = instance.instance_id + '::'
    if type(identifier) is not str or not identifier.startswith(prefix):
        raise ValueError('Node does not belong to this calculation instance.')
    node_id = identifier[len(prefix):]
    if node_id not in instance_nodes(instance):
        raise ValueError('Node is outside this instance dependency scope.')
    return node_id


def instance_graph(instance):
    source = _backend_graph(instance)
    if instance.workflow is not None:
        from seedvision.ui.native_workflow_helpers import configure_native_parameters
        configure_native_parameters(source, instance.workflow.recipe, typed=True, editable_only=True)
    scope = set(instance_nodes(instance))
    positions = {row[0]: row[1:] for row in instance.layout}
    nodes = [replace(node, identifier=display_id(instance, identifier),
        parameters=dict(node.parameters), input_port_types=dict(node.input_port_types),
        output_port_types=dict(node.output_port_types),
        x=positions.get(identifier, (node.x, node.y))[0],
        y=positions.get(identifier, (node.x, node.y))[1])
        for identifier, node in source.nodes.items() if identifier in scope]
    def projected(edge):
        return replace(edge, source=display_id(instance, edge.source), target=display_id(instance, edge.target))
    # Do not invoke the default catalogue's ID-based socket-name normalizer on
    # namespaced display IDs. The registered sockets are already explicit.
    graph = PipelineGraph((), ())
    graph.nodes = {node.identifier: node for node in nodes}
    graph.connections = tuple(projected(edge) for edge in source.connections
                              if edge.source in scope and edge.target in scope)
    graph.connection_templates = tuple(projected(edge) for edge in source.connection_templates
                                       if edge.source in scope and edge.target in scope)
    graph.instance_id = instance.instance_id
    graph.catalogue_key = instance.catalogue_key
    graph._connection_suspended = {display_id(instance, identifier) for identifier in source._connection_suspended
                                   if identifier in scope}
    graph._validate_connections()
    graph.topological_order()
    return graph


def sync_instance(instance, projection):
    """Copy this component's edits out of a possibly merged display graph."""
    source = _backend_graph(instance)
    scope = set(instance_nodes(instance))
    prefix = instance.instance_id + '::'
    expected = {display_id(instance, identifier) for identifier in scope}
    if {identifier for identifier in projection.nodes if identifier.startswith(prefix)} != expected:
        raise ValueError('A calculation instance must retain its registered dependency nodes.')
    for identifier in scope:
        edited, node = projection.node(display_id(instance, identifier)), source.node(identifier)
        node.parameters = dict(edited.parameters)
        node.enabled = edited.enabled
        if instance.workflow is not None:
            for parameter in instance.workflow.recipe.stage(identifier).parameters:
                value = node.parameters[parameter.parameter_id]
                if parameter.value_type == 'json' and isinstance(value, str):
                    node.parameters[parameter.parameter_id] = json.loads(value)
        node.x, node.y = float(edited.x), float(edited.y)
    edges = []
    for edge in projection.connections:
        belongs = edge.source.startswith(prefix), edge.target.startswith(prefix)
        if any(belongs) and not all(belongs):
            raise ValueError('Independent calculations require an explicit adapter to share outputs.')
        if all(belongs):
            edges.append(replace(edge, source=edge.source[len(prefix):], target=edge.target[len(prefix):]))
    source.connections = tuple(edge for edge in source.connections
                                if edge.source not in scope or edge.target not in scope) + tuple(edges)
    source._connection_suspended = (source._connection_suspended - scope) | {
        identifier[len(prefix):] for identifier in projection._connection_suspended if identifier.startswith(prefix)}
    source._validate_connections()
    source.topological_order()
    layout = tuple((identifier, source.node(identifier).x, source.node(identifier).y) for identifier in sorted(scope))
    if instance.workflow is not None:
        return replace(instance, workflow=workflow_from_graph(instance.workflow, source), layout=layout)
    settings = analysis_settings_profile_from_graph(source)
    return replace(instance, default_settings=settings, layout=layout)


def validate_node_instances(instances):
    if type(instances) is not tuple or len(instances) > MAX_NODE_INSTANCES:
        raise ValueError('Node instances must be a bounded immutable tuple.')
    seen = set()
    for instance in instances:
        if not isinstance(instance, NodeInstance) or instance.instance_id in seen:
            raise ValueError('Node instances require unique validated identities.')
        # Revalidate the immutable in-memory descriptor directly. Re-decoding
        # its entire canonical workflow would repeat the same source admission
        # both in PipelineWorkflow.from_dict and NodeInstance construction.
        instance.__post_init__()
        seen.add(instance.instance_id)
    return instances


def node_instances_to_payload(instances):
    values = [instance.to_dict() for instance in validate_node_instances(instances)]
    if len(json.dumps(values, allow_nan=False).encode('utf-8')) > MAX_NODE_INSTANCE_BYTES:
        raise ValueError('Node instances exceed their document byte bound.')
    return values


def node_instances_from_payload(value):
    if type(value) is not list or len(value) > MAX_NODE_INSTANCES:
        raise ValueError('Node instances must be a bounded JSON array.')
    if len(json.dumps(value, allow_nan=False).encode('utf-8')) > MAX_NODE_INSTANCE_BYTES:
        raise ValueError('Node instances exceed their document byte bound.')
    return validate_node_instances(tuple(NodeInstance.from_dict(row) for row in value))


def native_instance_recipes(instances):
    return tuple(instance.workflow.recipe for instance in validate_node_instances(instances)
                 if instance.workflow is not None)


def execute_instance(instance, path, *, cache=None, cancel=None, default_context=None,
                     model_assets=None, correction_paths=None, previous=None,
                     target_node=None, dirty_nodes=frozenset()):
    """Run one independent canonical backend with its selected dependency scope."""
    target = instance.target_node if target_node is None else target_node
    if target not in instance_nodes(instance):
        raise ValueError('Execution target is outside this calculation instance.')
    if instance.workflow is not None:
        if instance.workflow.recipe.recipe_id == 'plantcv_color_card_correct_color':
            from seedvision.benchmarking.native_correction_inputs import run_native_correction_files
            paths = dict(instance.workflow.input_paths if correction_paths is None else correction_paths)
            paths['source_rgb'] = str(path)
            if set(paths) != {'target_rgb', 'target_mask', 'source_rgb', 'source_mask'}:
                raise ValueError('Select the target RGB image and both chip-label masks for this correction instance.')
            return run_native_correction_files(instance.workflow.recipe, paths,
                cache=cache, previous=previous, cancel=cancel, working_workflow=instance.workflow)
        from seedvision.pipeline.workflows import run_workflow_path
        return run_workflow_path(instance.workflow, path, cache=cache, cancel=cancel,
            target_node_id=target, model_asset_paths=model_assets or {})
    from seedvision.pipeline.settings import production_settings
    from seedvision.segmentation.baseline import analyze_path
    graph = default_graph(instance)
    if not graph.node(target).implemented:
        raise ValueError('This registered node has no calculation backend yet.')
    scope, pending = {target}, [target]
    while pending:
        for parent in graph.upstream(pending.pop()):
            if parent not in scope:
                scope.add(parent)
                pending.append(parent)
    enabled = frozenset(identifier for identifier in scope if graph.node(identifier).enabled)
    context = instance_default_context(default_context or {})
    context.setdefault('background_colour_enabled', bool(graph.node('background_likelihood').parameters.get('background_colour_enabled', True)))
    return analyze_path(path, **production_settings(graph), enabled_nodes=enabled,
        node_cache=cache, dirty_nodes=dirty_nodes, cancellation_requested=cancel,
        **context)


def execute_bound_correction(instance, root, binding, image_record, path, *,
                             cache=None, previous=None, cancel=None):
    """Admit an owned selection around the unchanged canonical file adapter."""
    from seedvision.persistence.correction_inputs import (validate_correction_bindings,
        verify_correction_bindings, CorrectionRunOrigins, validate_correction_run_origins)
    from seedvision.benchmarking.project_correction import run_project_correction
    if binding.owner_id != instance.instance_id or instance.workflow is None:
        raise ValueError('Correction selection belongs to a different calculation instance.')
    validate_correction_bindings((binding,), (image_record,), node_instances=(instance,))
    recipe = instance.workflow.recipe
    result, prepared = run_project_correction(recipe, root, replace(binding, owner_id=None),
        image_record, path, cache=cache, previous=previous, cancel=cancel, working_workflow=instance.workflow)
    verify_correction_bindings(root, (binding,), (image_record,), node_instances=(instance,))
    origins = CorrectionRunOrigins(binding, image_record.source.location, image_record.source.path)
    validate_correction_run_origins(recipe, origins, result.input_sources)
    return replace(result, input_origins=origins), prepared
