"""Supported editable CUDA workflows, separate from canonical source references."""
from dataclasses import dataclass, replace
from types import MappingProxyType
from collections.abc import Mapping
from seedvision.benchmarking.recipes import (RecipeSpec, RecipeProvenanceMode, DependencyContract,
    LayoutPosition, BranchCondition, to_pipeline_graph)
from seedvision.pipeline.model import PipelineConnection, NodeStatus

FOREGROUND_OPERATIONS = frozenset(('threshold.binary', 'threshold.mean', 'threshold.otsu',
    'smpts_paper.intersection'))


def restrict_foreground(mask, bounds):
    """Exclude exterior foreground in a private CUDA mask, preserving its frame."""
    from seedvision.benchmarking.native_morphology_roi_recipe import selected_bounds
    roi = selected_bounds(bounds, mask.shape)
    if roi is None:
        return mask
    x0, y0, x1, y1 = roi
    result = mask.clone()
    result[:y0] = 0
    result[y1:] = 0
    result[y0:y1, :x0] = 0
    result[y0:y1, x1:] = 0
    return result


def workflow_catalogue():
    from seedvision.benchmarking.recipe_catalogue import build_recipe_catalogue
    from seedvision.benchmarking.native_recipe import build_native_seed_recipe, _SEED_RECIPES
    from seedvision.benchmarking.native_smpts_recipe import build_native_smpts_recipe
    catalogue = build_recipe_catalogue()
    from seedvision.benchmarking.native_extended_threshold_recipes import build_native_extended_threshold_recipes
    from seedvision.benchmarking.native_range_recipe import build_native_range_recipe
    from seedvision.benchmarking.native_naive_bayes_recipe import build_native_naive_bayes_recipe
    from seedvision.benchmarking.native_kmeans_recipe import build_native_kmeans_recipe
    from seedvision.benchmarking.native_card_recipe import build_native_card_recipe
    from seedvision.benchmarking.native_calibrated_seed_recipe import build_native_calibrated_seed_recipe
    from seedvision.benchmarking.native_core_recovery_recipe import build_native_core_recovery_recipe
    from seedvision.benchmarking.native_correction_recipe import build_native_correction_recipe
    return (tuple(build_native_seed_recipe(catalogue.recipe(key)) for key in sorted(_SEED_RECIPES))
        + tuple(build_native_extended_threshold_recipes().values())
        + (build_native_range_recipe(), build_native_naive_bayes_recipe(), build_native_kmeans_recipe(),
           build_native_card_recipe(), build_native_calibrated_seed_recipe('binary'),
           build_native_calibrated_seed_recipe('mean'), build_native_calibrated_seed_recipe('otsu'),
           build_native_calibrated_seed_recipe('range'), build_native_core_recovery_recipe(), build_native_smpts_recipe(), build_native_correction_recipe()))


def _working_provenance(source):
    return replace(source.provenance, mode=RecipeProvenanceMode.ENHANCED,
        evidence_note='Editable native workspace derived from the recorded source recipe. '
        'Working topology is separately validated; original parity and scientific validation are not asserted.')


def bypass_ports(stage):
    """Only unambiguous, semantics-preserving pass-through stages are bypassable."""
    inputs = [p for p in stage.ports if p.direction == 'input']
    outputs = [p for p in stage.ports if p.direction == 'output']
    result = {}
    for output in outputs:
        matches = [p for p in inputs if p.data_type == output.data_type and p.semantics == output.semantics]
        if len(matches) != 1:
            return {}
        result[output.name] = matches[0].name
    return result


@dataclass(frozen=True)
class PipelineWorkflow:
    source_recipe: RecipeSpec
    recipe: RecipeSpec
    disabled_nodes: tuple[str, ...] = ()
    input_paths: Mapping | None = None
    display_state: Mapping | None = None
    backend_default: str = 'native'
    node_backends: Mapping | None = None
    foreground_regions: Mapping | None = None

    def __post_init__(self):
        if self.input_paths is not None and not isinstance(self.input_paths, Mapping):
            raise ValueError("Contextual inputs must be a mapping.")
        object.__setattr__(self, "input_paths", MappingProxyType(dict(self.input_paths or {})))
        if self.display_state is not None and not isinstance(self.display_state, Mapping):
            raise ValueError("Workflow display state must be a mapping.")
        object.__setattr__(self, "display_state", MappingProxyType(dict(self.display_state or {})))
        if self.node_backends is not None and not isinstance(self.node_backends, Mapping):
            raise ValueError("Workflow node backends must be a mapping.")
        object.__setattr__(self, "node_backends", MappingProxyType(dict(self.node_backends or {})))
        if self.foreground_regions is not None and not isinstance(self.foreground_regions, Mapping):
            raise ValueError('Foreground regions must be a node-to-bounds mapping.')
        object.__setattr__(self, 'foreground_regions', MappingProxyType({
            key: tuple(bounds) if type(bounds) in (list, tuple) else bounds
            for key, bounds in (self.foreground_regions or {}).items()}))
        validate_workflow(self)

    def backend_for(self, stage_id):
        if stage_id not in {stage.stage_id for stage in self.recipe.stages}:
            raise ValueError('Backend node must belong to this workflow.')
        return self.node_backends.get(stage_id, self.backend_default)

    def to_dict(self):
        value = {'source_recipe': self.source_recipe.to_dict(), 'recipe': self.recipe.to_dict(),
                'disabled_nodes': list(self.disabled_nodes), 'input_paths': dict(self.input_paths),
                'display_state': dict(self.display_state)}
        # Keep native-only historical payloads byte-compatible at this seam.
        if self.backend_default != 'native' or self.node_backends:
            value.update(backend_default=self.backend_default, node_backends=dict(self.node_backends))
        if self.foreground_regions:
            value['foreground_regions'] = {key: list(bounds) for key, bounds in self.foreground_regions.items()}
        return value

    def to_json(self):
        import json
        return json.dumps(self.to_dict(), allow_nan=False)

    @classmethod
    def from_json(cls, value):
        import json
        from seedvision.benchmarking.recipes import MAX_RECIPE_BYTES, _reject_duplicate_pairs, _reject_constant
        if not isinstance(value, (str, bytes)) or len(value) > 2 * MAX_RECIPE_BYTES:
            raise ValueError('Working workflow JSON exceeds its bound.')
        return cls.from_dict(json.loads(value, object_pairs_hook=_reject_duplicate_pairs, parse_constant=_reject_constant))

    @classmethod
    def from_dict(cls, value):
        import json
        if (type(value) is not dict or set(value) - {'source_recipe', 'recipe', 'disabled_nodes', 'input_paths', 'display_state', 'backend_default', 'node_backends', 'foreground_regions'}
                or not {'source_recipe', 'recipe', 'disabled_nodes', 'input_paths'} <= set(value)):
            raise ValueError('Invalid working workflow fields.')
        if type(value.get('display_state', {})) is not dict:
            raise ValueError('Workflow display state must be a JSON object.')
        if type(value['input_paths']) is not dict:
            raise ValueError('Contextual inputs must be a JSON object.')
        if type(value['disabled_nodes']) is not list:
            raise ValueError('Disabled workflow nodes must be an array.')
        if type(value.get('node_backends', {})) is not dict:
            raise ValueError('Workflow node backends must be a JSON object.')
        return cls(RecipeSpec.from_json(json.dumps(value['source_recipe'])),
                   RecipeSpec.from_json(json.dumps(value['recipe'])), tuple(value['disabled_nodes']), value['input_paths'], value.get('display_state', {}),
                   value.get('backend_default', 'native'), value.get('node_backends', {}), value.get('foreground_regions', {}))


def validate_workflow(workflow):
    from seedvision.benchmarking.native_recipe import _validate_recipe, NativeRecipeError
    source, recipe = workflow.source_recipe, workflow.recipe
    if type(workflow.backend_default) is not str or workflow.backend_default not in ('native', 'plantcv'):
        raise NativeRecipeError('Workflow backend default must be native or plantcv.')
    _validate_source(source)
    if not isinstance(recipe, RecipeSpec) or (recipe.recipe_id, recipe.recipe_version) != (source.recipe_id, source.recipe_version):
        raise NativeRecipeError('Working recipe must retain its source preparation identity.')
    known_backends = {stage.stage_id for stage in recipe.stages}
    if (not isinstance(workflow.node_backends, Mapping)
            or any(type(key) is not str or key not in known_backends or type(value) is not str or value not in ('native', 'plantcv')
                   for key, value in workflow.node_backends.items())):
        raise NativeRecipeError('Workflow backend overrides must name registered stages and backends.')
    eligible = {stage.stage_id for stage in recipe.stages if stage.operation_id in FOREGROUND_OPERATIONS}
    for stage_id, bounds in workflow.foreground_regions.items():
        if (stage_id not in eligible or type(bounds) is not tuple or len(bounds) != 4
                or any(type(v) is not int or v < 0 for v in bounds)
                or bounds[0] >= bounds[2] or bounds[1] >= bounds[3]):
            raise NativeRecipeError('Foreground regions require eligible nodes and nonempty half-open integer bounds.')
    expected_provenance = _working_provenance(source) if recipe.dependencies != source.dependencies or workflow.disabled_nodes or workflow.foreground_regions else source.provenance
    if recipe.provenance != expected_provenance:
        raise NativeRecipeError('Working recipe must carry its separate enhanced provenance.')
    if {s.stage_id for s in recipe.stages} != {s.stage_id for s in source.stages}:
        raise NativeRecipeError('Working graph must retain the registered source operations.')
    for stage in recipe.stages:
        canonical = source.stage(stage.stage_id)
        expected_condition = BranchCondition('workspace.disabled', 'User selected typed pass-through', 'bypass') if stage.stage_id in workflow.disabled_nodes else canonical.condition
        if stage.condition != expected_condition:
            raise NativeRecipeError('Working bypass condition differs from the selected node state.')
        normalized = replace(stage, condition=canonical.condition, parameters=tuple(replace(p, value=c.value)
            for p, c in zip(stage.parameters, canonical.parameters, strict=True)))
        if normalized != canonical:
            raise NativeRecipeError('Working operation, ports, source or parameter contracts are unregistered.')
    if type(workflow.disabled_nodes) is not tuple or len(set(workflow.disabled_nodes)) != len(workflow.disabled_nodes):
        raise NativeRecipeError('Disabled nodes must be a unique immutable tuple.')
    for identifier in workflow.disabled_nodes:
        if identifier not in {s.stage_id for s in recipe.stages} or not bypass_ports(recipe.stage(identifier)):
            raise NativeRecipeError('Node does not have an unambiguous typed bypass.')
    if not isinstance(workflow.input_paths, Mapping) or set(workflow.input_paths) - {'target_rgb', 'target_mask', 'source_rgb', 'source_mask'}:
        raise NativeRecipeError('Unknown contextual correction input.')
    if any(type(v) is not str or not v.strip() or len(v) > 4096 or '\0' in v for v in workflow.input_paths.values()):
        raise NativeRecipeError('Contextual input paths must be bounded text.')
    if workflow.input_paths and recipe.recipe_id != 'plantcv_color_card_correct_color':
        raise NativeRecipeError('Only four-input correction accepts contextual file bindings.')
    if set(workflow.display_state) - {'selected_node', 'selected_output', 'image_visible', 'pipeline_visible', 'pipeline_name', 'context_layout'}:
        raise NativeRecipeError('Unknown workflow display state field.')
    for name, value in workflow.display_state.items():
        if name == 'context_layout':
            from seedvision.pipeline.project_context import validate_context_layout
            validate_context_layout(value)
        elif name in ('image_visible', 'pipeline_visible'):
            if type(value) is not bool:
                raise NativeRecipeError('Workflow view visibility must be boolean.')
        elif name == 'selected_node':
            from seedvision.pipeline.project_context import CONTEXT_NODE_IDS
            if type(value) is not str or value not in {s.stage_id for s in recipe.stages} | CONTEXT_NODE_IDS:
                raise NativeRecipeError('Selected workflow node must be registered.')
        elif name == 'pipeline_name':
            if type(value) is not str or not value.strip() or value != value.strip() or len(value) > 256 or '\0' in value:
                raise NativeRecipeError('Custom pipeline name must be bounded nonempty text.')
        elif type(value) is not str or len(value) > 256 or '\0' in value:
            raise NativeRecipeError('Selected workflow output must be bounded text.')
    # RecipeSpec construction enforces typed semantics, required sockets and a DAG.
    RecipeSpec.from_json(recipe.to_json())
    return workflow


def create_workflow(recipe):
    from seedvision.benchmarking.native_recipe import _validate_recipe
    _validate_source(recipe)
    return PipelineWorkflow(recipe, recipe)


def with_foreground_region(workflow, stage_id, bounds):
    regions = dict(workflow.foreground_regions)
    if bounds is None:
        regions.pop(stage_id, None)
    else:
        regions[stage_id] = bounds
    enhanced = bool(regions or workflow.disabled_nodes or workflow.recipe.dependencies != workflow.source_recipe.dependencies)
    provenance = _working_provenance(workflow.source_recipe) if enhanced else workflow.source_recipe.provenance
    return replace(workflow, recipe=replace(workflow.recipe, provenance=provenance), foreground_regions=regions)


def workflow_graph(workflow):
    validate_workflow(workflow)
    graph = to_pipeline_graph(workflow.recipe)
    for stage in workflow.recipe.stages:
        node = graph.node(stage.stage_id)
        node.implemented = True
        node.bypassable = bool(bypass_ports(stage))
        node.enabled = stage.stage_id not in workflow.disabled_nodes
        node.parameters = {p.parameter_id: p.value for p in stage.parameters}
        node.default_parameters = dict(node.parameters)
        node.status = NodeStatus.IDLE if node.enabled else NodeStatus.BYPASSED
        node.status_detail = 'Not run' if node.enabled else 'Typed pass-through'
    # Every alternative socket pair is checked against the immutable registered
    # port semantics, so canvas rewiring and CUDA dispatch share one contract.
    graph.connection_templates = tuple(PipelineConnection(a.stage_id, b.stage_id, p.data_type, p.name, q.name)
        for a in workflow.recipe.stages for b in workflow.recipe.stages if a != b
        for p in a.ports if p.direction == 'output'
        for q in b.ports if q.direction == 'input' and p.data_type == q.data_type and p.semantics == q.semantics)
    return graph


def workflow_from_graph(workflow, graph):
    from seedvision.pipeline.project_context import calculation_graph
    workspace = graph
    graph = calculation_graph(graph)
    if set(graph.nodes) != {s.stage_id for s in workflow.recipe.stages}:
        raise ValueError('Restore registered nodes before saving or running a workflow.')
    recipe = replace(workflow.recipe,
        stages=tuple(replace(stage, condition=(BranchCondition('workspace.disabled', 'User selected typed pass-through', 'bypass') if not graph.node(stage.stage_id).enabled else workflow.source_recipe.stage(stage.stage_id).condition), parameters=tuple(replace(p, value=graph.node(stage.stage_id).parameters[p.parameter_id])
            for p in stage.parameters)) for stage in workflow.recipe.stages),
        dependencies=tuple(DependencyContract(e.source, e.source_port, e.target, e.target_port, e.data_type)
            for e in graph.connections),
        layout=tuple(LayoutPosition(n.identifier, n.x, n.y) for n in graph.nodes.values()))
    disabled = tuple(n.identifier for n in graph.nodes.values() if not n.enabled)
    provenance = _working_provenance(workflow.source_recipe) if disabled or recipe.dependencies != workflow.source_recipe.dependencies or workflow.foreground_regions else workflow.source_recipe.provenance
    result = PipelineWorkflow(workflow.source_recipe, replace(recipe, provenance=provenance), disabled,
        workflow.input_paths, workflow.display_state, workflow.backend_default, workflow.node_backends, workflow.foreground_regions)
    graph.reference_recipe_identity = result.recipe.identity
    graph.reference_recipe_mode = result.recipe.provenance.mode.value
    workspace.reference_recipe_identity = result.recipe.identity
    workspace.reference_recipe_mode = result.recipe.provenance.mode.value
    return result


def run_workflow_path(workflow, path, *, cache=None, cancel=None, target_node_id=None, **kwargs):
    validate_workflow(workflow)
    from seedvision.benchmarking.native_recipe import run_native_recipe_path
    if workflow.recipe.recipe_id == 'plantcv_color_card_correct_color':
        raise ValueError('Four-input correction requires its contextual file or saved project binding adapter.')
    return run_native_recipe_path(workflow.recipe, path, cache=cache, cancel=cancel,
        working_workflow=workflow, target_node_id=target_node_id, **kwargs)


def _validate_source(recipe):
    if not isinstance(recipe, RecipeSpec):
        raise ValueError('Select a supported RecipeSpec source.')
    if recipe.recipe_id == 'plantcv_color_card_correct_color':
        from seedvision.benchmarking.native_correction_recipe import _validate_recipe
    else:
        from seedvision.benchmarking.native_recipe import _validate_recipe
    _validate_recipe(recipe)
