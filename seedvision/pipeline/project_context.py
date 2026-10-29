"""Shared project inputs around an immutable calculation recipe."""
from copy import copy
import math

from seedvision.pipeline.model import PipelineConnection, PipelineGraph, PipelineNode


CONTEXT_NODE_IDS = frozenset(('project', 'image_selector', 'metadata', 'species_reference_library'))
CONTEXT_CONNECTIONS = (
    PipelineConnection('project', 'image_selector', 'ProjectImages', 'images', 'project_images'),
    PipelineConnection('image_selector', 'metadata', 'RawImage', 'raw_image', 'image'),
    PipelineConnection('project', 'species_reference_library', 'SpeciesLibraryPin', 'species_library', 'pin'),
    PipelineConnection('metadata', 'species_reference_library', 'SpeciesMetadata', 'species', 'species'),
)


def context_nodes():
    """Fresh mutable nodes; project ownership and image selection are distinct."""
    return (
        PipelineNode('project', 'Inputs', 'Inputs',
            'Project images, image-local annotations and the species-library pin', 0, 40,
            details='Owns the image inventory, annotations and exact species-library pin. '
                'Image selector chooses the current raw image from this inventory.',
            output_ports=(('images', 'Images'), ('annotations', 'Annotations'),
                          ('species_library', 'Species library pin')),
            output_port_types={'images': 'ProjectImages', 'annotations': 'ImageAnnotations',
                               'species_library': 'SpeciesLibraryPin'}),
        PipelineNode('image_selector', 'Image selector', 'Inputs',
            'Current raw image, synchronized with the Images list', 320, 40,
            details='Choose an image here or in the Images list. Both select the same source '
                'file; switching images cancels obsolete work and replaces image-local outputs.',
            output_ports=(('raw_image', 'Raw image'),),
            output_port_types={'raw_image': 'RawImage'}, status_detail='No image selected'),
        PipelineNode('metadata', 'Species and metadata', 'Inputs', 'Species, lot and notes', 640, -200,
            details='Metadata belongs to the image selected above. Use the Species control '
                'in Image metadata to change its species.',
            output_ports=(('species', 'Species'),), output_port_types={'species': 'SpeciesMetadata'},
            status_detail='Choose species'),
        PipelineNode('species_reference_library', 'Species reference library', 'Inputs',
            'Resolve the exact species-library pin and exclude the current image', 960, 330,
            details='Checks the exact library version, species, trait vocabulary and descriptors, '
                'then excludes contributions from the current image. Banks affect only nodes '
                'connected to their outputs; PlantCV and SMPTS recipes do not implicitly use them.',
            output_ports=tuple((key, title) for key, title, _ in LIBRARY_PORTS),
            output_port_types={key: kind for key, _, kind in LIBRARY_PORTS},
            status_detail='No species library pinned'),
    )


LIBRARY_PORTS = (
    ('dimensions_shape', 'Dimensions and shape', 'SpeciesDimensionsShapeBank'),
    ('foreground_colour', 'Foreground colour', 'SpeciesForegroundColourBank'),
    ('foreground_noise', 'Foreground noise', 'SpeciesForegroundNoiseBank'),
    ('material_prototypes', 'Material prototypes', 'SpeciesMaterialPrototypeBank'),
    ('edge_prototypes', 'Edge prototypes', 'SpeciesEdgePrototypeBank'),
    ('seed_traits', 'Seed traits', 'SpeciesSeedTraitBank'),
)


def protect_context(graph, connections=CONTEXT_CONNECTIONS):
    graph.required_nodes = CONTEXT_NODE_IDS
    graph.fixed_connections = frozenset(connections)
    for identifier in CONTEXT_NODE_IDS:
        graph.node(identifier).bypassable = False


def validate_context_layout(value):
    if not isinstance(value, dict) or set(value) - CONTEXT_NODE_IDS:
        raise ValueError('Project-node layout must name shared context nodes.')
    for position in value.values():
        if (not isinstance(position, (tuple, list)) or len(position) != 2
                or any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 1e7 for v in position)):
            raise ValueError('Project-node positions must be finite canvas coordinates.')


def workspace_graph(workflow):
    """Add real current-project inputs without changing the factory RecipeSpec."""
    from seedvision.pipeline.workflows import workflow_graph
    graph = workflow_graph(workflow)
    context = PipelineGraph(context_nodes(), CONTEXT_CONNECTIONS)
    left = min((node.x for node in graph.nodes.values()), default=0) - 1280
    top = min((node.y for node in graph.nodes.values()), default=0) - 40
    layout = workflow.display_state.get('context_layout', {})
    for node in context.nodes.values():
        node.x, node.y = layout.get(node.identifier, (node.x + left, node.y + top))
    edges = list(context.connections)
    for stage in workflow.recipe.stages:
        # Four-input correction binds the selected source photograph directly;
        # other recipes prepare their RGB/grayscale tensors in external.image_inputs.
        if stage.operation_id == 'external.image_inputs' or workflow.recipe.recipe_id == 'plantcv_color_card_correct_color':
            node = graph.node(stage.stage_id)
            node.input_ports = (*node.input_ports, ('selected_image', 'Selected raw image'))
            node.input_port_types = dict(node.input_port_types) | {'selected_image': 'RawImage'}
            edges.append(PipelineConnection('image_selector', stage.stage_id, 'RawImage', 'raw_image', 'selected_image'))
    graph.nodes = dict(context.nodes) | graph.nodes
    graph.connections = (*edges, *graph.connections)
    graph.connection_templates = (*edges, *graph.connection_templates)
    graph.workspace_context = True
    protect_context(graph, edges)
    graph._validate_connections()
    graph.topological_order()
    return graph


def calculation_graph(graph):
    """Validate mandatory context, then isolate the editable calculation graph."""
    if not getattr(graph, 'workspace_context', False):
        return graph
    if not CONTEXT_NODE_IDS <= graph.nodes.keys() or any(not graph.node(key).enabled for key in CONTEXT_NODE_IDS):
        raise ValueError('Every pipeline requires its enabled project and image context nodes.')
    context_edges = {edge for edge in graph.connections if edge.source in CONTEXT_NODE_IDS or edge.target in CONTEXT_NODE_IDS}
    if context_edges != graph.fixed_connections:
        raise ValueError('Project and selected-image connections must remain connected.')
    result = copy(graph)
    result.nodes = {key: node for key, node in graph.nodes.items() if key not in CONTEXT_NODE_IDS}
    result.connections = tuple(edge for edge in graph.connections if edge not in context_edges)
    return result
