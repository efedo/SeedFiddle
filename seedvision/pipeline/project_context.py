"""Shared project inputs at the head of every pipeline graph."""
from seedvision.pipeline.model import PipelineConnection, PipelineNode


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
                'connected to their outputs.',
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
