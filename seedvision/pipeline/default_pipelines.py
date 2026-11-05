"""The default pipelines built on the shared calculation graph.

Each is the whole node catalogue with the nodes it does not use kept in the
unused-node toolbox, so projects, settings files and Add node see one catalogue
whichever of them a graph came from.
"""
from dataclasses import dataclass
from typing import Callable

from seedvision.pipeline.model import NodeStatus, PipelineGraph, build_default_pipeline

CELLPOSE_NODE = 'cellpose_sam_instances'
PROCEDURAL_NODE = 'procedural_instances'


def nodes_needed_by(graph, identifier):
    """``identifier`` and every node it depends on, directly or indirectly."""
    needed, pending = {identifier}, [identifier]
    while pending:
        for parent in graph.upstream(pending.pop()):
            if parent not in needed:
                needed.add(parent)
                pending.append(parent)
    return needed


def build_procedural_pipeline() -> PipelineGraph:
    """Seed analysis by procedural instance separation, without Cellpose-SAM."""
    graph = build_default_pipeline()
    graph.shelve_node(CELLPOSE_NODE, record_revision=False)
    return graph


def build_cellpose_pipeline() -> PipelineGraph:
    """Cellpose-SAM and only the nodes it needs."""
    graph = build_default_pipeline()
    needed = nodes_needed_by(graph, CELLPOSE_NODE)
    for identifier in tuple(graph.nodes):
        if identifier in needed:
            continue
        node = graph.node(identifier)
        graph.shelve_node(identifier, record_revision=False)
        node.enabled = False
        node.status = NodeStatus.BYPASSED
        node.status_detail = 'Disabled in the unused-node toolbox'
    cellpose = graph.node(CELLPOSE_NODE)
    cellpose.enabled = True
    cellpose.status = NodeStatus.IDLE
    cellpose.status_detail = 'Not run'
    return graph


@dataclass(frozen=True)
class GraphPipeline:
    """A default pipeline that runs on the shared graph rather than as a recipe."""

    recipe_id: str
    title: str
    build: Callable[[], PipelineGraph]
    selected_node: str


PROCEDURAL_SEED_ANALYSIS = GraphPipeline('procedural_seed_analysis', 'Procedural seed analysis',
    build_procedural_pipeline, 'seed_scale_estimation')
CELLPOSE_SEED_ANALYSIS = GraphPipeline('cellpose_seed_analysis', 'Cellpose seed analysis',
    build_cellpose_pipeline, CELLPOSE_NODE)
GRAPH_PIPELINES = (PROCEDURAL_SEED_ANALYSIS, CELLPOSE_SEED_ANALYSIS)


def graph_pipeline_of(active_nodes) -> GraphPipeline:
    """The default pipeline a graph with these active nodes is, or was copied from."""
    return PROCEDURAL_SEED_ANALYSIS if PROCEDURAL_NODE in active_nodes else CELLPOSE_SEED_ANALYSIS
