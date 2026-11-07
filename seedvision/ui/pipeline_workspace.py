"""Integrate standalone files and the default-pipeline lock into the existing desktop workspace."""
from seedvision.persistence.analysis_settings import (
    analysis_settings_profile_from_graph, apply_analysis_settings_profile,
)


def initialize_pipeline_files(window):
    from seedvision.persistence.pipeline_documents import PipelineDocumentStore
    from seedvision.ui.pipeline_documents import PipelineDocumentsController
    window._loaded_pipeline_document = None
    window.pipeline_documents = PipelineDocumentsController(
        window, capture=lambda name: capture_pipeline(window, name),
        install=lambda document: install_pipeline(window, document),
        store=PipelineDocumentStore(window._root))
    documents = window.pipeline_documents
    documents.lock_changed.connect(lambda locked: apply_lock(window, locked))
    documents.named.connect(lambda name: pipeline_named(window, name))
    documents.copies_changed.connect(window.pipeline_workflow.sync_copies)
    window.pipeline_workflow.rename_button.clicked.connect(lambda checked=False: documents.rename_current())
    window.pipeline_workflow.load_button.clicked.connect(lambda checked=False: documents.choose_pipeline())
    window.pipeline_workflow.save_button.clicked.connect(lambda checked=False: documents.choose_pipeline_save())
    window.pipeline_inspector.lock_notice.linkActivated.connect(lambda link: documents.offer_copy())
    window.pipeline_canvas.set_edit_guard(window._pipeline_editing_allowed)
    documents.loaded(window.pipeline_workflow.graph_pipeline.title)


def apply_lock(window, locked):
    window.pipeline_canvas.set_editing_locked(locked)
    window.pipeline_inspector.set_editing_locked(locked)
    node_id = window.catalogue_nodes._selected
    if window.catalogue_nodes.handles(node_id):
        window.catalogue_nodes.select_node(node_id)
    window.pipeline_workflow.sync_buttons()


def pipeline_named(window, name):
    """Record the working pipeline's custom name, which a project saves with it."""
    del name
    window._set_project_dirty()


def load_graph_pipeline(window, pipeline):
    """Make the shared graph this default pipeline; only the nodes that differ recalculate."""
    from seedvision.persistence.pipeline_documents import PipelineLayoutDocument
    factory = pipeline.build()
    result = apply_analysis_settings_profile(window.pipeline, analysis_settings_profile_from_graph(factory))
    if result.analytical_changed:
        for key in set(window._analysis_caches) | set(window._analyses):
            window._cache_dirty_nodes.setdefault(key, set()).update(result.affected_node_ids)
            window._analyses.pop(key, None)
    PipelineLayoutDocument.from_graph(factory).apply(window.pipeline)


def leave_copy(window, pipeline=None):
    """Clear what a copy put on the shared graph before a default pipeline takes over.

    ``pipeline`` is the one taking over the graph; by default, the one the copy came from.
    """
    window.catalogue_nodes.set_instances(())
    load_graph_pipeline(window, pipeline or window.pipeline_workflow.graph_pipeline)
    # A factory layout is arranged fresh, as at startup.
    window.pipeline_canvas.rebuild_from_graph(window._selected_pipeline_node)
    window.pipeline_canvas.auto_arrange()


def capture_pipeline(window, name):
    from seedvision.persistence.pipeline_documents import PipelineDocument
    return PipelineDocument(name, analysis_settings_profile_from_graph(window.pipeline),
        node_instances=window.catalogue_nodes.snapshot_instances(),
        layout=window.pipeline_documents.capture_layout())


def install_pipeline(window, document):
    """Validate a complete detached graph before replacing calculation state."""
    from seedvision.pipeline import build_default_pipeline
    from seedvision.pipeline.catalogue_nodes import instance_graph
    default = build_default_pipeline()
    apply_analysis_settings_profile(default, document.analysis_settings)
    for instance in document.node_instances:
        instance_graph(instance)
    # No user image, annotation, unresolved project record or historical run is replaced.
    window.catalogue_nodes.set_instances(())
    result = apply_analysis_settings_profile(window.pipeline, document.analysis_settings)
    if result.analytical_changed:
        for key in set(window._analysis_caches) | set(window._analyses):
            window._cache_dirty_nodes.setdefault(key, set()).update(result.affected_node_ids)
            window._analyses.pop(key, None)
    window.pipeline_workflow.install()
    window.catalogue_nodes.set_instances(document.node_instances)
    window._loaded_pipeline_document = document
    window.pipeline_documents.apply_layout(document.layout)
    window._set_project_dirty()
    return True
