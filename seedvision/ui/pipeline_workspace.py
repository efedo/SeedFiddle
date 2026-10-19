"""Integrate standalone files and edit locks into the existing desktop workspace."""
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import os
import tempfile

from seedvision.persistence.analysis_settings import (
    analysis_settings_profile_from_graph, apply_analysis_settings_profile,
)


def initialize_pipeline_files(window):
    from seedvision.persistence.pipeline_documents import PipelineDocumentStore
    from seedvision.ui.pipeline_documents import PipelineDocumentsController
    from seedvision.ui.pipeline_backends import PipelineBackendsController
    window._loaded_pipeline_document = None
    window._pipeline_held_model_assets = ()
    window._pipeline_held_correction_bindings = ()
    window.pipeline_documents = PipelineDocumentsController(
        window, capture=lambda name: capture_pipeline(window, name),
        install=lambda document: install_pipeline(window, document),
        store=PipelineDocumentStore(window._root,
            factory_names=tuple(recipe.title for recipe in window.pipeline_workflow._all_templates)))
    window.pipeline_backends = PipelineBackendsController(window)
    documents = window.pipeline_documents
    window.pipeline_workflow.strip.layout().insertWidget(2, documents.freeze_button)
    menu = window.pipeline_workflow.more_button.menu()
    first = menu.actions()[0] if menu.actions() else None
    for action in (documents.load_pipeline_action, documents.save_pipeline_action,
                   documents.load_layout_action, documents.save_layout_action):
        menu.insertAction(first, action)
    documents.frozen_changed.connect(lambda frozen: apply_freeze(window, frozen))
    documents.pipeline_saved.connect(lambda document, path: pipeline_saved(window, document))
    documents.loaded('Default seed analysis')


def apply_freeze(window, frozen):
    window.pipeline_canvas.set_editing_locked(frozen)
    window.pipeline_inspector.set_editing_locked(frozen)
    node_id = window.catalogue_nodes._selected
    if window.catalogue_nodes.handles(node_id):
        window.catalogue_nodes.select_node(node_id)
    elif window.pipeline_workflow.active:
        window.pipeline_workflow.select_node(window.pipeline_workflow._selected)
    window.pipeline_workflow.sync_buttons()


def _union(existing, incoming):
    result = {value.key: value for value in existing}
    for value in incoming:
        old = result.get(value.key)
        if old is not None and old != value:
            raise ValueError('Imported pipeline assets conflict with retained project references.')
        result[value.key] = value
    return tuple(result.values())


def pipeline_saved(window, document):
    """Record a custom name only after the file has been published successfully."""
    presenter = window.pipeline_workflow
    if presenter.active:
        presenter.workflow = replace(presenter.workflow,
            display_state=dict(presenter.workflow.display_state) | {'pipeline_name': document.name})
    window._set_project_dirty()


def capture_pipeline(window, name):
    from seedvision.persistence.pipeline_documents import PipelineDocument
    workflow = window.pipeline_workflow.sync_graph()
    if workflow is not None:
        workflow = replace(workflow, display_state=dict(workflow.display_state) | {'pipeline_name': name})
    models = _union(tuple(window._project_reference_model_assets), window._pipeline_held_model_assets)
    bindings = _union(tuple(window._project_correction_bindings), window._pipeline_held_correction_bindings)
    document = PipelineDocument(name, analysis_settings_profile_from_graph(window.pipeline),
        pipeline_workflow=workflow, node_instances=window.catalogue_nodes.snapshot_instances(),
        layout=window.pipeline_documents.capture_layout(), reference_model_assets=models,
        correction_bindings=bindings, asset_root=str(window._root.resolve()),
        backend_default=window.pipeline_backends.default)
    return document


def _import_assets(document, root):
    """Copy only matching bytes into canonical relative asset locations."""
    origin = Path(document.asset_root).resolve() if document.asset_root else root
    if origin == root:
        return
    selections = [(asset.path, asset.source_sha256, asset.source_bytes)
                  for asset in document.reference_model_assets]
    selections.extend((asset.path, asset.source_sha256, asset.source_bytes)
                      for binding in document.correction_bindings for asset in binding.inputs)
    for relative, digest, size in dict.fromkeys(selections):
        if relative is None:
            continue
        source, destination = (origin / relative).resolve(), (root / relative).resolve()
        if not source.is_relative_to(origin) or not destination.is_relative_to(root):
            raise ValueError('Pipeline asset locator is outside its recorded workspace.')
        if not source.is_file():
            # Keep the exact record; execution reports the missing asset.
            continue
        with source.open('rb') as stream:
            raw = stream.read(size + 1)
        if len(raw) != size or sha256(raw).hexdigest() != digest:
            raise ValueError('A referenced pipeline asset differs from its saved fingerprint.')
        if destination.exists():
            with destination.open('rb') as stream:
                previous = stream.read(size + 1)
            if previous != raw:
                raise ValueError('A pipeline asset conflicts with existing workspace bytes.')
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, filename = tempfile.mkstemp(prefix='.pipeline-asset-', dir=destination.parent)
        temporary = Path(filename)
        try:
            with os.fdopen(descriptor, 'wb') as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)


def install_pipeline(window, document):
    """Validate a complete detached graph before replacing calculation state."""
    from seedvision.pipeline import build_default_pipeline
    from seedvision.pipeline.workflows import workflow_graph
    from seedvision.pipeline.catalogue_nodes import instance_graph
    from seedvision.persistence.reference_models import _requirements, validate_reference_model_assets
    from seedvision.persistence.correction_inputs import validate_correction_bindings
    from seedvision.pipeline.backend_execution import validate_backend_selection
    # Standalone execution overrides use the same stable IDs as layout files.
    workflow = document.pipeline_workflow
    overrides = dict(document.node_backends)
    if workflow is not None:
        own = {stage.stage_id for stage in workflow.recipe.stages}
        values = dict(workflow.node_backends) | {key: value for key, value in overrides.items() if key in own}
        workflow = replace(workflow, backend_default=document.backend_default, node_backends=values)
        validate_backend_selection(workflow)
    instances = []
    from seedvision.pipeline.catalogue_nodes import display_id
    handled = set() if workflow is None else {stage.stage_id for stage in workflow.recipe.stages}
    for instance in document.node_instances:
        if instance.workflow is not None:
            own = {display_id(instance, stage.stage_id): stage.stage_id for stage in instance.workflow.recipe.stages}
            values = dict(instance.workflow.node_backends) | {own[key]: value for key, value in overrides.items() if key in own}
            chosen = replace(instance.workflow, backend_default=document.backend_default, node_backends=values)
            validate_backend_selection(chosen)
            instance = replace(instance, workflow=chosen)
            handled.update(own)
        instances.append(instance)
    if any(value != 'native' for key, value in overrides.items() if key not in handled):
        raise ValueError('The default analysis nodes only support native execution.')
    document = replace(document, pipeline_workflow=workflow, node_instances=tuple(instances))
    default = build_default_pipeline()
    apply_analysis_settings_profile(default, document.analysis_settings)
    if document.pipeline_workflow is not None:
        workflow_graph(document.pipeline_workflow)
    for instance in document.node_instances:
        instance_graph(instance)
    requirements = {(recipe.recipe_id, recipe.recipe_version, stage.stage_id, digest)
                    for recipe, stage, selection, digest in _requirements(
                        window._project_reference_graphs, window._project_native_recipes,
                        document.pipeline_workflow, document.node_instances)}
    previous_models = _union(window._project_reference_model_assets, window._pipeline_held_model_assets)
    previous_bindings = _union(window._project_correction_bindings, window._pipeline_held_correction_bindings)
    all_models, all_bindings = document.merge_asset_records(previous_models, previous_bindings)
    models = tuple(asset for asset in all_models if asset.key in requirements)
    validate_reference_model_assets(models, window._project_reference_graphs, window._project_native_recipes,
        pipeline_workflow=document.pipeline_workflow, node_instances=document.node_instances)
    records = {record.identifier: record for record in window._project_original_image_records.values()}
    records.update({record.identifier: record for record in window._project_unresolved_image_records.values()})
    images = tuple(records.values())
    image_ids = {image.identifier for image in images}
    imported = []
    for binding in all_bindings:
        if binding.image_id not in image_ids:
            continue
        try:
            validate_correction_bindings((binding,), images, window._project_reference_graphs,
                window._project_native_recipes, pipeline_workflow=document.pipeline_workflow,
                node_instances=document.node_instances)
        except ValueError:
            continue
        imported.append(binding)
    bindings = tuple(imported)
    _import_assets(document, window._root.resolve())
    # No user image, annotation, unresolved project record or historical run is replaced.
    window.catalogue_nodes.set_instances(())
    result = apply_analysis_settings_profile(window.pipeline, document.analysis_settings)
    if result.analytical_changed:
        for key in set(window._analysis_caches) | set(window._analyses):
            window._cache_dirty_nodes.setdefault(key, set()).update(result.affected_node_ids)
            window._analyses.pop(key, None)
    window._project_reference_model_assets = models
    window._project_correction_bindings = bindings
    window._pipeline_held_model_assets = tuple(asset for asset in all_models if asset.key not in requirements)
    active_keys = {binding.key for binding in bindings}
    window._pipeline_held_correction_bindings = tuple(binding for binding in all_bindings if binding.key not in active_keys)
    window.pipeline_backends.default = document.backend_default
    window.pipeline_workflow.install(document.pipeline_workflow)
    window.catalogue_nodes.set_instances(document.node_instances)
    window._loaded_pipeline_document = document
    window.pipeline_documents.apply_layout(document.layout)
    window.pipeline_backends.refresh()
    window._set_project_dirty()
    return True
