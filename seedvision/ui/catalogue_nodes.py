"""Independent catalogue components hosted by the existing Pipeline workspace."""
from __future__ import annotations

from collections import deque
from copy import copy
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import Event
from hashlib import sha256

import numpy as np
import torch
from PySide6.QtCore import QObject, QRunnable, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QFileDialog, QPushButton

from seedvision.pipeline import NodeStatus
from seedvision.segmentation.baseline import PipelineAnalysisCache


class _Signals(QObject):
    finished = Signal(object)


class _Job(QRunnable):
    def __init__(self, function, token):
        super().__init__()
        self.function, self.token = function, token
        self.cancel = Event()
        self.signals = _Signals()

    def run(self):
        try:
            result = self.function(self.cancel.is_set)
        except Exception as error:
            result = error
        self.signals.finished.emit(result)


@dataclass
class _Component:
    instance: object
    graph: object
    cache: PipelineAnalysisCache = field(default_factory=PipelineAnalysisCache)
    result: object = None
    prepared: object = None
    generation: int = 0
    dirty: set = field(default_factory=set)
    result_source: object = None
    correction_context: object = None


class CatalogueNodesController(QObject):
    """Own added components without borrowing the primary graph's mutable state.

    Only display identifiers are namespaced. Each backend sees its validated
    canonical graph, private settings/cache and the current source context.
    Queued work waits for the shared pool, including a primary run scheduled in
    the same event-loop turn. No cross-component connection is advertised.
    """
    graph_changed = Signal(object)
    selected = Signal(str)
    state_changed = Signal()

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.primary = None
        self.graph = None
        self._components = {}
        self._selected = None
        self._selected_output = ''
        self._queue = deque()
        self._job = None
        self._generation = 0
        self._source = None
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._advance)

    @property
    def busy(self):
        return self.running or bool(self._queue)

    @property
    def running(self):
        return self._job is not None

    @property
    def instances(self):
        return tuple(component.instance for component in self._components.values())

    @property
    def node_instance_ids(self):
        return tuple(self._components)

    @property
    def selected_instance(self):
        component = self._component(self._selected)
        return None if component is None else component.instance

    @property
    def can_export_selected(self):
        component = self._component(self._selected)
        return component is not None and component.result is not None and not self.busy

    def clear_selection(self):
        self._selected = None
        self._selected_output = ''

    def handles(self, node_id):
        return self._component(node_id) is not None

    def _component(self, node_id):
        if not isinstance(node_id, str) or '::' not in node_id:
            return None
        component = self._components.get(node_id.split('::', 1)[0])
        return component if component is not None and node_id in component.graph.nodes else None

    @staticmethod
    def _canonical(component, identifier):
        if identifier not in component.graph.nodes:
            raise ValueError('Node is outside this independent calculation.')
        return identifier[len(component.instance.instance_id) + 2:]

    def menu_entries(self):
        from seedvision.pipeline.catalogue_nodes import catalogue_entries, _contract_key
        from seedvision.pipeline import build_default_pipeline
        default = build_default_pipeline()
        default_nodes = {**default.nodes, **default.unused_nodes}
        counts = {}
        def present(key):
            counts[key] = counts.get(key, 0) + 1
        presenter = getattr(self.window, 'pipeline_workflow', None)
        primary_workflow = getattr(presenter, 'workflow', None)
        if self.primary is not None:
            from seedvision.pipeline.project_context import CONTEXT_NODE_IDS
            for identifier in self.primary.nodes:
                if primary_workflow is None or identifier in CONTEXT_NODE_IDS:
                    present('default:' + identifier)
                else:
                    present('native:' + _contract_key(primary_workflow.source_recipe.stage(identifier)))
        for component in self._components.values():
            for identifier in component.graph.nodes:
                canonical = self._canonical(component, identifier)
                if component.instance.workflow is None:
                    present('default:' + canonical)
                else:
                    present('native:' + _contract_key(component.instance.workflow.source_recipe.stage(canonical)))
        result = []
        for entry in catalogue_entries():
            count = counts.get(entry.key, 0)
            available = entry.backend != 'default' or default_nodes[entry.target_node].implemented
            result.append(dict(key=entry.key, title=entry.title, category=entry.category,
                description=f'{entry.title} · {entry.backend} · add another independent instance, including required inputs and supported internal connection routes'
                    + ('' if available else ' · planned calculation; unavailable for execution'),
                present=bool(count), count=count, available=available))
        return tuple(result)

    def install_primary(self, primary_graph):
        self.primary = primary_graph
        return self._merge()

    def _merge(self):
        if self.primary is None:
            return None
        if not self._components:
            self.graph = self.primary
        else:
            # Keep the primary node objects so its existing control/layout/status
            # routes keep their exact behavior. Avoid re-populating their ports.
            combined = copy(self.primary)
            combined.nodes = dict(self.primary.nodes)
            combined.unused_nodes = dict(self.primary.unused_nodes)
            connections = list(self.primary.connections)
            templates = list(self.primary.connection_templates)
            for component in self._components.values():
                if combined.nodes.keys() & component.graph.nodes.keys():
                    raise ValueError('Catalogue instance identifiers must be unique.')
                combined.nodes.update(component.graph.nodes)
                connections.extend(component.graph.connections)
                templates.extend(component.graph.connection_templates)
            combined.connections = tuple(connections)
            combined.connection_templates = tuple(templates)
            combined._connection_suspended = set(self.primary._connection_suspended)
            for component in self._components.values():
                combined._connection_suspended.update(component.graph._connection_suspended)
            combined._validate_connections()
            combined.topological_order()
            self.graph = combined
        self.graph_changed.emit(self.graph)
        return self.graph

    def sync_primary(self):
        if self.primary is not None and self.graph is not self.primary:
            identifiers = set(self.primary.nodes) | set(self.primary.unused_nodes)
            nodes = {identifier: self.graph.nodes[identifier] for identifier in identifiers if identifier in self.graph.nodes}
            unused = {identifier: self.graph.unused_nodes[identifier] for identifier in identifiers if identifier in self.graph.unused_nodes}
            if set(nodes) | set(unused) != identifiers:
                raise ValueError('The primary graph must retain its registered nodes.')
            changed = tuple(edge for edge in self.graph.connections
                if edge.source in nodes and edge.target in nodes)
            unused_edges = tuple(edge for edge in self.graph.unused_connections
                if edge.source in identifiers and edge.target in identifiers)
            suspended = {identifier for identifier in self.graph._connection_suspended if identifier in identifiers}
            membership_changed = set(nodes) != set(self.primary.nodes)
            if (changed != self.primary.connections or unused_edges != self.primary.unused_connections
                    or membership_changed or suspended != self.primary._connection_suspended):
                previous = self.primary.connections
                old_nodes, old_unused = self.primary.nodes, self.primary.unused_nodes
                old_unused_edges = self.primary.unused_connections
                old_suspended = self.primary._connection_suspended
                self.primary.nodes, self.primary.unused_nodes = nodes, unused
                self.primary.connections = changed
                self.primary.unused_connections = unused_edges
                self.primary._connection_suspended = suspended
                try:
                    self.primary._validate_connections()
                    self.primary.topological_order()
                except Exception:
                    self.primary.connections = previous
                    self.primary.nodes, self.primary.unused_nodes = old_nodes, old_unused
                    self.primary.unused_connections = old_unused_edges
                    self.primary._connection_suspended = old_suspended
                    raise
                self.primary.revision = max(self.primary.revision + 1, self.graph.revision)
                targets = {edge.target for edge in set(previous).symmetric_difference(changed)}
                targets.update(set(nodes).symmetric_difference(old_nodes))
                affected = set(targets)
                for identifier in targets:
                    if identifier in nodes:
                        affected.update(self.primary.downstream(identifier, recursive=True))
                self.primary.invalidate(affected, preserve_bypassed=True)
        return self.primary

    def set_instances(self, instances):
        from seedvision.pipeline.catalogue_nodes import instance_graph, validate_node_instances
        validate_node_instances(instances)
        self.stop()
        components = {}
        for instance in instances:
            if instance.instance_id in components:
                raise ValueError('Catalogue instance identifiers must be unique.')
            components[instance.instance_id] = _Component(instance, instance_graph(instance))
        self._components = components
        selected = next((item for item in instances if item.selected_node is not None), None)
        self._selected = None if selected is None else selected.instance_id + '::' + selected.selected_node
        self._selected_output = '' if selected is None else selected.selected_output
        self._merge()
        if self._selected is not None:
            self.selected.emit(self._selected)
        self.state_changed.emit()

    def snapshot_instances(self):
        self.layout_changed()
        for component in self._components.values():
            selected = self._component(self._selected) is component
            component.instance = replace(component.instance,
                selected_node=self._canonical(component, self._selected) if selected else None,
                selected_output=self._selected_output if selected else '')
        return self.instances

    def add_entry(self, key):
        guard = getattr(self.window, "_pipeline_editing_allowed", None)
        if guard is not None and not guard():
            return False
        from seedvision.pipeline.catalogue_nodes import (create_instance, display_id,
            instance_graph, MAX_NODE_INSTANCES, MAX_NODE_INSTANCE_BYTES)
        if len(self._components) >= MAX_NODE_INSTANCES:
            self.window.statusBar().showMessage(f'A project supports at most {MAX_NODE_INSTANCES} independent node instances.')
            return None
        instance = create_instance(key, existing_ids=self._components)
        backends = getattr(self.window, "pipeline_backends", None)
        if backends is not None:
            instance = backends.configure_new_instance(instance)
        graph = instance_graph(instance)
        if self.graph is not None and not instance.layout:
            bottom = max((node.y for node in self.graph.nodes.values()), default=0) + 240
            top = min((node.y for node in graph.nodes.values()), default=0)
            for node in graph.nodes.values():
                node.y += bottom - top
        component = _Component(instance, graph)
        self._sync(component)
        import json
        payload = [item.instance.to_dict() for item in self._components.values()] + [component.instance.to_dict()]
        if len(json.dumps(payload, allow_nan=False).encode('utf-8')) > MAX_NODE_INSTANCE_BYTES:
            self.window.statusBar().showMessage('Independent node instances exceed the saved-project document size limit.')
            return None
        self._components[instance.instance_id] = component
        self._merge()
        identifier = display_id(instance, instance.target_node)
        self._selected = identifier
        self.window._set_project_dirty()
        self.selected.emit(identifier)
        self.state_changed.emit()
        return identifier

    def _sync(self, component):
        from seedvision.pipeline.catalogue_nodes import sync_instance
        projection = self.graph if self.graph is not None and all(
            identifier in self.graph.nodes for identifier in component.graph.nodes) else component.graph
        component.instance = sync_instance(component.instance, projection)
        component.graph.connections = tuple(edge for edge in projection.connections
            if edge.source in component.graph.nodes and edge.target in component.graph.nodes)
        component.graph._connection_suspended = set(projection._connection_suspended) & component.graph.nodes.keys()

    def layout_changed(self):
        for component in self._components.values():
            self._sync(component)

    def _changed(self, component, affected=()):
        component.generation += 1
        component.result = None
        component.result_source = None
        component.dirty.update(self._canonical(component, identifier) for identifier in affected)
        if self._job is not None and self._job.token[1] == component.instance.instance_id:
            self._job.cancel.set()
        self.window._set_project_dirty()
        self.state_changed.emit()
        if self.handles(self._selected) and self._component(self._selected) is component:
            self.select_node(self._selected)

    def parameter_changed(self, node_id, key, value):
        guard = getattr(self.window, "_pipeline_editing_allowed", None)
        if guard is not None and not guard():
            return False
        component = self._component(node_id)
        if component is None:
            return False
        node = component.graph.node(node_id)
        before = dict(node.parameters)
        try:
            affected = component.graph.set_parameter(node_id, key, value)
            if key == 'channel' and component.instance.workflow is not None:
                operation = component.instance.workflow.recipe.stage(self._canonical(component, node_id)).operation_id
                if operation == 'threshold.custom_range':
                    import json
                    count = 1 if str(value).upper() in ('GRAY', 'GREY') else 4 if str(value).upper() == 'CMYK' else 3
                    for bounds, fill in (('lower', 0), ('upper', 255)):
                        selected = node.parameters[bounds]
                        parsed = json.loads(selected) if isinstance(selected, str) else list(selected)
                        node.parameters[bounds] = json.dumps((parsed + [fill] * count)[:count])
            self._sync(component)
        except (ValueError, TypeError, KeyError) as error:
            node.parameters = before
            self.window.statusBar().showMessage(str(error))
        else:
            self._changed(component, affected)
        self.window.pipeline_inspector.set_node(node)
        self.window.pipeline_canvas.refresh()
        return True

    def reset_parameters(self, node_id):
        guard = getattr(self.window, "_pipeline_editing_allowed", None)
        if guard is not None and not guard():
            return False
        from seedvision.pipeline.catalogue_nodes import create_instance, instance_graph
        component = self._component(node_id)
        if component is None:
            return False
        fresh = create_instance(component.instance.catalogue_key)
        fresh = replace(fresh, instance_id=component.instance.instance_id)
        original = instance_graph(fresh).node(node_id)
        node = component.graph.node(node_id)
        previous = dict(node.parameters)
        node.parameters = dict(original.parameters)
        try:
            self._sync(component)
        except (ValueError, TypeError) as error:
            node.parameters = previous
            self.window.statusBar().showMessage(str(error))
            return True
        affected = (node_id, *component.graph.downstream(node_id, recursive=True))
        component.graph.invalidate(affected, preserve_bypassed=True)
        self._changed(component, affected)
        self.select_node(node_id)
        self.window.pipeline_canvas.refresh()
        return True

    def enabled_changed(self, node_id, enabled):
        guard = getattr(self.window, "_pipeline_editing_allowed", None)
        if guard is not None and not guard():
            return False
        component = self._component(node_id)
        if component is None:
            return False
        before = [(node, node.enabled) for node in component.graph.nodes.values()]
        try:
            if component.instance.workflow is not None:
                node = component.graph.node(node_id)
                if not enabled and not node.bypassable:
                    raise ValueError('This operation has no supported typed pass-through.')
                node.enabled = bool(enabled)
                affected = (node_id, *component.graph.downstream(node_id, recursive=True))
                component.graph.invalidate(affected, preserve_bypassed=True)
            else:
                affected = component.graph.set_enabled(node_id, bool(enabled))
            self._sync(component)
        except (ValueError, TypeError) as error:
            for node, state in before:
                node.enabled = state
            self.window.statusBar().showMessage(str(error))
        else:
            self._changed(component, affected)
        self.select_node(node_id)
        self.window.pipeline_canvas.refresh()
        return True

    def connections_changed(self, affected=None):
        guard = getattr(self.window, "_pipeline_editing_allowed", None)
        if guard is not None and not guard():
            return False
        from seedvision.pipeline.catalogue_nodes import instance_graph
        affected = set(affected or ())
        primary_affected = not affected or any(not self.handles(identifier) for identifier in affected)
        self.sync_primary()
        for component in self._components.values():
            if affected and not affected.intersection(component.graph.nodes):
                continue
            try:
                self._sync(component)
            except (ValueError, TypeError, KeyError) as error:
                component.graph = instance_graph(component.instance)
                self.window.statusBar().showMessage(str(error))
            else:
                self._changed(component, affected.intersection(component.graph.nodes))
        self._merge()
        return not primary_affected

    def remove_selected(self):
        guard = getattr(self.window, "_pipeline_editing_allowed", None)
        if guard is not None and not guard():
            return False
        component = self._component(self._selected)
        if component is None:
            return False
        if self._job is not None and self._job.token[1] == component.instance.instance_id:
            self._job.cancel.set()
        self._components.pop(component.instance.instance_id)
        self.window._project_correction_bindings = tuple(binding for binding in self.window._project_correction_bindings
            if getattr(binding, 'owner_id', None) != component.instance.instance_id)
        self._queue = deque(item for item in self._queue if item[0] != component.instance.instance_id)
        self._selected = None
        self._selected_output = ''
        self._merge()
        self.window._set_project_dirty()
        self.state_changed.emit()
        return True

    def run(self, node_id=None):
        if node_id is not None and not self.handles(node_id):
            return False
        self.layout_changed()
        if node_id is None:
            from seedvision.pipeline.catalogue_nodes import display_id
            requests = [(component.instance.instance_id, display_id(component.instance, component.instance.target_node))
                for component in self._components.values()]
        else:
            requests = [(self._component(node_id).instance.instance_id, node_id)]
        queued = set(self._queue)
        for identifier, target in requests:
            component = self._components[identifier]
            component.result = None
            component.result_source = None
            node = component.graph.node(target)
            node.status = NodeStatus.IDLE
            node.status_detail = 'Queued independent calculation'
        self._queue.extend(item for item in requests if item not in queued)
        if requests:
            if self.handles(self._selected):
                self.select_node(self._selected)
            self._timer.start(0)
            self.state_changed.emit()
        return bool(requests)

    def _advance(self):
        if self._job is not None or not self._queue:
            return
        if self.window._thread_pool.activeThreadCount():
            self._timer.start(50)
            return
        identifier, target = self._queue.popleft()
        component = self._components.get(identifier)
        if component is None:
            self._timer.start(0)
            return
        path = self.window.image_view.image_path
        if path is None:
            self._queue.clear()
            self.state_changed.emit()
            return
        try:
            if not component.graph.node(target).implemented:
                raise ValueError('This registered node is planned; no calculation is currently available.')
            if not component.graph.node(target).enabled:
                raise ValueError('Enable this independent node before running it.')
            if hasattr(self.window, '_check_source_revision') and not self.window._check_source_revision(path):
                raise ValueError('Reload the changed image before running this node.')
            function = self._prepare_run(component, target, Path(path))
        except (ValueError, TypeError, OSError) as error:
            component.graph.node(target).status = NodeStatus.BLOCKED
            component.graph.node(target).status_detail = str(error)
            self.window.statusBar().showMessage(str(error))
            self.window.pipeline_canvas.refresh()
            if self._component(self._selected) is component:
                self.select_node(self._selected)
            self._timer.start(0)
            self.state_changed.emit()
            return
        token = (self._generation, identifier, component.generation, str(path))
        job = _Job(function, token)
        job.context_fingerprint = self._context_fingerprint(Path(path)) if component.instance.workflow is None else None
        self._job = job
        job.signals.finished.connect(lambda result: self._completed(job, target, result))
        component.graph.node(target).status = NodeStatus.RUNNING
        component.graph.node(target).status_detail = 'Running independent instance'
        self.window.pipeline_canvas.setEnabled(False)
        self.window.pipeline_inspector.setEnabled(False)
        self.window.pipeline_canvas.refresh()
        self.state_changed.emit()
        self.window._thread_pool.start(job)

    def _prepare_run(self, component, target, path):
        from seedvision.pipeline.catalogue_nodes import default_graph
        instance = component.instance
        canonical_target = self._canonical(component, target)
        if instance.workflow is None:
            from seedvision.pipeline.settings import production_settings
            from seedvision.segmentation.baseline import analyze_path
            graph = default_graph(instance)
            scope = set()
            pending = [target]
            while pending:
                current = pending.pop()
                if current in scope or not component.graph.node(current).enabled:
                    continue
                scope.add(current)
                pending.extend(component.graph.upstream(current))
            settings = production_settings(graph)
            settings.update(self._default_context(path, graph))
            settings.update(enabled_nodes=frozenset(self._canonical(component, node) for node in scope),
                node_cache=component.cache, dirty_nodes=frozenset(component.dirty))
            return lambda cancel: analyze_path(path, cancellation_requested=cancel, **settings)
        workflow = instance.workflow
        if workflow.recipe.recipe_id == 'plantcv_color_card_correct_color':
            context = self.window._capture_correction_project_context()
            component.correction_context = context
            binding = next((item for item in self.window._project_correction_bindings
                if context is not None and item.image_id == context.image.identifier
                and item.recipe_id == workflow.recipe.recipe_id
                and item.recipe_sha256 == workflow.recipe.identity.recipe_sha256
                and getattr(item, 'owner_id', None) == instance.instance_id), None)
            from seedvision.persistence.correction_inputs import resolve_correction_asset_path
            paths = dict(workflow.input_paths)
            # A copied binding transports the same explicit source selections.
            # A newer selection always uses the file adapter; its chooser also
            # evicts this owner's old binding. Never borrow a primary/sibling run.
            same_selection = binding is not None and all(
                asset.role == 'source_rgb' or asset.role not in paths or
                str(Path(paths[asset.role]).resolve()).casefold() == str(Path(asset.origin_path).resolve()).casefold()
                for asset in binding.inputs)
            if same_selection and all(resolve_correction_asset_path(context.root, asset.path).is_file() for asset in binding.inputs):
                from seedvision.pipeline.catalogue_nodes import execute_bound_correction
                return lambda cancel: execute_bound_correction(instance, context.root, binding,
                    context.image, context.source_path, cache=component.cache, previous=component.prepared, cancel=cancel)
            paths['source_rgb'] = str(path)
            if set(paths) != {'target_rgb', 'target_mask', 'source_rgb', 'source_mask'}:
                raise ValueError('Select target image and both chip-label masks for this correction instance.')
            from seedvision.benchmarking.native_correction_inputs import run_native_correction_files
            return lambda cancel: run_native_correction_files(workflow.recipe, paths,
                cache=component.cache, previous=component.prepared, cancel=cancel, working_workflow=workflow)
        from seedvision.persistence.reference_models import model_asset_paths
        from seedvision.pipeline.workflows import run_workflow_path
        assets = model_asset_paths(self.window._root, workflow.recipe, self.window._project_reference_model_assets)
        return lambda cancel: run_workflow_path(workflow, path, cache=component.cache,
            cancel=cancel, target_node_id=canonical_target, model_asset_paths=assets)

    def _default_context(self, path, graph):
        from seedvision.ui.main_window import _path_identity
        window, key = self.window, _path_identity(path)
        def captured(name):
            value = getattr(window, name, {}).get(key)
            return None if value is None else np.asarray(value).copy()
        resolved = window._resolved_species_library_for_current_image(path)
        return dict(background_reference_points=(), foreground_reference_points=(),
            background_reference_mask=captured('_applied_background_reference_masks'),
            foreground_reference_mask=captured('_applied_foreground_reference_masks'),
            background_exclusion_mask=captured('_applied_background_exclusion_masks'),
            foreground_exclusion_mask=captured('_applied_foreground_exclusion_masks'),
            physical_edge_reference_mask=captured('_applied_physical_edge_reference_masks'),
            non_edge_reference_mask=captured('_applied_non_edge_reference_masks'),
            seed_instance_annotations=captured('_applied_instance_annotations'),
            seed_instance_traits=tuple(sorted(window._applied_seed_annotations.get(key, {}).values(), key=lambda item: item.seed_id)),
            seed_trait_species=window._applied_seed_annotation_species.get(key, ''),
            seed_trait_coat_patterns=window._current_seed_trait_vocabulary().coat_patterns,
            seed_trait_conditions=window._seed_trait_catalogue.conditions,
            manual_seed_centres=window._manual_seed_centres_for_analysis(key),
            background_colour_enabled=bool(graph.node('background_likelihood').parameters.get('background_colour_enabled', True)),
            learning_root=window._root, species=window.species_combo.currentText(),
            biological_context=window._effective_biological_context(),
            library_shape_bank=None if resolved.artifact is None else resolved.artifact.dimensions_shape,
            species_library=resolved.artifact,
            reference_transform=None if window._reference_transforms.get(key) is None else np.asarray(window._reference_transforms[key]).copy(),
            reference_archive_source_sha256=window._reference_source_identities.get(key, (None, None))[0],
            reference_annotation_snapshot_source_sha256=window._reference_source_identities.get(key, (None, None))[1])

    def _context_fingerprint(self, path):
        """Hash held CPU annotation/context facts, without transferring analysis rasters."""
        from seedvision.ui.main_window import _path_identity
        window, key = self.window, _path_identity(path)
        state = sha256()
        for name in ('_applied_background_reference_masks', '_applied_foreground_reference_masks',
                     '_applied_background_exclusion_masks', '_applied_foreground_exclusion_masks',
                     '_applied_physical_edge_reference_masks', '_applied_non_edge_reference_masks',
                     '_applied_instance_annotations', '_reference_transforms'):
            value = getattr(window, name, {}).get(key)
            state.update(name.encode('utf-8'))
            if value is not None:
                array = np.ascontiguousarray(value)
                state.update(repr((array.shape, array.dtype.str)).encode('utf-8'))
                state.update(memoryview(array).cast('B'))
        facts = (window.species_combo.currentText(), window._effective_biological_context(),
            getattr(window, '_project_species_library_pin', None),
            tuple(sorted(window._applied_seed_annotations.get(key, {}).values(), key=lambda item: item.seed_id)),
            window._applied_seed_annotation_species.get(key, ''),
            window._current_seed_trait_vocabulary().coat_patterns, window._seed_trait_catalogue.conditions,
            window._manual_seed_centres_for_analysis(key), window._reference_source_identities.get(key, (None, None)))
        state.update(repr(facts).encode('utf-8'))
        try:
            stat = path.stat()
            state.update(repr((stat.st_size, stat.st_mtime_ns)).encode('ascii'))
        except OSError:
            state.update(b'source-unavailable')
        return state.hexdigest()

    def _completed(self, job, target, result):
        if self._job is not job:
            return
        self._job = None
        component = self._components.get(job.token[1])
        current = component is not None and job.token == (self._generation,
            component.instance.instance_id, component.generation, str(self.window.image_view.image_path))
        if current and component.instance.workflow is None:
            current = getattr(job, 'context_fingerprint', None) == self._context_fingerprint(Path(job.token[3]))
        if current and not job.cancel.is_set():
            if isinstance(result, Exception):
                component.graph.node(target).status = NodeStatus.FAILED
                component.graph.node(target).status_detail = str(result)
                self.window.statusBar().showMessage(f'Independent node failed: {result}')
                if self._component(self._selected) is component:
                    self.select_node(self._selected)
            else:
                if component.instance.workflow is not None and component.instance.workflow.recipe.recipe_id == 'plantcv_color_card_correct_color':
                    result, component.prepared = result
                    context = component.correction_context
                    if context is not None:
                        from seedvision.persistence.correction_inputs import binding_for_result, select_correction_binding
                        try:
                            origins = getattr(result, 'input_origins', None)
                            binding = getattr(origins, 'binding', None)
                            if (binding is None or getattr(binding, 'owner_id', None) != component.instance.instance_id
                                    or binding.image_id != context.image.identifier
                                    or binding.recipe_sha256 != component.instance.workflow.recipe.identity.recipe_sha256):
                                binding = binding_for_result(component.instance.workflow.recipe,
                                    context.image.identifier, result, owner_id=component.instance.instance_id)
                            self.window._project_correction_bindings = select_correction_binding(
                                self.window._project_correction_bindings, binding)
                            self.window._set_project_dirty()
                        except (ValueError, OSError) as error:
                            self.window.statusBar().showMessage(f'Independent correction result available; project binding failed: {error}')
                component.result = result
                component.result_source = self._source
                component.dirty.clear()
                self._result_status(component)
                if component.graph.node(target).status is NodeStatus.RUNNING:
                    component.graph.node(target).status = NodeStatus.WARNING
                    component.graph.node(target).status_detail = 'No computed product was reported for this node'
                if self._component(self._selected) is component:
                    self.select_node(self._selected)
        elif component is not None:
            component.graph.node(target).status = NodeStatus.IDLE
            component.graph.node(target).status_detail = 'Cancelled; obsolete outputs discarded'
        self.window.pipeline_canvas.setEnabled(True)
        self.window.pipeline_inspector.setEnabled(True)
        self.window.pipeline_canvas.refresh()
        self.state_changed.emit()
        self._timer.start(0)

    def _result_status(self, component):
        from seedvision.pipeline.catalogue_nodes import display_id
        if component.instance.workflow is not None:
            for node in component.result.nodes:
                identifier = display_id(component.instance, node.stage_id)
                if identifier in component.graph.nodes:
                    item = component.graph.node(identifier)
                    item.status = (NodeStatus.COMPLETE if node.status in ('executed', 'provided', 'empty')
                        else NodeStatus.BYPASSED if node.status == 'bypassed' else NodeStatus.WARNING)
                    item.status_detail = ('Reused · ' if node.reused else '') + node.status + (': ' + node.reason if node.reason else '')
        else:
            for canonical in component.cache.last_computed_nodes + component.cache.last_reused_nodes:
                identifier = display_id(component.instance, canonical)
                if identifier in component.graph.nodes:
                    node = component.graph.node(identifier)
                    node.status = NodeStatus.COMPLETE
                    node.status_detail = 'Reused' if canonical in component.cache.last_reused_nodes else 'Complete'

    def stop(self):
        self._queue.clear()
        self._timer.stop()
        self._generation += 1
        if self._job is not None:
            self._job.cancel.set()
        self.state_changed.emit()

    def image_changed(self):
        from seedvision.ui.main_window import _path_identity
        path = self.window.image_view.image_path
        source = None if path is None else (str(path), self.window._source_fingerprints.get(_path_identity(path)))
        if source == self._source:
            return
        self._source = source
        self.stop()
        for component in self._components.values():
            component.result = component.prepared = None
            component.correction_context = None
            component.result_source = None
            component.cache = PipelineAnalysisCache()
            component.graph.invalidate(tuple(component.graph.nodes), preserve_bypassed=True)
        if self.handles(self._selected):
            self.select_node(self._selected)

    def context_changed(self, affected=None):
        """Invalidate default instances when applied image/reference context changes."""
        affected = None if affected is None else set(affected)
        changed_ids = set()
        for component in self._components.values():
            if component.instance.workflow is not None:
                continue
            targets = tuple(identifier for identifier in component.graph.nodes
                if affected is None or self._canonical(component, identifier) in affected)
            if not targets:
                continue
            expanded = set(targets)
            for identifier in targets:
                expanded.update(component.graph.downstream(identifier, recursive=True))
            component.graph.invalidate(expanded, preserve_bypassed=True)
            self._changed(component, expanded)
            changed_ids.add(component.instance.instance_id)
        self._queue = deque(item for item in self._queue if item[0] not in changed_ids)
        self.window.pipeline_canvas.refresh()
        self.state_changed.emit()

    def select_node(self, node_id):
        component = self._component(node_id)
        if component is None:
            return False
        if self._selected != node_id:
            self._selected_output = ''
        self._selected = node_id
        window, presenter = self.window, self.window.pipeline_workflow
        window.pipeline_inspector.enable_unified_optimization(False)
        window.pipeline_inspector.set_node(component.graph.node(node_id))
        window.pipeline_inspector.overlay_combo.hide()
        window.calibration_section.hide()
        window.baseline_section.hide()
        presenter.histogram.hide()
        presenter.crop_selector.hide()
        presenter.facts.clear()
        presenter.facts.hide()
        presenter.preview.clear()
        window.image_display_stack.setCurrentWidget(presenter.preview)
        self._node_actions(component, self._canonical(component, node_id))
        if hasattr(window, 'pipeline_backends'):
            window.pipeline_backends.refresh()
        if component.result is None:
            node = component.graph.node(node_id)
            presenter.preview.setText(node.status_detail if node.status in (NodeStatus.FAILED, NodeStatus.BLOCKED)
                else 'Run this independent node on the selected project image to inspect outputs.')
            return True
        canonical = self._canonical(component, node_id)
        if component.instance.workflow is None:
            self._show_default(component, canonical)
            return True
        try:
            result_node = component.result.node(canonical)
        except StopIteration:
            presenter.preview.setText('No product was recorded for this node.')
            return True
        stage = component.instance.workflow.recipe.stage(canonical)
        facts = [f'{stage.title}: {result_node.status}', 'Backend: ' + ('PlantCV public call' if getattr(result_node, 'backend', 'native') == 'plantcv' else 'Native'), result_node.reason]
        ports = {port.name: port for port in stage.ports if port.direction == 'output'}
        candidates = [(name, value, ports[name].semantics) for name, value in result_node.outputs.items()
            if name in ports and torch.is_tensor(value) and value.ndim in (2, 3)
            and (value.ndim == 2 or value.shape[-1] == 3)]
        for name, value in result_node.outputs.items():
            facts.append(f'{name}: {tuple(value.shape)} · {value.dtype}' if torch.is_tensor(value) else f'{name}: {str(value)[:1000]}')
        presenter.facts.setPlainText('\n'.join(str(text) for text in facts if text))
        presenter.facts.show()
        chosen = next((item for item in candidates if item[0] == self._selected_output), candidates[0] if candidates else None)
        if chosen:
            presenter.render(chosen[1], chosen[2])
        else:
            presenter.preview.setText('This node has compact outputs; inspect its recorded facts below.')
        return True

    def output_requested(self, node_id, port):
        if not self.handles(node_id):
            return False
        self._selected_output = port
        return self.select_node(node_id)

    def _show_default(self, component, canonical):
        from seedvision.ui.main_window import VIEWER_NODE_MODES
        result, presenter = component.result, self.window.pipeline_workflow
        mode = VIEWER_NODE_MODES.get(canonical, canonical)
        raster = getattr(result, mode, None)
        layers = getattr(result, 'layers', None)
        if raster is None and layers is not None:
            raster = getattr(layers, mode, None)
        if raster is None:
            advanced = getattr(result, 'advanced', None)
            raster = None if advanced is None else getattr(advanced, 'rasters', {}).get(mode)
        if hasattr(raster, 'gpu_tensor'):
            raster = raster.gpu_tensor()
            if raster.ndim == 4:
                raster = raster[0]
            if raster.ndim == 3 and raster.shape[0] in (1, 3):
                raster = raster[0] if raster.shape[0] == 1 else raster.permute(1, 2, 0)
        if torch.is_tensor(raster) and raster.ndim in (2, 3):
            presenter.render(raster, {})
        elif canonical in ('project', 'raw_images', 'project_image') and 'raw.image' in component.cache.values:
            image = np.ascontiguousarray(component.cache.values['raw.image'])
            view = QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format.Format_BGR888).copy()
            presenter.preview.setPixmap(QPixmap.fromImage(view))
        else:
            presenter.preview.setText('This node completed; its compact result and timing are recorded below.')
        node = next((node for identifier, node in component.graph.nodes.items() if identifier.endswith('::' + canonical)), None)
        elapsed = component.cache.node_timings_seconds.get(canonical)
        presenter.facts.setPlainText('\n'.join(filter(None, [
            None if node is None else node.title + ': ' + node.status_detail,
            None if elapsed is None else f'Calculation: {elapsed:.4f} seconds',
            *getattr(result, 'warnings', ())])))
        presenter.facts.show()

    def _node_actions(self, component, canonical):
        presenter = self.window.pipeline_workflow
        layout = presenter.node_actions.layout()
        while layout.count():
            widget = layout.takeAt(0).widget()
            if widget is not None:
                widget.deleteLater()
        workflow = component.instance.workflow
        if workflow is not None:
            stage = workflow.recipe.stage(canonical)
            if any(parameter.parameter_id == 'model_source' for parameter in stage.parameters):
                button = QPushButton('Select model…', presenter.node_actions)
                button.clicked.connect(lambda checked=False: self.choose_model(component.instance.instance_id, canonical))
                layout.addWidget(button)
            if workflow.recipe.recipe_id == 'plantcv_color_card_correct_color':
                for role, title in (('target_rgb', 'Target image…'), ('target_mask', 'Target chip mask…'), ('source_mask', 'Source chip mask…')):
                    button = QPushButton(title, presenter.node_actions)
                    button.setToolTip(workflow.input_paths.get(role, 'Required input for this independent correction instance.'))
                    button.clicked.connect(lambda checked=False, role=role: self.choose_input(component.instance.instance_id, role))
                    layout.addWidget(button)
        documents = getattr(self.window, "pipeline_documents", None)
        if documents is not None and documents.frozen:
            for index in range(layout.count()):
                layout.itemAt(index).widget().setEnabled(False)
        presenter.node_actions.setVisible(bool(layout.count()))

    def choose_model(self, identifier, canonical):
        guard = getattr(self.window, "_pipeline_editing_allowed", None)
        if guard is not None and not guard():
            return False
        from seedvision.pipeline.catalogue_nodes import display_id, instance_graph
        component = self._components[identifier]
        path, _ = QFileDialog.getOpenFileName(self.window, 'Select frozen model')
        if not path:
            return
        try:
            workflow = component.instance.workflow
            if 'kmeans' in workflow.recipe.recipe_id:
                from seedvision.benchmarking.native_kmeans_recipe import model_source_for_path
            else:
                from seedvision.benchmarking.native_naive_bayes_recipe import model_source_for_path
            component.graph.node(display_id(component.instance, canonical)).parameters['model_source'] = model_source_for_path(path)
            self._sync(component)
        except (ValueError, OSError) as error:
            component.graph = instance_graph(component.instance)
            self._merge()
            self.window.statusBar().showMessage(str(error))
            return
        self._changed(component, tuple(component.graph.nodes))
        self.select_node(display_id(component.instance, canonical))

    def choose_input(self, identifier, role):
        guard = getattr(self.window, "_pipeline_editing_allowed", None)
        if guard is not None and not guard():
            return False
        component = self._components[identifier]
        path, _ = QFileDialog.getOpenFileName(self.window, 'Select ' + role.replace('_', ' '), '', 'Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)')
        if path:
            workflow = component.instance.workflow
            values = dict(workflow.input_paths)
            values[role] = str(Path(path).resolve())
            component.instance = replace(component.instance, workflow=replace(workflow, input_paths=values))
            self.window._project_correction_bindings = tuple(binding for binding in self.window._project_correction_bindings
                if getattr(binding, 'owner_id', None) != component.instance.instance_id)
            component.prepared = None
            self._changed(component, tuple(component.graph.nodes))

    def export_selected(self):
        if not self.handles(self._selected):
            return False
        directory = QFileDialog.getExistingDirectory(self.window, 'Export independent node result')
        if directory:
            self.export_instance(self._selected, directory)
        return True

    def show_source_selected(self):
        component = self._component(self._selected)
        if component is None:
            return False
        instance = component.instance
        facts = self.window.pipeline_workflow.facts
        if instance.workflow is None:
            text = 'Independent native seed-analysis instance\n' + instance.catalogue_key + '\nThe selected calculation uses its own settings, cache and required ancestors.'
        else:
            recipe = instance.workflow.source_recipe
            origin = recipe.provenance
            text = (f'{recipe.title}\nOrigin: {origin.mode.value}\nBackend: {origin.backend_id} {origin.backend_version}\n'
                    f'{origin.evidence_note}\nThis independent instance retains its registered input and connection contracts.')
        facts.setPlainText(text)
        facts.show()
        return True

    def export_instance(self, node_id, directory):
        component = self._component(node_id) or self._components.get(node_id)
        if component is None or component.result is None:
            self.window.statusBar().showMessage('Run this independent node before exporting its result.')
            return False
        if self.busy or self.window._thread_pool.activeThreadCount():
            self.window.statusBar().showMessage('Wait for current work before exporting this result.')
            return False
        instance, result = component.instance, component.result
        destination = Path(directory) / instance.instance_id
        if instance.workflow is not None:
            from seedvision.export.native_results import export_native_recipe_result
            function = lambda cancel: export_native_recipe_result(destination, instance.workflow.recipe, result, cancel=cancel)
        else:
            from seedvision.export.results import result_report, export_report
            from seedvision.pipeline.catalogue_nodes import default_graph, instance_nodes
            from seedvision.persistence.analysis_settings import analysis_settings_profile_to_payload
            graph = default_graph(instance)
            def function(cancel):
                if cancel():
                    raise ValueError('Export cancelled before publication.')
                report = result_report(result, enabled={identifier for identifier in instance_nodes(instance) if graph.node(identifier).enabled},
                    settings=analysis_settings_profile_to_payload(instance.default_settings), execution_provenance={'catalogue_instance': instance.instance_id, 'catalogue_key': instance.catalogue_key})
                return export_report(destination, result, report)
        token = (self._generation, instance.instance_id, component.generation, str(self.window.image_view.image_path))
        job = _Job(function, token)
        self._job = job
        def finished(outcome):
            if self._job is not job:
                return
            self._job = None
            self.window.statusBar().showMessage(str(outcome) if isinstance(outcome, Exception) else f'Independent result exported to {destination}')
            self.state_changed.emit()
            self._timer.start(0)
        job.signals.finished.connect(finished)
        self.state_changed.emit()
        self.window._thread_pool.start(job)
        return True
