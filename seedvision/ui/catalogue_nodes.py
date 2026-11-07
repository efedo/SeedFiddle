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
from PySide6.QtWidgets import QFileDialog

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
    generation: int = 0
    dirty: set = field(default_factory=set)
    result_source: object = None


class CatalogueNodesController(QObject):
    """Own added components without borrowing the primary graph's mutable state.

    Only display identifiers are namespaced. Each instance runs its validated
    canonical graph with private settings/cache and the current source context.
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
        from seedvision.pipeline.catalogue_nodes import catalogue_entries
        from seedvision.pipeline import build_default_pipeline
        default = build_default_pipeline()
        default_nodes = {**default.nodes, **default.unused_nodes}
        counts = {}
        def present(key):
            counts[key] = counts.get(key, 0) + 1
        if self.primary is not None:
            for identifier in self.primary.nodes:
                present('default:' + identifier)
        for component in self._components.values():
            for identifier in component.graph.nodes:
                present('default:' + self._canonical(component, identifier))
        result = []
        for entry in catalogue_entries():
            count = counts.get(entry.key, 0)
            available = default_nodes[entry.target_node].implemented
            result.append(dict(key=entry.key, title=entry.title, category=entry.category,
                description=f'{entry.title} · add another independent instance, including required inputs and supported internal connection routes'
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
            self.window.statusBar().showMessage('Select an added node to remove it.')
            return False
        if self._job is not None and self._job.token[1] == component.instance.instance_id:
            self._job.cancel.set()
        self._components.pop(component.instance.instance_id)
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
        job.context_fingerprint = self._context_fingerprint(Path(path))
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
        from seedvision.pipeline.settings import production_settings
        from seedvision.segmentation.baseline import analyze_path
        graph = default_graph(component.instance)
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
        if current:
            current = getattr(job, 'context_fingerprint', None) == self._context_fingerprint(Path(job.token[3]))
        if current and not job.cancel.is_set():
            if isinstance(result, Exception):
                component.graph.node(target).status = NodeStatus.FAILED
                component.graph.node(target).status_detail = str(result)
                self.window.statusBar().showMessage(f'Independent node failed: {result}')
                if self._component(self._selected) is component:
                    self.select_node(self._selected)
            else:
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
            component.result = None
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
        presenter.facts.clear()
        presenter.facts.hide()
        presenter.preview.clear()
        window.image_display_stack.setCurrentWidget(presenter.preview)
        if component.result is None:
            node = component.graph.node(node_id)
            presenter.preview.setText(node.status_detail if node.status in (NodeStatus.FAILED, NodeStatus.BLOCKED)
                else 'Run this independent node on the selected project image to inspect outputs.')
            return True
        self._show_default(component, self._canonical(component, node_id))
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
            presenter.render(raster)
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

    def export_selected(self):
        if not self.handles(self._selected):
            return False
        directory = QFileDialog.getExistingDirectory(self.window, 'Export independent node result')
        if directory:
            self.export_instance(self._selected, directory)
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
