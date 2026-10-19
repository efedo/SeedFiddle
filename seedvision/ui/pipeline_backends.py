"""Execution backend controls shared by primary and independent pipeline nodes."""
from dataclasses import replace

from PySide6.QtCore import QObject, QSignalBlocker
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QWidget


class PipelineBackendsController(QObject):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.default = 'native'
        self.global_selector = QComboBox(window.pipeline_workflow.strip)
        self.global_selector.setObjectName('pipelineBackendSelector')
        self.global_selector.addItem('Native', 'native')
        self.global_selector.addItem('PlantCV', 'plantcv')
        self.global_selector.setFixedWidth(84)
        self.global_selector.setToolTip('Default execution backend. Native-only preparation and unsupported equivalents remain explicitly native.')
        window.pipeline_workflow.strip.layout().insertWidget(2, self.global_selector)
        self.global_selector.currentIndexChanged.connect(self._default_changed)
        self.node_controls = QWidget(window.pipeline_inspector)
        row = QHBoxLayout(self.node_controls)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel('Backend', self.node_controls))
        self.node_selector = QComboBox(self.node_controls)
        self.node_selector.setObjectName('pipelineNodeBackendSelector')
        row.addWidget(self.node_selector, 1)
        window.pipeline_inspector.layout().insertWidget(1, self.node_controls)
        self.node_selector.currentIndexChanged.connect(self._node_changed)
        self.refresh()

    def _context(self):
        window = self.window
        catalogue = window.catalogue_nodes
        component = catalogue._component(catalogue._selected)
        if component is not None:
            workflow = component.instance.workflow
            identifier = catalogue._canonical(component, catalogue._selected)
            return workflow, identifier, component
        presenter = window.pipeline_workflow
        return presenter.workflow, presenter._selected, None

    def refresh(self):
        from seedvision.pipeline.backend_execution import backend_capability, resolve_backend
        workflow, identifier, component = self._context()
        primary = self.window.pipeline_workflow.workflow
        if primary is not None:
            self.default = primary.backend_default
        busy = (self.window.pipeline_workflow._job is not None or self.window.catalogue_nodes.busy)
        with QSignalBlocker(self.global_selector):
            self.global_selector.setCurrentIndex(self.global_selector.findData(self.default))
            self.global_selector.setEnabled(not busy)
        with QSignalBlocker(self.node_selector):
            self.node_selector.clear()
            if workflow is None or identifier not in {s.stage_id for s in workflow.recipe.stages}:
                self.node_selector.addItem('Native only', 'native')
                self.node_selector.setEnabled(False)
                self.node_selector.setToolTip('This calculation has no equivalent original PlantCV call.')
                return
            capability = backend_capability(workflow.recipe, identifier)
            try:
                effective = resolve_backend(workflow, identifier)
            except (ValueError, NotImplementedError):
                effective = 'unavailable'
            inherited = 'PlantCV' if workflow.backend_default == 'plantcv' and 'plantcv' in capability.backends else 'Native'
            self.node_selector.addItem('Inherit · ' + inherited, '')
            self.node_selector.addItem('Native', 'native')
            if 'plantcv' in capability.backends or effective == 'unavailable':
                self.node_selector.addItem('PlantCV' if effective != 'unavailable' else 'PlantCV · unavailable', 'plantcv')
            self.node_selector.setCurrentIndex(max(0, self.node_selector.findData(workflow.node_backends.get(identifier, ''))))
            self.node_selector.setEnabled(not busy and (len(capability.backends) > 1 or effective == 'unavailable'))
            self.node_selector.setToolTip(capability.reason or (
                'All internal watershed stages share one public-call backend.' if capability.atomic_group else
                'Select the original audited PlantCV CPU call or the equivalent native calculation.'))

    def configure_new_instance(self, instance):
        if instance.workflow is not None:
            return replace(instance, workflow=replace(instance.workflow, backend_default=self.default))
        return instance

    def _install_changed(self, old, new, component):
        from seedvision.pipeline.backend_execution import resolve_backend, validate_backend_selection
        validate_backend_selection(new)
        def previous_backend(identifier):
            try:
                return resolve_backend(old, identifier)
            except (ValueError, NotImplementedError):
                return 'unavailable'
        changed = {s.stage_id for s in new.recipe.stages
                   if previous_backend(s.stage_id) != resolve_backend(new, s.stage_id)}
        if component is None:
            presenter = self.window.pipeline_workflow
            presenter.workflow = new
            affected = set(changed)
            for node_id in changed:
                affected.update(presenter.graph.downstream(node_id, recursive=True))
            if affected:
                presenter.graph.invalidate(affected, preserve_bypassed=True)
                presenter.changed()
        else:
            from seedvision.pipeline.catalogue_nodes import display_id
            component.instance = replace(component.instance, workflow=new)
            affected = {display_id(component.instance, node_id) for node_id in changed
                        if display_id(component.instance, node_id) in component.graph.nodes}
            for node_id in tuple(affected):
                affected.update(component.graph.downstream(node_id, recursive=True))
            if affected:
                component.graph.invalidate(affected, preserve_bypassed=True)
                self.window.catalogue_nodes._changed(component, affected)
        self.window._set_project_dirty()
        self.window.pipeline_canvas.refresh()

    def _default_changed(self, index):
        if self.window.pipeline_workflow._job is not None or self.window.catalogue_nodes.busy:
            self.refresh()
            return
        from seedvision.pipeline.backend_execution import validate_backend_selection
        selected = self.global_selector.itemData(index)
        try:
            candidates = []
            presenter = self.window.pipeline_workflow
            if presenter.active:
                presenter.sync_graph()
                candidates.append((presenter.workflow, replace(presenter.workflow, backend_default=selected), None))
            for component in self.window.catalogue_nodes._components.values():
                self.window.catalogue_nodes._sync(component)
                workflow = component.instance.workflow
                if workflow is not None:
                    candidates.append((workflow, replace(workflow, backend_default=selected), component))
            for old, new, component in candidates:
                validate_backend_selection(new)
            for old, new, component in candidates:
                self._install_changed(old, new, component)
        except (ValueError, TypeError, NotImplementedError) as error:
            self.window.statusBar().showMessage(str(error))
        else:
            self.default = selected
            self.window._set_project_dirty()
            self.window.statusBar().showMessage('Execution backend: ' + ('PlantCV; native-only stages remain native.' if selected == 'plantcv' else 'Native.'))
        self.refresh()

    def _node_changed(self, index):
        if self.window.pipeline_workflow._job is not None or self.window.catalogue_nodes.busy:
            self.refresh()
            return
        from seedvision.pipeline.backend_execution import backend_capability
        workflow, identifier, component = self._context()
        if workflow is None or identifier is None:
            return
        selected = self.node_selector.itemData(index)
        values = dict(workflow.node_backends)
        capability = backend_capability(workflow.recipe, identifier)
        for stage_id in capability.atomic_group or (identifier,):
            if selected:
                values[stage_id] = selected
            else:
                values.pop(stage_id, None)
        try:
            self._install_changed(workflow, replace(workflow, node_backends=values), component)
        except (ValueError, TypeError, NotImplementedError) as error:
            self.window.statusBar().showMessage(str(error))
        else:
            self.window.statusBar().showMessage('Node backend updated; run the pipeline to calculate its outputs.')
        self.refresh()
