"""Freeze state and explicit file commands for the native Pipeline workspace."""
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QObject, QSignalBlocker, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QCheckBox, QFileDialog, QInputDialog, QMessageBox

from seedvision.persistence.pipeline_documents import (
    PIPELINE_EXTENSION, LAYOUT_EXTENSION, PipelineDocumentError,
    PipelineDocumentStore, PipelineLayoutDocument, _name_key, _pipeline_name,
)


class PipelineDocumentsController(QObject):
    """Own editing permission; callers guard their calculation mutation routes."""

    frozen_changed = Signal(bool)
    pipeline_saved = Signal(object, object)

    def __init__(self, window, *, capture, install, store=None):
        super().__init__(window)
        self.window = window
        self._capture = capture
        self._install = install
        self.store = store or PipelineDocumentStore(window._root)
        self._frozen = True
        self.current_path = None
        self.current_name = 'Custom pipeline'
        self.factory_name = None
        self.freeze_action = QAction('Frozen', window)
        self.freeze_action.setCheckable(True)
        self.freeze_action.setChecked(True)
        self.freeze_action.setToolTip(
            'Freeze node parameters and connections. Running, backend selection, layout and inspection remain available.')
        self.freeze_action.toggled.connect(self.set_frozen)
        self.freeze_button = QCheckBox('Freeze', window)
        self.freeze_button.setObjectName('pipelineFreezeButton')
        self.freeze_button.setToolTip(self.freeze_action.toolTip())
        self.freeze_button.setChecked(True)
        self.freeze_button.toggled.connect(self.set_frozen)
        self.load_pipeline_action = QAction('Load pipeline…', window)
        self.load_pipeline_action.triggered.connect(self.choose_pipeline)
        self.save_pipeline_action = QAction('Save pipeline as…', window)
        self.save_pipeline_action.triggered.connect(self.choose_pipeline_save)
        self.load_layout_action = QAction('Load layout…', window)
        self.load_layout_action.triggered.connect(self.choose_layout)
        self.save_layout_action = QAction('Save layout as…', window)
        self.save_layout_action.triggered.connect(self.choose_layout_save)

    @property
    def frozen(self):
        return self._frozen

    def set_frozen(self, frozen=True):
        frozen = bool(frozen)
        changed = frozen != self._frozen
        self._frozen = frozen
        with QSignalBlocker(self.freeze_action):
            self.freeze_action.setChecked(frozen)
            self.freeze_action.setText('Frozen' if frozen else 'Editable')
        with QSignalBlocker(self.freeze_button):
            self.freeze_button.setChecked(frozen)
        if changed:
            self.frozen_changed.emit(frozen)

    def loaded(self, factory_name=None, *, name=None, path=None):
        """Call after every template/project load, including cached graph copies."""
        self.factory_name = factory_name
        self.current_name = name or factory_name or 'Custom pipeline'
        self.current_path = None if path is None else Path(path).resolve()
        was_frozen = self.frozen
        self.set_frozen(True)
        # Reapply locks when the graph/inspector changed but the flag did not.
        if was_frozen:
            self.frozen_changed.emit(True)

    def can_edit(self):
        if self._frozen:
            self.window.statusBar().showMessage('Pipeline is frozen. Clear Freeze to edit node parameters and connections.')
            return False
        return True

    def _busy(self):
        return (self.window._background_work_is_active()
                or bool(getattr(getattr(self.window, 'catalogue_nodes', None), 'busy', False))
                or self.window._thread_pool.activeThreadCount() > 0)

    def _error(self, action, error):
        QMessageBox.warning(self.window, action, str(error))

    def load_pipeline(self, path):
        if self._busy():
            self.window.statusBar().showMessage('Wait for current work to finish before loading a pipeline.')
            return False
        try:
            document = self.store.load_pipeline(path)
            if self._install(document) is False:
                return False
            statuses = document.asset_statuses(self.window._root)
        except (ValueError, TypeError, KeyError, OSError, RecursionError, NotImplementedError) as error:
            self._error('Could not load pipeline', error)
            return False
        factory = ('Default seed analysis' if document.pipeline_workflow is None
                   else document.pipeline_workflow.source_recipe.title)
        self.loaded(factory, name=document.name, path=path)
        unavailable = sum(row['status'] not in ('available', 'unchecked') for row in statuses)
        self.window.statusBar().showMessage(
            f'Loaded pipeline {document.name}; calculations are frozen.'
            + (f' {unavailable} referenced asset(s) remain unavailable.' if unavailable else ''))
        return True

    def save_pipeline_as(self, path, name):
        if self._busy():
            self.window.statusBar().showMessage('Wait for current work to finish before saving a pipeline.')
            return False
        try:
            if self.factory_name is not None and _name_key(name) == _name_key(self.factory_name):
                raise PipelineDocumentError('Choose a distinct custom name for a factory-derived pipeline.')
            document = self._capture(name)
            if document.name != name:
                document = replace(document, name=name)
            destination = self.store.save_pipeline(document, path)
        except (ValueError, TypeError, KeyError, OSError, RecursionError) as error:
            self._error('Could not save pipeline', error)
            return False
        self.current_name, self.current_path = name, destination
        self.pipeline_saved.emit(document, destination)
        self.window.statusBar().showMessage(f'Saved pipeline {name} to {destination}.')
        return True

    def capture_layout(self):
        window, canvas = self.window, self.window.pipeline_canvas
        selected = next((identifier for identifier, item in canvas.node_items.items() if item.isSelected()), None)
        output = ''
        catalogue = getattr(window, 'catalogue_nodes', None)
        if catalogue is not None and catalogue.handles(selected):
            output = catalogue._selected_output
        elif getattr(window.pipeline_workflow, 'active', False):
            output = window.pipeline_workflow._preferred_output
        elif getattr(window, 'overlay_combo', None) is not None:
            output = str(window.overlay_combo.currentData() or '')
        return PipelineLayoutDocument.from_graph(
            canvas.graph, bundle_cables=canvas.cable_bundling_enabled,
            route_around_nodes=canvas.obstacle_routing_enabled, selected_node=selected,
            selected_output=output, image_visible=window.image_workspace_action.isChecked(),
            pipeline_visible=window.pipeline_workspace_action.isChecked())

    def apply_layout(self, document):
        window, canvas = self.window, self.window.pipeline_canvas
        missing = document.apply(canvas.graph)
        with QSignalBlocker(canvas):
            canvas.set_cable_bundling_enabled(document.bundle_cables)
            canvas.set_obstacle_routing_enabled(document.route_around_nodes)
            selected = canvas.rebuild_from_graph(document.selected_node)
        if selected is not None:
            canvas.select_node(selected)
            catalogue = getattr(window, 'catalogue_nodes', None)
            if catalogue is not None and catalogue.handles(selected):
                catalogue.output_requested(selected, document.selected_output)
            elif window.pipeline_workflow.active and document.selected_output:
                window.pipeline_workflow._preferred_output = document.selected_output
                window.pipeline_workflow.select_node(selected)
            elif getattr(window, 'overlay_combo', None) is not None and document.selected_output:
                index = window.overlay_combo.findData(document.selected_output)
                if index >= 0:
                    window.overlay_combo.setCurrentIndex(index)
        window.image_workspace_action.setChecked(document.image_visible)
        window.pipeline_workspace_action.setChecked(document.pipeline_visible)
        if getattr(window, 'catalogue_nodes', None) is not None:
            window.catalogue_nodes.layout_changed()
        window._set_project_dirty()
        return missing

    def load_layout(self, path):
        if self._busy():
            self.window.statusBar().showMessage('Wait for current work to finish before loading a layout.')
            return False
        try:
            document = self.store.load_layout(path)
            missing = self.apply_layout(document)
        except (ValueError, TypeError, KeyError, OSError, RecursionError) as error:
            self._error('Could not load layout', error)
            return False
        self.window.statusBar().showMessage(
            f'Loaded pipeline layout from {path}.'
            + (f' {len(missing)} node ID(s) are absent from this pipeline.' if missing else ''))
        return True

    def save_layout_as(self, path):
        if self._busy():
            self.window.statusBar().showMessage('Wait for current work to finish before saving a layout.')
            return False
        try:
            self.store.save_layout(self.capture_layout(), path)
        except (ValueError, TypeError, KeyError, OSError, RecursionError) as error:
            self._error('Could not save layout', error)
            return False
        self.window.statusBar().showMessage(f'Saved pipeline layout to {path}.')
        return True

    def choose_pipeline(self):
        path, _ = QFileDialog.getOpenFileName(self.window, 'Load pipeline', '',
                                             f'Seed Fiddle pipelines (*{PIPELINE_EXTENSION})')
        return bool(path) and self.load_pipeline(path)

    def choose_pipeline_save(self):
        initial = self.current_name
        if self.factory_name is not None and _name_key(initial) == _name_key(self.factory_name):
            initial = 'Custom ' + initial
        name, accepted = QInputDialog.getText(self.window, 'Save pipeline as', 'Custom pipeline name:', text=initial)
        if not accepted:
            return False
        try:
            _pipeline_name(name)
        except PipelineDocumentError as error:
            self._error('Could not save pipeline', error)
            return False
        path, _ = QFileDialog.getSaveFileName(
            self.window, 'Save pipeline as', str(self.current_path or self.window._root / (name + PIPELINE_EXTENSION)),
            f'Seed Fiddle pipelines (*{PIPELINE_EXTENSION})')
        if path and not path.casefold().endswith(PIPELINE_EXTENSION):
            path += PIPELINE_EXTENSION
        return bool(path) and self.save_pipeline_as(path, name)

    def choose_layout(self):
        path, _ = QFileDialog.getOpenFileName(self.window, 'Load pipeline layout', '',
                                             f'Seed Fiddle layouts (*{LAYOUT_EXTENSION})')
        return bool(path) and self.load_layout(path)

    def choose_layout_save(self):
        path, _ = QFileDialog.getSaveFileName(
            self.window, 'Save pipeline layout', str(self.window._root / ('pipeline' + LAYOUT_EXTENSION)),
            f'Seed Fiddle layouts (*{LAYOUT_EXTENSION})')
        if path and not path.casefold().endswith(LAYOUT_EXTENSION):
            path += LAYOUT_EXTENSION
        return bool(path) and self.save_layout_as(path)
