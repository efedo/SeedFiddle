"""Default-pipeline lock, named copies and pipeline file commands for the Pipeline workspace.

Node layouts are not separate files: projects and pipeline files carry them.
"""
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QObject, QSignalBlocker, Signal
from PySide6.QtWidgets import QFileDialog, QInputDialog, QMessageBox

from seedvision.persistence.pipeline_documents import (
    PIPELINE_EXTENSION, PipelineDocumentError,
    PipelineDocumentStore, PipelineLayoutDocument, _name_key, _pipeline_name,
)

COPY_ERRORS = (ValueError, TypeError, KeyError, OSError, RecursionError, NotImplementedError)


class PipelineDocumentsController(QObject):
    """Decide what may be edited; callers guard their calculation mutation routes.

    A default pipeline is always locked. An editable pipeline is a named copy in
    ``custom``: the working graph holds the current one, whose entry is ``None``,
    and every other copy is held as a complete ``PipelineDocument``.
    """

    lock_changed = Signal(bool)
    named = Signal(str)
    copies_changed = Signal()

    def __init__(self, window, *, capture, install, store=None):
        super().__init__(window)
        self.window = window
        self._capture = capture
        self._install = install
        self.store = store or PipelineDocumentStore(window._root)
        self._locked = True
        self.custom = {}
        self.current_path = None
        self.current_name = 'Custom pipeline'
        self.factory_name = None

    @property
    def locked(self):
        return self._locked

    @property
    def active_copy(self):
        """The current copy's name, or None while a default pipeline is loaded."""
        return None if self._locked else self.current_name

    @property
    def stored_copies(self):
        """Every copy but the current one, whose state is the working graph."""
        return tuple(document for name, document in self.custom.items()
                     if document is not None and name != self.active_copy)

    def set_stored_copies(self, documents):
        """Replace the copies with those of the project being opened."""
        self.custom = {document.name: document for document in documents}
        self.copies_changed.emit()

    def loaded(self, factory_name=None, *, name=None, path=None):
        """Call after every template/project load, including cached graph copies."""
        self.factory_name = factory_name
        self.current_name = name or factory_name or 'Custom pipeline'
        self.current_path = None if path is None else Path(path).resolve()
        self._locked = name is None or _name_key(name) in self.store.factory_names
        # Only a caller that replaced the graph without keeping its copy leaves one behind.
        self.custom = {key: value for key, value in self.custom.items() if value is not None}
        if not self._locked:
            self.custom[name] = None
        self.copies_changed.emit()
        # Reapply locks when the graph/inspector changed but the flag did not.
        self.lock_changed.emit(self._locked)

    def can_edit(self):
        """True when the current pipeline may change, offering a copy of a default one.

        Accepting moves the working graph onto the copy, so the caller's edit
        continues there. Declining leaves the default pipeline unchanged.
        """
        return not self._locked or self.offer_copy()

    def offer_copy(self):
        answer = QMessageBox.question(self.window, 'Default pipeline',
            'Default pipelines cannot be edited.\n\n'
            f'Would you like to create an editable copy of {self.current_name}?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes)
        if answer != QMessageBox.StandardButton.Yes:
            self.window.statusBar().showMessage(f'{self.current_name} was not changed.')
            return False
        name = self._ask_name('Create editable copy', self.unique_name(self.current_name + ' copy'))
        if name is None:
            self.window.statusBar().showMessage(f'{self.current_name} was not changed.')
            return False
        self.create_copy(name)
        return True

    def create_copy(self, name):
        """Make the working graph the named copy; nothing needs recalculating."""
        name = self._validated_name(name)
        source = self.current_name
        self.stash_current()
        self.current_name, self.current_path = name, None
        self._locked = False
        self.custom[name] = None
        self.named.emit(name)
        self.copies_changed.emit()
        self.lock_changed.emit(False)
        self.window.statusBar().showMessage(f'Created {name}, an editable copy of {source}.')

    def rename_current(self):
        """Rename a copy; a default pipeline cannot be renamed, so it offers a copy instead."""
        if self._locked:
            return self.offer_copy()
        old = self.current_name
        name = self._ask_name('Rename pipeline', old, exclude=old)
        if name is None or name == old:
            return False
        self.custom = {(name if key == old else key): value for key, value in self.custom.items()}
        self.current_name, self.current_path = name, None
        self.named.emit(name)
        self.copies_changed.emit()
        self.window.statusBar().showMessage(f'Renamed {old} to {name}.')
        return True

    def stash_current(self):
        """Keep the working copy so the list can return to it. False if it cannot be captured."""
        if self._locked:
            return True
        try:
            self.custom[self.current_name] = self._capture(self.current_name)
        except COPY_ERRORS as error:
            self._error('Could not keep this pipeline', error)
            return False
        return True

    def activate(self, name):
        """Replace the working graph with a stored copy."""
        document = self.custom[name]
        try:
            if self._install(document) is False:
                return False
        except COPY_ERRORS as error:
            self._error('Could not load pipeline', error)
            return False
        self.loaded(self._factory_of(document), name=name)
        return True

    def unique_name(self, name):
        """``name``, numbered if the project already has a pipeline called that."""
        candidate, number = name, 2
        taken = {_name_key(key) for key in self.custom} | set(self.store.factory_names)
        while _name_key(candidate) in taken:
            candidate, number = f'{name} {number}', number + 1
        return candidate

    def _ask_name(self, title, initial, *, exclude=None):
        """Ask until the name is valid. A refused name is kept so it can be corrected."""
        label, text = 'Pipeline name:', initial
        while True:
            text, accepted = QInputDialog.getText(self.window, title, label, text=text)
            if not accepted:
                return None
            try:
                return self._validated_name(text.strip(), exclude=exclude)
            except PipelineDocumentError as error:
                label = f'{error}\n\nPipeline name:'

    def _validated_name(self, name, *, exclude=None):
        if not name:
            raise PipelineDocumentError('Enter a name for the pipeline.')
        _pipeline_name(name)
        key = _name_key(name)
        if key in self.store.factory_names:
            raise PipelineDocumentError(f'{name} is a default pipeline. Choose another name.')
        if any(_name_key(other) == key for other in self.custom if other != exclude):
            raise PipelineDocumentError(f'This project already has a pipeline called {name}.')
        return name

    @staticmethod
    def _factory_of(document):
        return ('Default seed analysis' if document.pipeline_workflow is None
                else document.pipeline_workflow.source_recipe.title)

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
            if not self.stash_current():
                return False
            if self._install(document) is False:
                return False
            statuses = document.asset_statuses(self.window._root)
        except COPY_ERRORS as error:
            self._error('Could not load pipeline', error)
            return False
        name = self.unique_name(document.name)
        self.loaded(self._factory_of(document), name=name, path=path)
        if name != document.name:
            self.named.emit(name)
        unavailable = sum(row['status'] not in ('available', 'unchecked') for row in statuses)
        self.window.statusBar().showMessage(
            f'Loaded pipeline {name}.'
            + (f' {unavailable} referenced asset(s) remain unavailable.' if unavailable else ''))
        return True

    def save_pipeline_as(self, path, name):
        """Publish the working graph to a file and carry on under that name, as Save As does."""
        if self._busy():
            self.window.statusBar().showMessage('Wait for current work to finish before saving a pipeline.')
            return False
        try:
            if self.factory_name is not None and _name_key(name) == _name_key(self.factory_name):
                raise PipelineDocumentError('Choose a distinct custom name for a factory-derived pipeline.')
            self._validated_name(name, exclude=self.active_copy)
            document = self._capture(name)
            if document.name != name:
                document = replace(document, name=name)
            destination = self.store.save_pipeline(document, path)
        except (ValueError, TypeError, KeyError, OSError, RecursionError) as error:
            self._error('Could not save pipeline', error)
            return False
        if name != self.active_copy:
            if not self.stash_current():
                return False
            self.current_name, self._locked = name, False
            self.custom[name] = None
            self.named.emit(name)
            self.copies_changed.emit()
            self.lock_changed.emit(False)
        self.current_path = destination
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
