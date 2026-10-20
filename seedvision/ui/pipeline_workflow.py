"""Editable native workflows in the existing desktop Pipeline workspace."""
from __future__ import annotations

from pathlib import Path
from threading import Event
from time import monotonic
import json
from dataclasses import replace
from collections.abc import Mapping

import torch
from PySide6.QtCore import QObject, QRunnable, QSignalBlocker, Qt, Signal, QRectF
from PySide6.QtGui import QPixmap, QPen, QColor
from PySide6.QtWidgets import (QComboBox, QFileDialog, QGraphicsScene, QGraphicsView,
    QHBoxLayout, QLabel, QMenu, QMessageBox, QPushButton, QTextBrowser, QToolButton,
    QSizePolicy, QVBoxLayout, QWidget)

from seedvision.pipeline import NodeStatus
from seedvision.pipeline.project_context import CONTEXT_NODE_IDS, workspace_graph
from seedvision.segmentation import PipelineAnalysisCache
from seedvision.ui.native_workflow_helpers import configure_native_parameters, configure_foreground_regions, native_preview_image


class _Signals(QObject):
    completed = Signal(object)
    progress = Signal(str, str, str)


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
        self.signals.completed.emit(result)


class _RasterView(QGraphicsView):
    region_selected = Signal(object)
    selecting_region = False
    _region_item = None

    def mousePressEvent(self, event):
        if self.selecting_region and event.button() == Qt.MouseButton.LeftButton:
            self._region_start = self.mapToScene(event.position().toPoint())
            self._region_item = self.scene().addRect(QRectF(self._region_start, self._region_start),
                QPen(QColor('#ff44bb'), 2))
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.selecting_region and self._region_item is not None:
            self._region_item.setRect(QRectF(self._region_start,
                self.mapToScene(event.position().toPoint())).normalized())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.selecting_region and self._region_item is not None:
            rectangle = self._region_item.rect().intersected(self.sceneRect())
            self.scene().removeItem(self._region_item)
            self._region_item = None
            self.selecting_region = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.region_selected.emit(rectangle)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        self.scale(factor, factor)
        event.accept()


class WorkflowPreview(QWidget):
    """Sampled Qt display with native pan, zoom, fit and pixel inspection."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        controls = QHBoxLayout()
        self.message = QLabel(self)
        self.message.setWordWrap(True)
        self.message.setTextFormat(Qt.TextFormat.PlainText)
        controls.addWidget(self.message, 1)
        fit = QPushButton('Fit', self)
        fit.clicked.connect(self.fit_image)
        actual = QPushButton('1:1', self)
        actual.clicked.connect(lambda: self.view.resetTransform())
        controls.addWidget(fit)
        controls.addWidget(actual)
        layout.addLayout(controls)
        self.view = _RasterView(self)
        self.view.setScene(QGraphicsScene(self.view))
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        layout.addWidget(self.view, 1)

    def clear(self):
        self.view._region_item = None
        self.view.selecting_region = False
        self.view.scene().clear()
        self.output_item = None
        self.message.clear()

    def setText(self, text):
        self.clear()
        self.message.setText(text)

    def setPixmap(self, pixmap):
        self.clear()
        item = self.view.scene().addPixmap(pixmap)
        self.output_item = item
        self.view.scene().setSceneRect(item.boundingRect())
        self.message.setText('Node output · drag to pan · scroll to zoom')
        self.fit_image()

    def set_base(self, pixmap):
        item = self.view.scene().addPixmap(pixmap)
        item.setZValue(-1)

    def fit_image(self):
        self.view.fitInView(self.view.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)


class PipelineWorkflowController(QObject):
    """Keep default analysis state intact while routing the active native graph."""
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.workflow = None
        self.graph = None
        self.result = None
        self.cache = PipelineAnalysisCache()
        self._job = None
        self._generation = 0
        self._image_path = None
        self._image_fingerprint = None
        self._selected = None
        self._preferred_output = ''
        self._copies = {}
        self._correction_context = None
        self._correction_prepared = None
        self._toolbar_visibility = None
        self._panel_visibility = None
        self._default_device_label = None
        self._default_selection = window._selected_pipeline_node
        from seedvision.ui.workflow_display import WorkflowDisplay
        self.display = WorkflowDisplay(self)
        self.activity = ''
        self._started_at = None
        self.strip = QWidget(window)
        self.strip.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.strip.setObjectName('pipelineWorkflowStrip')
        row = QHBoxLayout(self.strip)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(3)
        self.workflow_label = QLabel('Workflow', self.strip)
        row.addWidget(self.workflow_label)
        self.selector = QComboBox(self.strip)
        self.selector.setFixedWidth(120)
        self.selector.setMinimumContentsLength(8)
        self.selector.view().setMinimumWidth(360)
        self.selector.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.selector.setObjectName('pipelineWorkflowSelector')
        self.selector.addItem('Default seed analysis', None)
        from seedvision.pipeline.workflows import workflow_catalogue
        self._all_templates = workflow_catalogue()
        primary = ('plantcv_color_card_correct_color', 'plantcv_seed_binary_working',
            'plantcv_seed_mean_working', 'plantcv_seed_otsu_working', 'smpts_paper_fixed_mean')
        by_id = {recipe.recipe_id: recipe for recipe in self._all_templates}
        for identifier in primary:
            recipe = by_id[identifier]
            self.selector.addItem(recipe.title, recipe)
            self.selector.setItemData(self.selector.count()-1, recipe.description, Qt.ItemDataRole.ToolTipRole)
        self._catalogue_count = self.selector.count()
        row.addWidget(self.selector)
        self.save_button = QToolButton(self.strip)
        self.save_button.setText('Save')
        self.export_button = QToolButton(self.strip)
        self.export_button.setText('Export')
        for button in (self.save_button, self.export_button):
            row.addWidget(button)
        self.more_button = QToolButton(self.strip)
        self.more_button.setText('…')
        self.more_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(self.more_button)
        alternatives = menu.addMenu('Other pipelines')
        for recipe in self._all_templates:
            if recipe.recipe_id not in primary:
                alternatives.addAction(recipe.title, lambda checked=False, recipe=recipe: self.load_recipe(recipe))
        menu.addAction('Import legacy recipe…', self.open_saved)
        menu.addAction('Source and availability…', self.show_source)
        self.export_menu_action = menu.addAction('Export run…', self.export)
        self.export_menu_action.setVisible(False)
        menu.addAction('Recorded run QC…', self.show_qc)
        self.replay_action = menu.addAction('Replay selected saved stage…', self.replay)
        menu.addSeparator()
        self.remove_added_action = menu.addAction('Remove selected added node',
            lambda checked=False: self.window.catalogue_nodes.remove_selected())
        self.remove_added_action.setToolTip('Remove this independent instance and its private required input nodes.')
        self.more_button.setMenu(menu)
        row.addWidget(self.more_button)
        self.strip.adjust_for_width = self.adjust_for_width
        self.facts = QTextBrowser(window)
        self.facts.setObjectName('pipelineWorkflowResults')
        self.facts.setMaximumHeight(240)
        self.facts.hide()
        self.preview = WorkflowPreview(window)
        self.preview.setText('Run a workflow to inspect its node outputs.')
        self.preview.setObjectName('pipelineWorkflowPreview')
        self.preview.setMinimumSize(240, 180)
        self.crop_selector = QComboBox(window)
        self.crop_selector.setObjectName('pipelineWorkflowCropSelector')
        self.crop_selector.hide()
        self.crop_selector.currentIndexChanged.connect(self.select_crop)
        self.node_actions = QWidget(window)
        self.node_actions.setLayout(QHBoxLayout())
        self.node_actions.layout().setContentsMargins(0, 0, 0, 0)
        self.node_actions.hide()
        from seedvision.ui.color_histogram import HueHistogramView
        self.histogram = HueHistogramView(window)
        self.histogram.hide()
        self.selector.currentIndexChanged.connect(self.select_workflow)
        self.preview.view.region_selected.connect(self.region_selected)
        self.save_button.clicked.connect(self.save)
        self.export_button.clicked.connect(self.export)
        self.sync_buttons()

    @property
    def active(self):
        return self.workflow is not None

    def sync_buttons(self):
        busy = self._job is not None or bool(getattr(getattr(self.window, 'catalogue_nodes', None), 'busy', False))
        self.selector.setEnabled(not busy)
        self.window.analyze_action.setEnabled(not busy and not self.window._thread_pool.activeThreadCount()
            and self.window.image_view.image_path is not None)
        self.save_button.setEnabled(not busy)
        self.export_button.setEnabled(not busy and (self.result is not None if self.active else bool(self.window._analyses)))
        catalogue = getattr(self.window, 'catalogue_nodes', None)
        if catalogue is not None:
            self.remove_added_action.setEnabled(catalogue.selected_instance is not None and not busy)
            if catalogue.selected_instance is not None:
                self.export_button.setEnabled(not busy and catalogue.can_export_selected)
        self.selector.setToolTip(self.selector.currentText())
        documents = getattr(self.window, 'pipeline_documents', None)
        if documents is not None:
            self.remove_added_action.setEnabled(self.remove_added_action.isEnabled() and not documents.frozen)
        if hasattr(self.window, 'pipeline_backends'):
            self.window.pipeline_backends.refresh()
        self.export_menu_action.setEnabled(self.export_button.isEnabled())
        if self.active:
            self.display.sync_controls()

    def adjust_for_width(self, available):
        """Keep file commands accessible when a narrow pane uses large text."""
        layout = self.strip.layout()
        widgets = [layout.itemAt(index).widget() for index in range(layout.count())
                   if layout.itemAt(index).widget() is not None]
        shown = set(widgets)
        margins = layout.contentsMargins()
        def required():
            return (margins.left() + margins.right() + max(0, len(shown) - 1) * layout.spacing()
                    + sum(min(widget.maximumWidth(), max(widget.minimumWidth(), widget.sizeHint().width()))
                          for widget in shown))
        for widget in (self.workflow_label, self.export_button, self.save_button):
            if required() <= available:
                break
            shown.discard(widget)
        for widget in (self.workflow_label, self.export_button, self.save_button):
            if widget.isHidden() == (widget in shown):
                widget.setVisible(widget in shown)
        self.export_menu_action.setVisible(self.export_button.isHidden())

    def load_recipe(self, recipe):
        from seedvision.pipeline.workflows import create_workflow
        if self.active:
            self.sync_graph()
            self._copies[self.workflow.source_recipe.identity.recipe_sha256] = self.workflow
        workflow = create_workflow(recipe)
        if hasattr(self.window, 'pipeline_backends'):
            workflow = replace(workflow, backend_default=self.window.pipeline_backends.default)
        self.install(workflow)

    def select_workflow(self, index):
        from seedvision.pipeline.workflows import create_workflow
        recipe = self.selector.itemData(index)
        if self.active:
            self.sync_graph()
            self._copies[self.workflow.source_recipe.identity.recipe_sha256] = self.workflow
        else:
            self._default_selection = self.window._selected_pipeline_node
        workflow = None if recipe is None else (create_workflow(recipe) if index < self._catalogue_count else self._copies.get(recipe.identity.recipe_sha256, create_workflow(recipe)))
        if workflow is not None and index < self._catalogue_count and hasattr(self.window, 'pipeline_backends'):
            workflow = replace(workflow, backend_default=self.window.pipeline_backends.default)
        self.install(workflow)

    def install(self, workflow):
        if not self.active:
            self._default_selection = self.window._selected_pipeline_node
        self.stop()
        self._generation += 1
        self.workflow = workflow
        with QSignalBlocker(self.selector):
            for item in range(self.selector.count()-1, -1, -1):
                if self.selector.itemData(item, Qt.ItemDataRole.UserRole + 1) == 'current':
                    self.selector.removeItem(item)
            index = 0
            if workflow is not None:
                index = next((i for i in range(1, self.selector.count())
                    if self.selector.itemData(i).identity.recipe_sha256 == workflow.source_recipe.identity.recipe_sha256), -1)
                if index < 0:
                    self.selector.addItem('Current · '+workflow.recipe.title, workflow.source_recipe)
                    index = self.selector.count() - 1
                    self.selector.setItemData(index, 'current', Qt.ItemDataRole.UserRole + 1)
            self.selector.setCurrentIndex(index)
        self.clear_result_display()
        self.activity = ''
        self.display.mode, self.display.title = 'source', 'Source image'
        self.display.blend_available = False
        self.cache = PipelineAnalysisCache()
        self._correction_prepared = None
        self.graph = None if workflow is None else workspace_graph(workflow)
        self.window.pipeline_inspector.set_analysis_result(None)
        self.window.pipeline_inspector.enable_unified_optimization(False)
        canvas = self.window.pipeline_canvas
        if workflow is None:
            canvas.graph = self.window.pipeline
            canvas.rebuild_from_graph(self._default_selection)
            self.window.pipeline_inspector.enable_unified_optimization()
            self.window.image_display_stack.setCurrentWidget(self.window.image_view)
            self.facts.hide()
            self.node_actions.hide()
            self.histogram.hide()
            self.crop_selector.hide()
            self.window.baseline_section.show()
            if self._default_device_label is not None:
                self.window.compute_status_label.setText(self._default_device_label)
                self._default_device_label = None
            if self._toolbar_visibility is not None:
                for action, visible in self._toolbar_visibility:
                    action.setVisible(visible)
                self._toolbar_visibility = None
            if self._panel_visibility is not None:
                for widget, visible in self._panel_visibility:
                    widget.setVisible(visible)
                self._panel_visibility = None
            for control, enabled, tooltip in getattr(self, '_default_display_controls', ()):
                control.setEnabled(enabled)
                control.setToolTip(tooltip)
            self._default_display_controls = ()
        else:
            if self._default_device_label is None:
                self._default_device_label = self.window.compute_status_label.text()
            self.window.compute_status_label.setText('Not run (pipeline)')
            if self._toolbar_visibility is None:
                self._toolbar_visibility = [(action, action.isVisible()) for action in self.window.workflow_toolbar.actions()]
                self._default_display_controls = [(control, control.isEnabled(), control.toolTip()) for control in (
                    self.window.overlay_opacity_slider, self.window.annotation_opacity_slider,
                    self.window.show_material_references_checkbox, self.window.show_instance_annotations_checkbox,
                    self.window.paint_background_action, self.window.annotate_instances_action)]
            if self._panel_visibility is None:
                self._panel_visibility = [(widget, not widget.isHidden()) for widget in self.window.default_pipeline_details]
            for widget in self.window.default_pipeline_details:
                widget.hide()
            configure_native_parameters(self.graph, workflow.recipe, typed=True, editable_only=True)
            configure_foreground_regions(self.graph, workflow)
            canvas.graph = self.graph
            self._selected = workflow.display_state.get('selected_node', workflow.recipe.stages[0].stage_id)
            self._preferred_output = workflow.display_state.get('selected_output', '')
            canvas.rebuild_from_graph(self._selected)
            self.window.image_display_stack.setCurrentWidget(self.preview)
            self.window.calibration_section.hide()
            self.window.baseline_section.hide()
            self.select_node(self._selected)
            if workflow.display_state:
                self.window.image_workspace_action.setChecked(workflow.display_state.get('image_visible', True))
                self.window.pipeline_workspace_action.setChecked(workflow.display_state.get('pipeline_visible', True))
                self.strip.setVisible(self.window.pipeline_workspace_action.isChecked())
            else:
                self.window._show_split_workspace()
        self.window._set_project_dirty()
        if getattr(self.window, 'catalogue_nodes', None) is not None:
            self.window.catalogue_nodes.install_primary(self.graph if self.active else self.window.pipeline)
        self.sync_buttons()
        self.window._refresh_analysis_activity_panel()
        self.window._refresh_project_node()
        if workflow is None:
            self.window._rebuild_overlay_menu(str(self.window.overlay_combo.currentData() or 'raw_image'))
            mode = str(self.window.overlay_combo.currentData() or 'raw_image')
            self.window._sync_overlay_display_controls(mode)
            self.window.image_view.set_overlay_mode(mode)
            self.window._sync_background_controls()
            self.window._update_analysis_availability()
        documents = getattr(self.window, "pipeline_documents", None)
        if documents is not None:
            documents.loaded("Default seed analysis" if workflow is None else workflow.source_recipe.title,
                name=None if workflow is None else workflow.display_state.get("pipeline_name"))

    def sync_graph(self):
        if self.active:
            from seedvision.pipeline.workflows import workflow_from_graph
            saved = {key: dict(node.parameters) for key, node in self.graph.nodes.items()}
            try:
                for stage in self.workflow.recipe.stages:
                    values = self.graph.node(stage.stage_id).parameters
                    for parameter in stage.parameters:
                        value = values[parameter.parameter_id]
                        if parameter.value_type == 'json' and isinstance(value, str):
                            values[parameter.parameter_id] = json.loads(value)
                self.workflow = workflow_from_graph(self.workflow, self.graph)
                self.workflow = replace(self.workflow, display_state=dict(self.workflow.display_state) | {
                    'selected_node': self._selected, 'selected_output': self._preferred_output,
                    'context_layout': {key: [self.graph.node(key).x, self.graph.node(key).y]
                                       for key in CONTEXT_NODE_IDS},
                    'image_visible': self.window.image_workspace_action.isChecked(),
                    'pipeline_visible': self.window.pipeline_workspace_action.isChecked()})
            finally:
                for key, values in saved.items():
                    self.graph.node(key).parameters = values
        return self.workflow

    def refresh_saved(self, recipes=()):
        from seedvision.pipeline.workflows import create_workflow
        with QSignalBlocker(self.selector):
            while self.selector.count() > self._catalogue_count:
                self.selector.removeItem(self.selector.count()-1)
            for recipe in recipes:
                self.selector.addItem('Saved · '+recipe.title, recipe)
                self._copies[recipe.identity.recipe_sha256] = create_workflow(recipe)

    def clear_result_display(self):
        """Release computed display products without removing node controls."""
        self.result = None
        self.display.blend_available = False
        self.display.mode, self.display.title = 'source', 'Source image'
        self._preview_candidates = []
        self.preview.clear()
        self.facts.clear()
        self.facts.hide()
        self.histogram.set_chart(None)
        self.histogram.hide()
        self.crop_selector.hide()

    def changed(self):
        self.stop()
        self._generation += 1
        self.clear_result_display()
        self._started_at = None
        self.report_activity('Pipeline changed — run to update outputs.')
        self.window.compute_status_label.setText('Not run (pipeline)')
        self.crop_selector.hide()
        self.window._set_project_dirty()
        self.sync_buttons()

    def parameter_changed(self, node_id, key, value):
        guard = getattr(self.window, "_pipeline_editing_allowed", None)
        if guard is not None and not guard():
            return False
        node = self.graph.node(node_id)
        old = dict(node.parameters)
        if key == 'foreground_roi_bounds_xyxy':
            from seedvision.pipeline.workflows import with_foreground_region
            try:
                bounds = json.loads(value) if isinstance(value, str) else value
                self.workflow = with_foreground_region(self.workflow, node_id, bounds)
            except (ValueError, TypeError) as error:
                self.report_activity(f'Invalid foreground region: {error}')
                return False
            configure_foreground_regions(self.graph, self.workflow)
            self.graph.invalidate((node_id, *self.graph.downstream(node_id, recursive=True)))
            self.changed()
            self.select_node(node_id)
            return True
        try:
            self.graph.set_parameter(node_id, key, value)
            if key == 'channel' and self.workflow.recipe.stage(node_id).operation_id == 'threshold.custom_range':
                count = 1 if str(value).upper() in ('GRAY', 'GREY') else 4 if str(value).upper() == 'CMYK' else 3
                for bounds, fill in (('lower', 0), ('upper', 255)):
                    old_bounds = node.parameters[bounds]
                    parsed = json.loads(old_bounds) if isinstance(old_bounds, str) else list(old_bounds)
                    node.parameters[bounds] = json.dumps((parsed + [fill]*count)[:count])
            self.sync_graph()
        except (ValueError, TypeError, KeyError) as error:
            node.parameters = old
            self.window.statusBar().showMessage(str(error))
        else:
            self.changed()
        self.window.pipeline_inspector.set_node(node)
        self.window.pipeline_canvas.refresh()

    def reset_parameters(self, node_id):
        if node_id in CONTEXT_NODE_IDS:
            self.window._refresh_project_node()
            return
        guard = getattr(self.window, "_pipeline_editing_allowed", None)
        if guard is not None and not guard():
            return False
        from seedvision.pipeline.workflows import workflow_graph
        source = workflow_graph(type(self.workflow)(self.workflow.source_recipe, self.workflow.source_recipe))
        configure_native_parameters(source, self.workflow.source_recipe, typed=True, editable_only=True)
        old = dict(self.graph.node(node_id).parameters)
        self.graph.node(node_id).parameters = dict(source.node(node_id).parameters)
        try:
            from seedvision.pipeline.workflows import with_foreground_region
            if node_id in self.workflow.foreground_regions:
                self.workflow = with_foreground_region(self.workflow, node_id, None)
            self.sync_graph()
            configure_foreground_regions(self.graph, self.workflow)
        except Exception as error:
            self.graph.node(node_id).parameters = old
            self.window.statusBar().showMessage(str(error))
            return
        self.changed()
        self.graph.invalidate((node_id, *self.graph.downstream(node_id, recursive=True)))
        self.select_node(node_id)

    def connections_changed(self, affected=None):
        guard = getattr(self.window, "_pipeline_editing_allowed", None)
        if guard is not None and not guard():
            return False
        try:
            self.sync_graph()
        except (ValueError, TypeError) as error:
            self.window.statusBar().showMessage(str(error))
            self.graph = workspace_graph(self.workflow)
            configure_native_parameters(self.graph, self.workflow.recipe, typed=True, editable_only=True)
            self.window.pipeline_canvas.graph = self.graph
            self.window.pipeline_canvas.rebuild_from_graph(self._selected)
            self.window.catalogue_nodes.install_primary(self.graph)
            return
        self.changed()
        self.select_node(self._selected)

    def enabled_changed(self, node_id, enabled):
        if node_id in CONTEXT_NODE_IDS:
            self.select_node(node_id)
            return False
        guard = getattr(self.window, "_pipeline_editing_allowed", None)
        if guard is not None and not guard():
            return False
        node = self.graph.node(node_id)
        if not node.bypassable and not enabled:
            self.window.statusBar().showMessage('This operation has no supported typed pass-through.')
            self.select_node(node_id)
            return
        node.enabled = bool(enabled)
        self.graph.invalidate((node_id, *self.graph.downstream(node_id, recursive=True)), preserve_bypassed=True)
        self.connections_changed()

    def image_changed(self):
        from seedvision.ui.main_window import _path_identity
        path = self.window.image_view.image_path
        fingerprint = self.window._source_fingerprints.get(_path_identity(path)) if path is not None else None
        if path != self._image_path or fingerprint != self._image_fingerprint:
            self._image_path = path
            self._image_fingerprint = fingerprint
            if self.active:
                self.changed()
                self.graph.invalidate(tuple(self.graph.nodes), preserve_bypassed=True)
                self.cache = PipelineAnalysisCache()
                self._correction_prepared = None
                self.select_node(self._selected)
        self.sync_buttons()

    def run(self, target_node_id=None):
        if not self.active:
            self.window._analyze_current_image()
            return
        if target_node_id in CONTEXT_NODE_IDS:
            self.window._refresh_project_node()
            self.report_activity('Project inputs are synchronized with the current image.')
            return
        if self._job is not None or self.window._thread_pool.activeThreadCount():
            self.report_activity('Wait for the current work to finish.')
            return
        path = self.window.image_view.image_path
        if path is None:
            self.report_activity('Open an image before running this pipeline.')
            return
        try:
            workflow = self.sync_graph()
        except (ValueError, TypeError) as error:
            self.report_activity(f'Cannot run pipeline: {error}')
            return
        from seedvision.pipeline.workflows import run_workflow_path
        self.clear_result_display()
        cache = self.cache
        if workflow.recipe.recipe_id == 'plantcv_color_card_correct_color':
            try:
                context = self.window._capture_correction_project_context()
            except (ValueError, OSError) as error:
                self.report_activity(f'Cannot prepare correction: {error}')
                return
            self._correction_context = context
            binding = next((item for item in self.window._project_correction_bindings
                if context is not None and item.image_id == context.image.identifier
                and getattr(item, 'owner_id', None) is None
                and item.recipe_id == workflow.recipe.recipe_id
                and item.recipe_sha256 == workflow.recipe.identity.recipe_sha256), None)
            from seedvision.persistence.correction_inputs import resolve_correction_asset_path
            previous = self._correction_prepared
            if binding is not None and all(resolve_correction_asset_path(context.root, asset.path).is_file() for asset in binding.inputs):
                from seedvision.benchmarking.project_correction import run_project_correction
                self.start_job(lambda cancel: run_project_correction(workflow.recipe, context.root, binding,
                    context.image, context.source_path, cache=cache, previous=previous, cancel=cancel, working_workflow=workflow), self.correction_completed)
                return
            paths = dict(workflow.input_paths)
            paths['source_rgb'] = str(path)
            if set(paths) != {'target_rgb', 'target_mask', 'source_rgb', 'source_mask'}:
                self.report_activity('Select the target image and both chip-label masks in the correction node.')
                return
            from seedvision.benchmarking.native_correction_inputs import run_native_correction_files
            self.start_job(lambda cancel: run_native_correction_files(workflow.recipe, paths,
                cache=cache, previous=previous, cancel=cancel, working_workflow=workflow), self.correction_completed)
            return
        from seedvision.persistence.reference_models import model_asset_paths
        try:
            models = model_asset_paths(self.window._root, workflow.recipe,
                self.window._project_reference_model_assets)
        except (ValueError, OSError) as error:
            self.report_activity(f'Cannot prepare pipeline models: {error}')
            return
        self.start_job(lambda cancel: run_workflow_path(workflow, path, cache=cache,
            cancel=cancel, target_node_id=target_node_id, model_asset_paths=models,
            progress=self.progress_callback), self.run_completed)

    def report_activity(self, text):
        self.activity = text
        self.window.statusBar().showMessage(text)
        self.window._refresh_analysis_activity_panel()

    def progress_callback(self, stage_id, status, reason=''):
        # Only immutable text crosses from the calculation thread into Qt.
        job = self._job
        if job is not None:
            job.signals.progress.emit(stage_id, status, reason)

    def node_progress(self, job, stage_id, status, reason):
        if self._job is not job or job.token != (self._generation, self.window.image_view.image_path):
            return
        title = self.graph.node(stage_id).title if stage_id in self.graph.nodes else 'Preparing image'
        if stage_id in self.graph.nodes:
            node = self.graph.node(stage_id)
            node.status = NodeStatus.RUNNING if status == 'running' else NodeStatus.COMPLETE if status in ('executed', 'provided', 'empty', 'reused', 'bypassed') else NodeStatus.BLOCKED
            node.status_detail = status + (': ' + reason if reason else '')
            self.window.pipeline_canvas.refresh()
            self.window.pipeline_inspector.refresh_status()
        self.report_activity(f'{title}: {status}' + (f'\n{reason}' if reason else ''))

    def start_job(self, function, completion):
        token = (self._generation, self.window.image_view.image_path)
        job = _Job(function, token)
        self._job = job
        self._started_at = monotonic()
        job.signals.completed.connect(lambda result: self.job_completed(job, result, completion))
        job.signals.progress.connect(lambda stage, status, reason: self.node_progress(job, stage, status, reason))
        self.window.pipeline_inspector.setEnabled(False)
        self.window.pipeline_canvas.setEnabled(False)
        self.report_activity('Preparing image…')
        self.sync_buttons()
        self.window._update_analysis_availability()
        self.window._thread_pool.start(job)

    def job_completed(self, job, result, completion):
        if self._job is not job:
            return
        self._job = None
        self.window.pipeline_inspector.setEnabled(True)
        self.window.pipeline_canvas.setEnabled(True)
        if job.cancel.is_set() or job.token != (self._generation, self.window.image_view.image_path):
            self.report_activity('Pipeline cancelled; obsolete outputs discarded.')
        elif isinstance(result, Exception):
            self.report_activity(f'Pipeline failed: {result}')
            self.preview.setText(self.activity)
            self.facts.setPlainText(self.activity)
            self.facts.show()
        else:
            completion(result)
        self.sync_buttons()
        self.window._update_analysis_availability()
        self._started_at = None

    def stop(self):
        if self._job is not None:
            self._job.cancel.set()
            self.report_activity('Stopping pipeline; waiting for the current calculation to finish safely…')
        else:
            self.window._stop_current_work()

    def run_completed(self, result):
        self.result = result
        self.window.compute_status_label.setText('CUDA · '+result.status)
        if self.workflow.recipe.recipe_id == 'plantcv_color_card_correct_color' and self._correction_context is not None:
            from seedvision.persistence.correction_inputs import binding_for_result, select_correction_binding
            try:
                binding = binding_for_result(self.workflow.recipe, self._correction_context.image.identifier, result)
                self.window._project_correction_bindings = select_correction_binding(self.window._project_correction_bindings, binding)
                self.window._set_project_dirty()
            except (ValueError, OSError) as error:
                self.window.statusBar().showMessage(f'Correction result available; project binding failed: {error}')
        for item in result.nodes:
            node = self.graph.node(item.stage_id)
            node.status = (NodeStatus.BYPASSED if item.status == 'bypassed' else NodeStatus.COMPLETE if item.status in ('provided', 'executed', 'empty')
                else NodeStatus.WARNING if item.status == 'partial' else NodeStatus.BLOCKED)
            node.status_detail = ('Reused · ' if item.reused else '') + item.status + (': ' + item.reason if item.reason else '')
        self.window.pipeline_canvas.refresh()
        self.select_node(self._selected)
        failed = next((node for node in result.nodes if node.status in ('failed', 'unavailable', 'partial')), None)
        if failed is not None:
            title = self.graph.node(failed.stage_id).title
            self.report_activity(f'{title}: {failed.status}\n{failed.reason}')
            self.window.pipeline_canvas.select_node(failed.stage_id)
        else:
            elapsed = monotonic() - self._started_at if self._started_at is not None else 0
            self.report_activity(f'Pipeline {result.status} in {elapsed:.1f}s. Select a node to inspect outputs.')
            # A first run should show its useful final raster, not an unchanged input.
            if self._selected in CONTEXT_NODE_IDS or self.workflow.recipe.stage(self._selected).operation_id.startswith('external.'):
                target = next((node.stage_id for node in reversed(result.nodes)
                    if any(torch.is_tensor(value) and value.ndim in (2, 3) for value in node.outputs.values())), self._selected)
                self.window.pipeline_canvas.select_node(target)

    def correction_completed(self, completion):
        result, prepared = completion
        self._correction_prepared = prepared
        self.run_completed(result)

    def select_node(self, node_id):
        if node_id not in self.graph.nodes:
            return
        if node_id != self._selected:
            self._preferred_output = ''
            self.window._set_project_dirty()
        self._selected = node_id
        if node_id in CONTEXT_NODE_IDS:
            if node_id == 'species_reference_library':
                self.window._context_library_key = None
            self.window._refresh_project_node()
            self.window.pipeline_inspector.set_node(self.graph.node(node_id))
            self.window.pipeline_inspector.overlay_combo.hide()
            self.replay_action.setEnabled(False)
            self.node_actions.hide()
            self.histogram.hide()
            self.facts.hide()
            self.crop_selector.hide()
            self.display.show_source()
            if hasattr(self.window, 'pipeline_backends'):
                self.window.pipeline_backends.refresh()
            return
        from seedvision.benchmarking.native_stage_replay import SUPPORTED_REPLAY_OPERATIONS
        self.replay_action.setEnabled(self.workflow.recipe.stage(node_id).operation_id in SUPPORTED_REPLAY_OPERATIONS and self._job is None)
        self.window.pipeline_inspector.set_node(self.graph.node(node_id))
        self.window.pipeline_inspector.overlay_combo.hide()
        stage = self.workflow.recipe.stage(node_id)
        self.update_node_actions(stage)
        if hasattr(self.window, "pipeline_backends"):
            self.window.pipeline_backends.refresh()
        self.histogram.hide()
        self.facts.clear()
        self.facts.hide()
        model = next((p.value for p in stage.parameters if p.parameter_id == 'model_source'), None)
        if model is not None:
            self.facts.append('Selected model: '+str(model.get('path', '')))
            if 'class_names' in model:
                self.facts.append('Classes: '+', '.join(model['class_names']))
            if 'cluster_count' in model:
                self.facts.append(f"Clusters: {model['cluster_count']}")
            self.facts.show()
        self.crop_selector.hide()
        self._preview_candidates = []
        self.preview.clear()
        if self.result is None:
            self.display.show_source()
            return
        try:
            node = self.result.node(node_id)
        except StopIteration:
            self.preview.setText('This input is bound to the selected image or file.')
            return
        facts = [f'{stage.title}: {node.status}', 'Backend: ' + ('PlantCV public call' if getattr(node, 'backend', 'native') == 'plantcv' else 'Native'), node.reason]
        for name, value in node.outputs.items():
            if torch.is_tensor(value):
                facts.append(f'{name}: {tuple(value.shape)} · {value.dtype}')
            elif name == 'observations' and hasattr(value, 'labels'):
                for item in value.labels[:20]:
                    facts.append(f'{item.sample}: {item.pixel_count} pixels')
                    facts.extend(f'{stat.variable}: {stat.value:.6g} {stat.units}' + ('' if stat.available else ' · unavailable') for stat in item.statistics)
                if len(value.labels) > 20:
                    facts.append(f'{len(value.labels)} objects measured; first 20 shown. Export contains all measurements.')
            elif name == 'observations' and isinstance(value, Mapping):
                for label, measurements in list(value.items())[:20]:
                    facts.append(str(label))
                    for variable, record in list(measurements.items())[:12]:
                        if isinstance(record, Mapping) and 'value' in record:
                            selected = record['value']
                            rendered = f'{selected:.6g}' if isinstance(selected, float) else str(selected)
                            facts.append(f"{variable.replace('_', ' ')}: {rendered} {record.get('scale', '')}")
                if len(value) > 20:
                    facts.append(f'{len(value)} objects measured; first 20 shown. Export contains all measurements.')
            elif name == 'histogram':
                self.histogram.set_chart(value)
                self.histogram.show()
            elif isinstance(value, Mapping):
                facts.append(name.replace('_', ' ')+': '+', '.join(str(key) for key in list(value)[:12]))
                for key, selected in list(value.items())[:12]:
                    if type(selected) in (str, bool, int, float) and len(str(selected)) < 120:
                        facts.append(f'{str(key).replace("_", " ")}: {selected}')
            elif isinstance(value, (tuple, list)):
                facts.append(f'{name.replace("_", " ")}: {len(value)} entries')
            elif type(value) in (str, bool, int, float):
                facts.append(f'{name.replace("_", " ")}: {value}')
            else:
                facts.append(f'{name.replace("_", " ")}: {type(value).__name__}')
        self.facts.setPlainText('\n'.join(facts))
        self.facts.show()
        if node.operation_id == 'smpts_paper.crops' and 'packed_rgb' in node.outputs:
            with QSignalBlocker(self.crop_selector):
                self.crop_selector.clear()
                self.crop_selector.addItem('Full-frame final labels')
                for crop in node.outputs['crops']:
                    self.crop_selector.addItem(f"Crop {crop['label_id']} · parent {crop['parent_id']}")
                preferred = self.crop_selector.findText(self._preferred_output)
                self.crop_selector.setCurrentIndex(max(0, preferred))
            self.crop_selector.show()
            self.select_crop(max(0, preferred))
            return
        ports = {port.name: port for port in stage.ports if port.direction == 'output'}
        candidates = []
        for name, value in node.outputs.items():
            if name not in ports or ports[name].data_type == 'PeakCoordinates':
                continue
            values = value.items() if isinstance(value, dict) or hasattr(value, 'items') else [(name, value)]
            for title, raster in values:
                if torch.is_tensor(raster) and raster.ndim in (2, 3) and (raster.ndim == 2 or raster.shape[-1] == 3):
                    candidates.append((str(title), raster, ports[name].semantics))
        selected = next((candidate for candidate in candidates if candidate[0] == self._preferred_output), candidates[0] if candidates else None)
        if selected is None:
            self.preview.setText('This node produces measurements or metadata. See the node results.')
            return
        name, raster, semantics = selected
        self.render(raster, semantics)
        frame = semantics.get('frame', self.result.coordinate_frame.frame_id)
        color = semantics.get('color_order', semantics.get('representation', 'raster'))
        self.facts.append(f'Preview: {name} · {frame} · {color}')
        if len(candidates) > 1:
            self._preview_candidates = candidates
            with QSignalBlocker(self.crop_selector):
                self.crop_selector.clear()
                self.crop_selector.addItems([name for name, raster, semantics in candidates])
                self.crop_selector.setCurrentIndex(max(0, self.crop_selector.findText(self._preferred_output)))
            self.crop_selector.show()

    def update_node_actions(self, stage):
        layout = self.node_actions.layout()
        while layout.count():
            widget = layout.takeAt(0).widget()
            if widget is not None:
                widget.deleteLater()
        actions = []
        operation = stage.operation_id
        from seedvision.pipeline.workflows import FOREGROUND_OPERATIONS
        if operation in FOREGROUND_OPERATIONS:
            button = QPushButton('Select foreground region…', self.node_actions)
            button.setToolTip('Draw a rectangle around the seeds in the source image. Pixels outside remain background.')
            source_frame = all(port.semantics.get('frame') in (None, 'source.decoded')
                for port in stage.ports if port.direction == 'input')
            button.setEnabled(source_frame)
            if not source_frame:
                button.setToolTip('This node uses a transformed frame. Enter region bounds in its input pixels in Settings.')
            button.clicked.connect(self.select_region)
            layout.addWidget(button)
        from seedvision.benchmarking.native_reader_recipe import BASE_IDS as READER_BASE_IDS
        if operation == 'external.image_inputs' and self.workflow.recipe.recipe_id in READER_BASE_IDS:
            actions.append(('Grayscale input mode…', 'native_reader_recipe', 'build_native_reader_recipe'))
        if operation == 'filter.median_blur':
            actions.append(('Median ROI…', 'native_median_roi_recipe', 'build_native_median_roi_recipe'))
        if operation in ('morphology.erode', 'morphology.dilate'):
            actions.append(('Morphology ROI…', 'native_morphology_roi_recipe', 'build_native_morphology_roi_recipe'))
        if operation == 'morphology.fill':
            actions.append(('Small-object ROI…', 'native_fill_roi_recipe', 'build_native_fill_roi_recipe'))
        if operation == 'reference.internal.watershed.masked_flood':
            actions.append(('Watershed diagnostic', 'native_watershed_diagnostic_recipe', 'build_native_watershed_diagnostic_recipe'))
        for title, module, factory in actions:
            button = QPushButton(title, self.node_actions)
            button.clicked.connect(lambda checked=False, module=module, factory=factory: self.add_variant(module, factory))
            layout.addWidget(button)
        if any(p.parameter_id == 'model_source' for p in stage.parameters):
            button = QPushButton('Select model…', self.node_actions)
            button.clicked.connect(self.choose_model)
            layout.addWidget(button)
            train = QPushButton('Train…', self.node_actions)
            train.clicked.connect(lambda checked=False: self.train_model('kmeans' if 'kmeans' in self.workflow.recipe.recipe_id else 'bayes'))
            layout.addWidget(train)
        if self.workflow.recipe.recipe_id == 'plantcv_color_card_correct_color':
            for port, title in (('target_rgb', 'Target image…'), ('source_mask', 'Source chip mask…'), ('target_mask', 'Target chip mask…')):
                button = QPushButton(title, self.node_actions)
                button.setToolTip(self.workflow.input_paths.get(port, 'Required input; source image is the selected project image.'))
                button.clicked.connect(lambda checked=False, port=port: self.choose_input(port))
                layout.addWidget(button)
        documents = getattr(self.window, "pipeline_documents", None)
        if documents is not None and documents.frozen:
            for index in range(layout.count()):
                layout.itemAt(index).widget().setEnabled(False)
        self.node_actions.setVisible(bool(layout.count()))

    def select_region(self):
        if not self.window._pipeline_editing_allowed():
            return
        stage = self.workflow.recipe.stage(self._selected)
        if any(port.semantics.get('frame') not in (None, 'source.decoded')
                for port in stage.ports if port.direction == 'input'):
            self.report_activity('Select region requires source-image coordinates. Enter bounds in this node input frame instead.')
            return
        if self.window.image_view._source_pixmap is None:
            self.report_activity('Open an image before selecting a foreground region.')
            return
        self._region_node = self._selected
        self.display.show_source()
        self.preview.view.selecting_region = True
        self.preview.view.setCursor(Qt.CursorShape.CrossCursor)
        self.preview.message.setText('Drag a rectangle around the seed foreground. The original image dimensions are preserved.')

    def region_selected(self, rectangle):
        import math
        pixmap = self.window.image_view._source_pixmap
        scene = self.preview.view.sceneRect()
        if pixmap is None or rectangle.isEmpty() or scene.isEmpty():
            return
        sx, sy = pixmap.width()/scene.width(), pixmap.height()/scene.height()
        bounds = [max(0, math.floor(rectangle.left()*sx)), max(0, math.floor(rectangle.top()*sy)),
            min(pixmap.width(), math.ceil(rectangle.right()*sx)), min(pixmap.height(), math.ceil(rectangle.bottom()*sy))]
        self.parameter_changed(self._region_node, 'foreground_roi_bounds_xyxy', json.dumps(bounds))
        self.report_activity(f'Foreground region: {bounds}. Run pipeline to calculate it.')

    def choose_input(self, port):
        guard = getattr(self.window, "_pipeline_editing_allowed", None)
        if guard is not None and not guard():
            return False
        path, _ = QFileDialog.getOpenFileName(self.window, 'Select '+port.replace('_', ' '), '', 'Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)')
        if path:
            try:
                context = self.window._capture_correction_project_context()
            except (ValueError, OSError) as error:
                self.window.statusBar().showMessage(str(error))
                return
            if context is not None:
                from seedvision.persistence.correction_inputs import resolve_correction_asset_path
                retained = next((binding for binding in self.window._project_correction_bindings
                    if binding.image_id == context.image.identifier and binding.recipe_id == self.workflow.recipe.recipe_id
                    and getattr(binding, 'owner_id', None) is None), None)
                if retained is not None:
                    paths = dict(self.workflow.input_paths)
                    for asset in retained.inputs:
                        if asset.role != 'source_rgb' and asset.path is not None:
                            portable = resolve_correction_asset_path(context.root, asset.path)
                            paths[asset.role] = str(portable if portable.is_file() else asset.origin_path)
                    self.workflow = replace(self.workflow, input_paths=paths)
                self.window._project_correction_bindings = tuple(binding for binding in self.window._project_correction_bindings
                    if not (binding.image_id == context.image.identifier and binding.recipe_id == self.workflow.recipe.recipe_id
                            and getattr(binding, 'owner_id', None) is None))
            values = dict(self.workflow.input_paths)
            values[port] = str(Path(path).resolve())
            self.workflow = replace(self.workflow, input_paths=values)
            self.changed()
            self.select_node(self._selected)

    def choose_model(self):
        guard = getattr(self.window, "_pipeline_editing_allowed", None)
        if guard is not None and not guard():
            return False
        stage = self.workflow.recipe.stage(self._selected)
        path, _ = QFileDialog.getOpenFileName(self.window, 'Select frozen model')
        if not path:
            return
        if 'kmeans' in self.workflow.recipe.recipe_id:
            from seedvision.benchmarking.native_kmeans_recipe import model_source_for_path
        else:
            from seedvision.benchmarking.native_naive_bayes_recipe import model_source_for_path
        try:
            value = model_source_for_path(path)
        except (ValueError, OSError) as error:
            self.window.statusBar().showMessage(str(error))
            return
        self.graph.node(stage.stage_id).parameters['model_source'] = value
        self.sync_graph()
        self.changed()
        self.select_node(stage.stage_id)

    def train_model(self, kind):
        if self._job is not None or self.window._thread_pool.activeThreadCount():
            self.window.statusBar().showMessage('Wait for current work before training a model.')
            return
        from PySide6.QtWidgets import QDialog
        if kind == 'kmeans':
            from seedvision.ui.native_kmeans_training_dialog import NativeKMeansTrainingDialog
            dialog = NativeKMeansTrainingDialog(self.window)
        else:
            from seedvision.ui.native_naive_bayes_training_dialog import NativeNaiveBayesTrainingDialog
            dialog = NativeNaiveBayesTrainingDialog(self.window)
        selected = None
        try:
            if dialog.exec() == QDialog.DialogCode.Accepted:
                if kind == 'kmeans' and dialog.saved_model_path:
                    from seedvision.benchmarking.native_kmeans_recipe import model_source_for_path
                    selected = model_source_for_path(dialog.saved_model_path)
                elif kind == 'bayes':
                    selected = dialog.saved_model_source
        except (ValueError, OSError) as error:
            self.window.statusBar().showMessage(str(error))
        finally:
            dialog._completion = None
            if hasattr(dialog, '_cached_rows'):
                dialog._cached_rows = None
            dialog.deleteLater()
        expected = 'plantcv_kmeans_predict_masks' if kind == 'kmeans' else 'plantcv_naive_bayes_classifier'
        if selected is not None and self.active and self.workflow.recipe.recipe_id == expected:
            if not self.window._pipeline_editing_allowed():
                self.window.statusBar().showMessage('Model saved; pipeline is frozen, so its selection was kept.')
                return
            stage = next(stage for stage in self.workflow.recipe.stages if any(p.parameter_id == 'model_source' for p in stage.parameters))
            self.graph.node(stage.stage_id).parameters['model_source'] = selected
            self.sync_graph()
            self.changed()
            self.select_node(stage.stage_id)

    def select_crop(self, index):
        if self.result is None or index < 0:
            return
        self._preferred_output = self.crop_selector.itemText(index)
        self.window._set_project_dirty()
        if self._preview_candidates:
            name, raster, semantics = self._preview_candidates[index]
            self.render(raster, semantics)
            return
        node = self.result.node(self._selected)
        if node.operation_id != 'smpts_paper.crops':
            return
        if index == 0:
            self.render(node.outputs['labels'], {})
        else:
            crop = node.outputs['crops'][index - 1]
            height, width = crop['crop_shape']
            offset = crop['offset_pixels'] * 3
            self.render(node.outputs['packed_rgb'][offset:offset + height*width*3].reshape(height, width, 3), {'color_order': 'RGB'})

    def render(self, raster, semantics):
        if self.active and self.display.mode == 'annotations':
            self.window._stop_reference_point_editing()
        image = native_preview_image(raster, semantics)
        self.preview.setPixmap(QPixmap.fromImage(image))
        self.preview.message.setText(f'Node output · {raster.shape[1]}×{raster.shape[0]} source pixels · sampled to {image.width()}×{image.height()} display pixels · drag / scroll to pan / zoom')
        if semantics.get('unit') == 'pixel' and 'distanceTransform' in semantics.get('source', ''):
            self.preview.message.setText(self.preview.message.text() + ' · distance in pixels; logarithmic colour display; black = zero')
        self.window.image_display_stack.setCurrentWidget(self.preview)
        catalogue = getattr(self.window, 'catalogue_nodes', None)
        if self.active and self.result is not None and not (catalogue and catalogue.handles(catalogue._selected)):
            self.display.rendered(raster, semantics)
        elif self.active:
            self.display.mode, self.display.title = 'output', 'Added node output'
            self.display.blend_available = False
            self.display.sync_controls()

    def save(self):
        documents = getattr(self.window, 'pipeline_documents', None)
        if documents is not None:
            return documents.choose_pipeline_save()
        self.window._save_project()

    def open_saved(self):
        path, _ = QFileDialog.getOpenFileName(self.window, 'Open saved workflow', '', 'SeedFiddle workflow (*.seedfiddle-workflow.json);;SeedFiddle recipe (*.seedfiddle-recipe.json)')
        if not path:
            return
        from seedvision.benchmarking.recipes import RecipeSpec
        from seedvision.pipeline.workflows import create_workflow, PipelineWorkflow
        from seedvision.pipeline.backend_execution import validate_backend_selection
        try:
            content = Path(path).read_bytes()
            if 'source_recipe' in json.loads(content):
                workflow = PipelineWorkflow.from_json(content)
                recipe = workflow.source_recipe
            else:
                recipe = RecipeSpec.from_json(content)
                workflow = create_workflow(recipe)
            validate_backend_selection(workflow)
        except (ValueError, OSError, NotImplementedError) as error:
            self.window.statusBar().showMessage(str(error))
            return
        self.selector.addItem('Saved · '+recipe.title, recipe)
        self._copies[recipe.identity.recipe_sha256] = workflow
        self.selector.setCurrentIndex(self.selector.count()-1)

    def export(self):
        catalogue = getattr(self.window, 'catalogue_nodes', None)
        if catalogue is not None and catalogue.selected_instance is not None:
            catalogue.export_selected()
            return
        if not self.active:
            self.window._review_results()
            return
        if self.result is None or self._job is not None:
            return
        directory = QFileDialog.getExistingDirectory(self.window, 'Export workflow run')
        if directory:
            from seedvision.export.native_results import export_native_recipe_result
            recipe, result = self.workflow.recipe, self.result
            self.start_job(lambda cancel: export_native_recipe_result(directory, recipe, result, cancel=cancel),
                lambda path: self.window.statusBar().showMessage(f'Workflow exported to {path}'))

    def show_source(self):
        if self.window.catalogue_nodes.show_source_selected():
            return
        if self.active:
            recipe = self.workflow.source_recipe
            provenance = recipe.provenance
            QMessageBox.information(self.window, 'Workflow source',
                f'{recipe.title}\n{recipe.description}\n\n'
                f'Origin: {provenance.mode.value}\nBackend: {provenance.backend_id} {provenance.backend_version}\n'
                f'Evidence: {provenance.evidence_status.value}\n{provenance.evidence_note}\n\n'
                'Working parameters and wiring are saved separately from source evidence.')

    def show_qc(self):
        from seedvision.ui.method_qc_dialog import MethodQCDialog
        directory = QFileDialog.getExistingDirectory(self.window, 'Review exported workflow QC')
        if directory:
            try:
                from seedvision.export.method_qc import load_method_qc, FILENAME
                MethodQCDialog(load_method_qc(Path(directory)/FILENAME), self.window).exec()
            except (ValueError, OSError) as error:
                self.window.statusBar().showMessage(str(error))

    def replay(self):
        if not self.active or self._job is not None:
            return
        directory = QFileDialog.getExistingDirectory(self.window, 'Replay saved workflow stage')
        if directory:
            from seedvision.benchmarking.native_stage_replay import replay_native_stage
            selected, identity = self._selected, self.workflow.recipe.identity
            self.start_job(lambda cancel: replay_native_stage(directory, selected, cancel=cancel,
                expected_recipe_identity=identity), self.replay_completed)

    def replay_completed(self, result):
        self.facts.setPlainText('Saved stage replay\n'+str(result))
        self.facts.show()
        stage = self.workflow.recipe.stage(self._selected)
        ports = {p.name: p for p in stage.ports if p.direction == 'output'}
        for name, value in result.outputs.items():
            if name in ports and torch.is_tensor(value) and value.ndim in (2, 3) and (value.ndim == 2 or value.shape[-1] == 3):
                self.render(value, ports[name].semantics)
                break

    def add_variant(self, module, factory):
        guard = getattr(self.window, "_pipeline_editing_allowed", None)
        if guard is not None and not guard():
            return False
        if not self.active or self._job is not None:
            return
        from importlib import import_module
        from seedvision.pipeline.workflows import create_workflow, workflow_graph, workflow_from_graph
        try:
            current = self.sync_graph()
            if current.disabled_nodes or current.recipe.dependencies != current.source_recipe.dependencies:
                raise ValueError('Restore the source wiring and enabled nodes before adding this operation variant.')
            recipe = getattr(import_module('seedvision.benchmarking.'+module), factory)(current.recipe)
            upgraded = create_workflow(recipe)
            graph = workflow_graph(upgraded)
            # Factory variants retain stage IDs. Preserve existing selected
            # values and layout before introducing any new operation controls.
            for identifier, old_node in self.graph.nodes.items():
                if identifier in graph.nodes:
                    node = graph.node(identifier)
                    node.x, node.y = old_node.x, old_node.y
            upgraded = workflow_from_graph(upgraded, graph)
            from seedvision.pipeline.workflows import with_foreground_region
            for stage_id, bounds in current.foreground_regions.items():
                upgraded = with_foreground_region(upgraded, stage_id, bounds)
            upgraded = replace(upgraded, display_state=current.display_state)
            self.install(upgraded)
        except (ValueError, TypeError) as error:
            self.window.statusBar().showMessage(str(error))
