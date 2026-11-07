"""The Pipeline list, its file commands and the added-node output display."""
from __future__ import annotations

import math

import numpy as np
import torch
from PySide6.QtCore import QObject, QSignalBlocker, Qt
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (QComboBox, QFrame, QGraphicsScene, QGraphicsView, QHBoxLayout,
    QLabel, QMenu, QSizePolicy, QTextBrowser, QToolButton, QVBoxLayout, QWidget)

from seedvision.pipeline.default_pipelines import CELLPOSE_SEED_ANALYSIS, GRAPH_PIPELINES, graph_pipeline_of
from seedvision.ui.canvas_controls import COMPACT_ICON_SIZE, load_icon, rename_icon, save_icon, style_canvas_control_bar


DEFAULTS_HEADING = 'Default pipelines'
SAVED_HEADING = 'Saved in this project'
# Item role: 'heading', 'template' (a default pipeline), 'saved-heading' or
# 'custom' (a named, editable copy saved in the project).
ENTRY_KIND = Qt.ItemDataRole.UserRole + 1
# The list shrinks towards its minimum only in a pane too narrow for it.
SELECTOR_WIDTH = 260
SELECTOR_MINIMUM_WIDTH = 120


def raster_preview_image(raster, *, maximum=1024):
    """A sampled display copy of a CUDA raster; only the sample leaves the device."""
    stride = max(1, math.ceil(raster.shape[0] / maximum), math.ceil(raster.shape[1] / maximum))
    sampled = raster[::stride, ::stride]
    if sampled.dtype == torch.bool:
        sampled = sampled.to(torch.uint8) * 255
    elif sampled.is_floating_point():
        lower, upper = sampled.min(), sampled.max()
        sampled = ((sampled - lower) / (upper - lower).clamp_min(1e-12) * 255).to(torch.uint8)
    elif sampled.dtype != torch.uint8:
        # Distinct colours for label IDs; zero stays black.
        ids = sampled.to(torch.int64)
        sampled = torch.stack(((ids * 53) % 251, (ids * 97) % 253, (ids * 193) % 255), -1).to(torch.uint8)
    array = np.ascontiguousarray(sampled.cpu().numpy())
    format = QImage.Format.Format_RGB888 if array.ndim == 3 else QImage.Format.Format_Grayscale8
    return QImage(array.data, array.shape[1], array.shape[0], array.strides[0], format).copy()


class _RasterView(QGraphicsView):
    def wheelEvent(self, event):
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        self.scale(factor, factor)
        event.accept()


class NodeOutputPreview(QWidget):
    """Sampled Qt display of an added node's output, with pan, zoom and fit."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        header = QFrame(self)
        header.setFixedHeight(38)
        controls = QHBoxLayout(header)
        controls.setContentsMargins(8, 4, 8, 4)
        controls.setSpacing(5)
        self.header_layout = controls
        controls.addStretch(1)
        fit = QToolButton(header)
        fit.setText('Fit')
        fit.clicked.connect(self.fit_image)
        actual = QToolButton(header)
        actual.setText('1:1')
        actual.clicked.connect(lambda: self.view.resetTransform())
        controls.addWidget(fit)
        controls.addWidget(actual)
        style_canvas_control_bar(header, 'workflowPreviewHeader')
        layout.addWidget(header)
        self.message = QLabel(self)
        self.message.setWordWrap(True)
        self.message.setTextFormat(Qt.TextFormat.PlainText)
        self.message.setContentsMargins(8, 3, 8, 3)
        layout.addWidget(self.message)
        self.view = _RasterView(self)
        self.view.setScene(QGraphicsScene(self.view))
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        layout.addWidget(self.view, 1)

    def dock_header_controls(self, controls):
        """Show controls shared with another display at the left of this header."""
        self.header_layout.insertWidget(0, controls)
        controls.offer_width(None)

    def clear(self):
        self.view.scene().clear()
        self.message.clear()

    def setText(self, text):
        self.clear()
        self.message.setText(text)

    def setPixmap(self, pixmap):
        self.clear()
        item = self.view.scene().addPixmap(pixmap)
        self.view.scene().setSceneRect(item.boundingRect())
        self.message.setText('Node output · drag to pan · scroll to zoom')
        self.fit_image()

    def fit_image(self):
        self.view.fitInView(self.view.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)


class PipelineWorkflowController(QObject):
    """Choose the default pipeline or a saved copy for the shared graph."""
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.strip = QWidget(window)
        self.strip.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.strip.setObjectName('pipelineWorkflowStrip')
        row = QHBoxLayout(self.strip)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(3)
        self.workflow_label = QLabel('Pipeline', self.strip)
        row.addWidget(self.workflow_label)
        self.selector = QComboBox(self.strip)
        self.selector.setFixedWidth(SELECTOR_WIDTH)
        self.selector.setMinimumContentsLength(8)
        self.selector.view().setMinimumWidth(360)
        self.selector.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.selector.setObjectName('pipelineWorkflowSelector')
        self._add_heading(DEFAULTS_HEADING)
        for pipeline in GRAPH_PIPELINES:
            self.selector.addItem(pipeline.title, pipeline)
            self.selector.setItemData(self.selector.count() - 1, 'template', ENTRY_KIND)
        with QSignalBlocker(self.selector):
            self.selector.setCurrentIndex(self.index_of(None))
        row.addWidget(self.selector)
        self.rename_button = self._icon_button(rename_icon(), 'Rename pipeline')
        self.rename_button.setToolTip('Rename pipeline. Default pipelines cannot be renamed, so this offers an editable copy instead.')
        self.load_button = self._icon_button(load_icon(), 'Load pipeline')
        self.load_button.setToolTip('Load pipeline from a file, as an editable copy.')
        self.save_button = self._icon_button(save_icon(), 'Save pipeline')
        self.save_button.setToolTip('Save pipeline to a file under a new name, and carry on editing that copy.')
        self.export_button = QToolButton(self.strip)
        self.export_button.setText('Export')
        row.addWidget(self.export_button)
        self.more_button = QToolButton(self.strip)
        self.more_button.setText('…')
        self.more_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(self.more_button)
        self.export_menu_action = menu.addAction('Export run…', self.export)
        self.export_menu_action.setVisible(False)
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
        self.preview = NodeOutputPreview(window)
        self.preview.setText('Select an added node to inspect its outputs.')
        self.preview.setObjectName('pipelineWorkflowPreview')
        self.preview.setMinimumSize(240, 180)
        self.selector.currentIndexChanged.connect(self.select_pipeline)
        self.export_button.clicked.connect(self.export)
        self.sync_buttons()

    def _icon_button(self, icon, name):
        """A compact icon-only command; its tooltip and accessible name say what it does."""
        button = QToolButton(self.strip)
        button.setIcon(icon)
        button.setIconSize(COMPACT_ICON_SIZE)
        button.setStyleSheet('padding: 2px 3px;')
        button.setAccessibleName(name)
        self.strip.layout().addWidget(button)
        return button

    def _add_heading(self, text, position=None, kind='heading'):
        """A bold, unselectable group heading in the Pipeline list."""
        position = self.selector.count() if position is None else position
        self.selector.insertItem(position, text.upper())
        self.selector.setItemData(position, kind, ENTRY_KIND)
        item = self.selector.model().item(position)
        item.setFlags(Qt.ItemFlag.NoItemFlags)
        font = item.font()
        font.setBold(True)
        item.setFont(font)

    def _entries(self, *kinds):
        return [index for index in range(self.selector.count())
                if self.selector.itemData(index, ENTRY_KIND) in kinds]

    @property
    def graph_pipeline(self):
        """The default pipeline the shared graph is, or was copied from."""
        return graph_pipeline_of(self.window.pipeline.nodes)

    def index_of(self, recipe_id):
        """The list index of a default pipeline; None is the one on the shared graph."""
        wanted = self.graph_pipeline.recipe_id if recipe_id is None else recipe_id
        return next(index for index in self._entries('template')
                    if self.selector.itemData(index).recipe_id == wanted)

    def select_template(self, recipe_id):
        """Load a fresh copy of a default pipeline, as choosing it in the list does."""
        self.selector.setCurrentIndex(self.index_of(recipe_id))

    def sync_buttons(self):
        catalogue = getattr(self.window, 'catalogue_nodes', None)
        documents = getattr(self.window, 'pipeline_documents', None)
        busy = bool(getattr(catalogue, 'busy', False))
        self.selector.setEnabled(not busy)
        self.window._sync_run_controls()
        self.load_button.setEnabled(not busy)
        self.save_button.setEnabled(not busy)
        self.export_button.setEnabled(not busy and bool(self.window._analyses))
        if catalogue is not None:
            # Always offered while locked: using it explains the lock.
            removable = catalogue.selected_instance is not None or (documents is not None and documents.locked)
            self.remove_added_action.setEnabled(removable and not busy)
            if catalogue.selected_instance is not None:
                self.export_button.setEnabled(not busy and catalogue.can_export_selected)
        self.selector.setToolTip(self.selector.currentText())
        self.export_menu_action.setEnabled(self.export_button.isEnabled())

    def adjust_for_width(self, available):
        """Keep file commands accessible when a narrow pane uses large text."""
        layout = self.strip.layout()
        widgets = [layout.itemAt(index).widget() for index in range(layout.count())
                   if layout.itemAt(index).widget() is not None]
        shown = set(widgets)
        self.selector.setFixedWidth(SELECTOR_WIDTH)
        margins = layout.contentsMargins()
        def required():
            return (margins.left() + margins.right() + max(0, len(shown) - 1) * layout.spacing()
                    + sum(min(widget.maximumWidth(), max(widget.minimumWidth(), widget.sizeHint().width()))
                          for widget in shown))
        # The compact icons stay: nothing else reaches their commands.
        for widget in (self.workflow_label, self.export_button):
            if required() <= available:
                break
            shown.discard(widget)
        excess = required() - available
        if excess > 0:
            self.selector.setFixedWidth(max(SELECTOR_MINIMUM_WIDTH, SELECTOR_WIDTH - excess))
        for widget in (self.workflow_label, self.export_button):
            if widget.isHidden() == (widget in shown):
                widget.setVisible(widget in shown)
        self.export_menu_action.setVisible(self.export_button.isHidden())

    def select_pipeline(self, index):
        kind = self.selector.itemData(index, ENTRY_KIND)
        if kind in ('heading', 'saved-heading'):
            # Only reachable programmatically; a heading stands for its first entry.
            self.selector.setCurrentIndex(index + 1)
            return
        documents = getattr(self.window, 'pipeline_documents', None)
        leaving_copy = documents is not None and not documents.locked
        pipeline = self.selector.itemData(index) if kind == 'template' else None
        # Back to the pipeline already on the graph keeps its values and layout.
        replacing_graph = pipeline not in (None, self.graph_pipeline)
        if leaving_copy or kind == 'custom' or replacing_graph:
            # A copy shares the default graph, which must not change under running work.
            if documents._busy():
                self.window.statusBar().showMessage('Wait for current work to finish before changing pipeline.')
                self.sync_copies()
                return
            if not documents.stash_current():
                self.sync_copies()
                return
        if kind == 'custom':
            if not documents.activate(self.selector.itemData(index)):
                self.sync_copies()
            return
        if replacing_graph:
            # It would describe a result the change clears.
            self.window.statusBar().clearMessage()
        if pipeline is CELLPOSE_SEED_ANALYSIS and not self.window._cellpose_available():
            self.sync_copies()
            return
        from seedvision.ui.pipeline_workspace import leave_copy, load_graph_pipeline
        if leaving_copy:
            leave_copy(self.window, pipeline if replacing_graph else None)
        elif replacing_graph:
            load_graph_pipeline(self.window, pipeline)
        if replacing_graph:
            self.window._selected_pipeline_node = pipeline.selected_node
            self.window._rebuild_overlay_combo(str(self.window.overlay_combo.currentData() or 'raw_image'))
            self.window._sync_inspector_overlay_options()
        self.install()
        if replacing_graph and self.window.image_view.image_path is not None:
            self.window._show_pending_result()
        if replacing_graph:
            # A default pipeline is laid out fresh; saved layouts load as they were.
            self.window.pipeline_canvas.auto_arrange()

    def install(self):
        """Show the shared default graph on the canvas and in the Pipeline list."""
        window = self.window
        # Cancel only running work: an idle Stop would hold back queued analyses.
        if window._stoppable_work_is_active():
            window._stop_current_work()
        with QSignalBlocker(self.selector):
            self.selector.setCurrentIndex(self.index_of(None))
        self.facts.hide()
        window.pipeline_inspector.set_analysis_result(None)
        window.pipeline_inspector.enable_unified_optimization()
        canvas = window.pipeline_canvas
        canvas.graph = window.pipeline
        canvas.rebuild_from_graph(window._selected_pipeline_node)
        window.image_display_stack.setCurrentWidget(window.image_view)
        window.baseline_section.show()
        window._set_project_dirty()
        if getattr(window, 'catalogue_nodes', None) is not None:
            window.catalogue_nodes.install_primary(window.pipeline)
        self.sync_buttons()
        window._refresh_analysis_activity_panel()
        window._refresh_project_node()
        mode = str(window.overlay_combo.currentData() or 'raw_image')
        window._rebuild_overlay_menu(mode)
        window._sync_overlay_display_controls(mode)
        window.image_view.set_overlay_mode(mode)
        window._sync_background_controls()
        window._update_analysis_availability()
        documents = getattr(window, 'pipeline_documents', None)
        if documents is not None:
            documents.loaded(self.graph_pipeline.title)

    def _add_saved_entry(self, text, data, kind):
        """List a project pipeline under its own heading, after the defaults."""
        if not self._entries('saved-heading'):
            self._add_heading(SAVED_HEADING, None, 'saved-heading')
        position = max(self._entries('saved-heading', 'custom')) + 1
        self.selector.insertItem(position, text, data)
        self.selector.setItemData(position, kind, ENTRY_KIND)
        return position

    def sync_copies(self):
        """Match the list to the project's named copies and show the current pipeline."""
        documents = getattr(self.window, 'pipeline_documents', None)
        if documents is None:
            return
        with QSignalBlocker(self.selector):
            for item in reversed(self._entries('custom', 'saved-heading')):
                self.selector.removeItem(item)
            for name in documents.custom:
                self._add_saved_entry(name, name, 'custom')
            index = next((item for item in self._entries('custom')
                          if self.selector.itemData(item) == documents.active_copy), None)
            if index is None:
                index = self.index_of(None)
            self.selector.setCurrentIndex(index)
        self.selector.setToolTip(self.selector.currentText())

    def render(self, raster):
        """Show an added node's raster output in place of the image view."""
        from PySide6.QtGui import QPixmap
        image = raster_preview_image(raster)
        self.preview.setPixmap(QPixmap.fromImage(image))
        self.preview.message.setText(f'Node output · {raster.shape[1]}×{raster.shape[0]} source pixels · '
            f'sampled to {image.width()}×{image.height()} display pixels · drag / scroll to pan / zoom')
        self.window.image_display_stack.setCurrentWidget(self.preview)

    def export(self):
        catalogue = getattr(self.window, 'catalogue_nodes', None)
        if catalogue is not None and catalogue.selected_instance is not None:
            catalogue.export_selected()
            return
        self.window._review_results()
