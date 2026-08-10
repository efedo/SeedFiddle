"""Native Qt graphics canvas for the left-to-right analysis pipeline."""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, QSignalBlocker, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QWheelEvent
from PySide6.QtWidgets import (
    QGraphicsObject,
    QGraphicsPathItem,
    QGraphicsProxyWidget,
    QGraphicsScene,
    QGraphicsView,
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QSpinBox,
    QStyleOptionGraphicsItem,
    QWidget,
)

from seedvision.pipeline import NodeStatus, PipelineConnection, PipelineGraph, PipelineNode


STATUS_COLOURS = {
    NodeStatus.IDLE: QColor("#66717e"),
    NodeStatus.RUNNING: QColor("#3d8ed0"),
    NodeStatus.COMPLETE: QColor("#3ba46f"),
    NodeStatus.WARNING: QColor("#c48a28"),
    NodeStatus.BLOCKED: QColor("#8a5a66"),
    NodeStatus.PLANNED: QColor("#736a9c"),
    NodeStatus.BYPASSED: QColor("#555d66"),
    NodeStatus.FAILED: QColor("#c95353"),
}


def _format_calculation_time(seconds: float | None) -> str:
    """Return a compact node-footer duration."""

    if seconds is None:
        return "Last calc: --"
    seconds = max(0.0, float(seconds))
    if seconds < 0.001:
        return f"Last calc: {seconds * 1000.0:.2f} ms"
    if seconds < 1.0:
        return f"Last calc: {seconds * 1000.0:.1f} ms"
    return f"Last calc: {seconds:.2f} s"


class PipelineNodeItem(QGraphicsObject):
    """Movable visual representation of one pipeline node."""

    DEFAULT_WIDTH = 200.0
    DEFAULT_HEIGHT = 108.0
    PORT_Y = 59.0
    NAMED_PORT_Y = 92.0
    ROW_HEIGHT = 24.0

    def __init__(
        self,
        node: PipelineNode,
        moved_callback,
        released_callback,
        parameter_callback,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.node = node
        self._moved_callback = moved_callback
        self._released_callback = released_callback
        self._parameter_callback = parameter_callback
        self.width = 220.0 if node.inline_parameters else self.DEFAULT_WIDTH
        named_port_count = max(len(node.input_ports), len(node.output_ports))
        port_bottom = (
            self.NAMED_PORT_Y + 10.0
            + max(0, named_port_count - 1) * self.ROW_HEIGHT
            if named_port_count
            else 80.0
        )
        self._controls_separator_y = max(84.0, port_bottom + 4.0)
        self._controls_y = self._controls_separator_y + 5.0
        if node.inline_parameters:
            controls_bottom = (
                self._controls_y
                + (len(node.inline_parameters) - 1) * self.ROW_HEIGHT
                + 22.0
            )
            self._footer_y = controls_bottom + 4.0
        else:
            self._footer_y = max(84.0, port_bottom + 4.0)
        self.height = max(self.DEFAULT_HEIGHT, self._footer_y + 24.0)
        self._inline_editors: dict[str, QWidget] = {}
        self.setPos(node.x, node.y)
        self.setFlags(
            self.GraphicsItemFlag.ItemIsMovable
            | self.GraphicsItemFlag.ItemIsSelectable
            | self.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setCacheMode(self.CacheMode.DeviceCoordinateCache)
        self.setToolTip(node.description)
        self.setZValue(2)
        self._build_inline_editors()

    def boundingRect(self) -> QRectF:  # noqa: N802 - Qt override
        return QRectF(-8, -8, self.width + 16, self.height + 16)

    def input_anchor(self, port_id: str = "") -> QPointF:
        return self.mapToScene(QPointF(0, self._port_y(self.node.input_ports, port_id)))

    def output_anchor(self, port_id: str = "") -> QPointF:
        return self.mapToScene(
            QPointF(self.width, self._port_y(self.node.output_ports, port_id))
        )

    @staticmethod
    def _port_y(ports: tuple[tuple[str, str], ...], port_id: str) -> float:
        if not ports:
            return PipelineNodeItem.PORT_Y
        index = next(
            (index for index, (identifier, _) in enumerate(ports) if identifier == port_id),
            0,
        )
        return PipelineNodeItem.NAMED_PORT_Y + index * PipelineNodeItem.ROW_HEIGHT

    def _build_inline_editors(self) -> None:
        for index, (key, _) in enumerate(self.node.inline_parameters):
            spec = next(item for item in self.node.parameter_specs if item.key == key)
            if spec.kind == "int":
                editor = QSpinBox()
                editor.setRange(int(spec.minimum), int(spec.maximum))
                editor.setSingleStep(int(spec.step or 1))
                editor.setValue(int(self.node.parameters[key]))
            elif spec.kind == "float":
                editor = QDoubleSpinBox()
                editor.setDecimals(2)
                editor.setRange(float(spec.minimum), float(spec.maximum))
                editor.setSingleStep(float(spec.step or 0.01))
                editor.setValue(float(self.node.parameters[key]))
            elif spec.kind == "bool":
                editor = QCheckBox()
                editor.setChecked(bool(self.node.parameters[key]))
            elif spec.kind == "choice":
                editor = QComboBox()
                editor.addItems(spec.choices)
                editor.setCurrentText(str(self.node.parameters[key]))
            else:
                raise ValueError(f"Unsupported inline parameter kind {spec.kind!r}.")
            if isinstance(editor, QAbstractSpinBox):
                editor.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
                editor.setFixedSize(92, 22)
                editor.setStyleSheet(
                    "QAbstractSpinBox { background: #151a20; color: #edf3f7; "
                    "border: 1px solid #566573; border-radius: 3px; padding: 1px 4px; }"
                    "QAbstractSpinBox:focus { border-color: #57d8ff; }"
                )
                editor.editingFinished.connect(
                    lambda key=key, editor=editor: self._parameter_callback(
                        self.node.identifier, key, editor.value()
                    )
                )
            elif isinstance(editor, QComboBox):
                editor.setFixedSize(108, 22)
                editor.setStyleSheet(
                    "QComboBox { background: #151a20; color: #edf3f7; "
                    "border: 1px solid #566573; border-radius: 3px; padding: 1px 4px; }"
                    "QComboBox:focus { border-color: #57d8ff; }"
                )
                editor.currentTextChanged.connect(
                    lambda value, key=key: self._parameter_callback(
                        self.node.identifier, key, value
                    )
                )
            else:
                editor.setFixedSize(92, 22)
                editor.setStyleSheet(
                    "QCheckBox { background: transparent; color: #edf3f7; }"
                    "QCheckBox::indicator { width: 16px; height: 16px; }"
                )
                editor.toggled.connect(
                    lambda value, key=key: self._parameter_callback(
                        self.node.identifier, key, value
                    )
                )
            editor.setToolTip(spec.description)
            editor.setEnabled(self.node.enabled and self.node.implemented)
            proxy = QGraphicsProxyWidget(self)
            proxy.setWidget(editor)
            proxy.setPos(
                self.width - editor.width() - 12.0,
                self._controls_y + index * self.ROW_HEIGHT,
            )
            proxy.setZValue(4)
            self._inline_editors[key] = editor

    def refresh(self) -> None:
        for key, editor in self._inline_editors.items():
            with QSignalBlocker(editor):
                if isinstance(editor, QAbstractSpinBox):
                    editor.setValue(self.node.parameters[key])
                elif isinstance(editor, QComboBox):
                    editor.setCurrentText(str(self.node.parameters[key]))
                else:
                    editor.setChecked(bool(self.node.parameters[key]))
            editor.setEnabled(self.node.enabled and self.node.implemented)
        self.update()

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        del option, widget
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        body = QRectF(0, 0, self.width, self.height)
        selected = self.isSelected()
        border = QColor("#8fdcff") if selected else QColor("#4d5865")
        border_pen = QPen(border, 2.5 if selected else 1.3)
        border_pen.setCosmetic(True)
        painter.setPen(border_pen)
        painter.setBrush(QColor("#272e36") if self.node.enabled else QColor("#24282d"))
        painter.drawRoundedRect(body, 9, 9)

        header = QRectF(1, 1, self.width - 2, 34)
        status_colour = STATUS_COLOURS[self.node.status]
        if not self.node.enabled:
            status_colour = STATUS_COLOURS[NodeStatus.BYPASSED]
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(status_colour)
        painter.drawRoundedRect(header, 8, 8)
        painter.drawRect(QRectF(1, 24, self.width - 2, 11))

        painter.setPen(QColor("#f6f8fa"))
        title_font = QFont(painter.font())
        title_font.setBold(True)
        title_font.setPointSizeF(10.0)
        painter.setFont(title_font)
        painter.drawText(QRectF(12, 5, self.width - 24, 26), self.node.title)

        body_font = QFont(painter.font())
        body_font.setBold(False)
        body_font.setPointSizeF(8.5)
        painter.setFont(body_font)
        painter.setPen(QColor("#aeb8c3"))
        painter.drawText(QRectF(12, 43, self.width - 24, 18), self.node.category.upper())
        painter.setPen(QColor("#e2e7ec"))
        detail = self.node.status_detail or self.node.status.value.capitalize()
        if self.node.inline_parameters:
            painter.drawText(QRectF(12, 62, self.width - 24, 18), detail)
            painter.setPen(QPen(QColor("#46515d"), 1.0))
            painter.drawLine(
                QPointF(10, self._controls_separator_y),
                QPointF(self.width - 10, self._controls_separator_y),
            )
            painter.setPen(QColor("#cbd4dc"))
            for index, (_, label) in enumerate(self.node.inline_parameters):
                painter.drawText(
                    QRectF(
                        14,
                        self._controls_y + 1.0 + index * self.ROW_HEIGHT,
                        self.width - 126,
                        22,
                    ),
                    Qt.AlignmentFlag.AlignVCenter,
                    label,
                )
        else:
            painter.drawText(
                QRectF(12, 62, self.width - 24, 18),
                detail,
            )

        footer_y = self._footer_y
        painter.setPen(QPen(QColor("#46515d"), 1.0))
        painter.drawLine(QPointF(10, footer_y), QPointF(self.width - 10, footer_y))
        footer_font = QFont(body_font)
        footer_font.setPointSizeF(8.0)
        painter.setFont(footer_font)
        painter.setPen(QColor("#8fa0b2"))
        timing_text = (
            "Calculation: running..."
            if self.node.status == NodeStatus.RUNNING
            else _format_calculation_time(self.node.calculation_seconds)
        )
        painter.drawText(
            QRectF(12, footer_y + 3.0, self.width - 24, 20),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            timing_text,
        )

        port_pen = QPen(QColor("#172029"), 1.5)
        port_pen.setCosmetic(True)
        painter.setPen(port_pen)
        painter.setBrush(QColor("#9ba8b5"))
        if self.node.input_ports:
            for port_id, label in self.node.input_ports:
                y = self._port_y(self.node.input_ports, port_id)
                painter.drawEllipse(QPointF(0, y), 6, 6)
                painter.setPen(QColor("#9ba8b5"))
                painter.drawText(QRectF(10, y - 10, 112, 20), label)
                painter.setPen(port_pen)
        else:
            painter.drawEllipse(QPointF(0, self.PORT_Y), 6, 6)
        if self.node.output_ports:
            for port_id, label in self.node.output_ports:
                y = self._port_y(self.node.output_ports, port_id)
                painter.drawEllipse(QPointF(self.width, y), 6, 6)
                painter.setPen(QColor("#d6dee5"))
                painter.drawText(
                    QRectF(self.width - 150, y - 10, 138, 20),
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                    label,
                )
                painter.setPen(port_pen)
        else:
            painter.drawEllipse(QPointF(self.width, self.PORT_Y), 6, 6)

    def itemChange(self, change, value):  # noqa: N802 - Qt override
        result = super().itemChange(change, value)
        if change == self.GraphicsItemChange.ItemPositionHasChanged:
            self.node.x = float(self.pos().x())
            self.node.y = float(self.pos().y())
            self._moved_callback(self.node.identifier)
        return result

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().mouseReleaseEvent(event)
        self._released_callback(self.node.identifier)


class PipelineEdgeItem(QGraphicsPathItem):
    """Bezier connection between two node ports."""

    def __init__(
        self,
        connection: PipelineConnection,
        source: PipelineNodeItem,
        target: PipelineNodeItem,
    ) -> None:
        super().__init__()
        self.connection = connection
        self.source = source
        self.target = target
        self.set_highlighted(False)
        self.setZValue(0)
        self.setToolTip(connection.data_type)
        self.update_path()

    def set_highlighted(self, highlighted: bool) -> None:
        """Emphasize every connection incident to the selected node."""

        pen = QPen(
            QColor("#57d8ff") if highlighted else QColor("#778594"),
            4.8 if highlighted else 2.2,
        )
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setZValue(1 if highlighted else 0)

    def update_path(self) -> None:
        start = self.source.output_anchor(self.connection.source_port)
        end = self.target.input_anchor(self.connection.target_port)
        horizontal = max(60.0, abs(end.x() - start.x()) * 0.48)
        path = QPainterPath(start)
        path.cubicTo(
            QPointF(start.x() + horizontal, start.y()),
            QPointF(end.x() - horizontal, end.y()),
            end,
        )
        self.setPath(path)


class PipelineCanvas(QGraphicsView):
    """Zoomable, pannable, and movable Qt pipeline-node canvas."""

    node_selected = Signal(str)
    parameter_changed = Signal(str, str, object)
    NODE_GAP = 18.0

    def __init__(self, graph: PipelineGraph, parent=None) -> None:
        super().__init__(parent)
        self.graph = graph
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.node_items: dict[str, PipelineNodeItem] = {}
        self.edge_items: list[PipelineEdgeItem] = []
        self._zoom_steps = 0
        self._fit_pending = True
        self._arranging_nodes = False

        self.setBackgroundBrush(QColor("#1c2229"))
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.TextAntialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self._populate()
        self._scene.selectionChanged.connect(self._selection_changed)

    def _populate(self) -> None:
        for node in self.graph.nodes.values():
            item = PipelineNodeItem(
                node,
                self._node_moved,
                self._node_released,
                self._inline_parameter_changed,
            )
            self.node_items[node.identifier] = item
            self._scene.addItem(item)
        self._resolve_all_overlaps()
        for connection in self.graph.connections:
            edge = PipelineEdgeItem(
                connection,
                self.node_items[connection.source],
                self.node_items[connection.target],
            )
            self.edge_items.append(edge)
            self._scene.addItem(edge)
        self._update_scene_rect()

    def _node_moved(self, node_id: str) -> None:
        del node_id
        if self._arranging_nodes:
            return
        for edge in self.edge_items:
            edge.update_path()
        self._update_scene_rect()

    def _node_released(self, node_id: str) -> None:
        """Move a dropped node to the nearest non-overlapping vertical slot."""

        item = self.node_items[node_id]
        new_y = self._nearest_available_y(
            item,
            float(item.pos().y()),
            tuple(other for other in self.node_items.values() if other is not item),
        )
        if not math.isclose(new_y, float(item.pos().y()), abs_tol=0.01):
            self._arranging_nodes = True
            try:
                item.setPos(item.pos().x(), new_y)
            finally:
                self._arranging_nodes = False
        self._node_moved(node_id)

    def _resolve_all_overlaps(self) -> None:
        """Compact the authored graph positions without allowing node collisions."""

        placed: list[PipelineNodeItem] = []
        ordered = sorted(
            self.node_items.values(),
            key=lambda item: (float(item.pos().x()), float(item.pos().y())),
        )
        self._arranging_nodes = True
        try:
            for item in ordered:
                new_y = self._nearest_available_y(
                    item, float(item.pos().y()), tuple(placed)
                )
                item.setPos(item.pos().x(), new_y)
                placed.append(item)
        finally:
            self._arranging_nodes = False

    def _nearest_available_y(
        self,
        item: PipelineNodeItem,
        desired_y: float,
        obstacles: tuple[PipelineNodeItem, ...],
    ) -> float:
        relevant = tuple(
            other
            for other in obstacles
            if self._horizontal_ranges_overlap(item, other)
        )
        if not relevant:
            return desired_y
        candidates = {desired_y}
        for other in relevant:
            candidates.add(float(other.pos().y()) + other.height + self.NODE_GAP)
            candidates.add(
                float(other.pos().y()) - item.height - self.NODE_GAP
            )
        valid = [
            candidate
            for candidate in candidates
            if all(
                not self._items_overlap(item, candidate, other)
                for other in relevant
            )
        ]
        if not valid:
            return max(
                float(other.pos().y()) + other.height + self.NODE_GAP
                for other in relevant
            )
        return min(
            valid,
            key=lambda candidate: (
                abs(candidate - desired_y),
                0 if candidate >= desired_y else 1,
                candidate,
            ),
        )

    def _horizontal_ranges_overlap(
        self, first: PipelineNodeItem, second: PipelineNodeItem
    ) -> bool:
        first_left = float(first.pos().x())
        second_left = float(second.pos().x())
        return not (
            first_left + first.width + self.NODE_GAP <= second_left
            or second_left + second.width + self.NODE_GAP <= first_left
        )

    def _items_overlap(
        self,
        item: PipelineNodeItem,
        candidate_y: float,
        other: PipelineNodeItem,
    ) -> bool:
        if not self._horizontal_ranges_overlap(item, other):
            return False
        other_y = float(other.pos().y())
        return not (
            candidate_y + item.height + self.NODE_GAP <= other_y
            or other_y + other.height + self.NODE_GAP <= candidate_y
        )

    def _inline_parameter_changed(self, node_id: str, key: str, value) -> None:
        self.parameter_changed.emit(node_id, key, value)

    def _update_scene_rect(self) -> None:
        bounds = self._scene.itemsBoundingRect().adjusted(-90, -90, 90, 90)
        self._scene.setSceneRect(bounds)

    def _selection_changed(self) -> None:
        selected = [
            item for item in self._scene.selectedItems() if isinstance(item, PipelineNodeItem)
        ]
        selected_id = selected[0].node.identifier if selected else None
        for edge in self.edge_items:
            edge.set_highlighted(
                selected_id is not None
                and selected_id
                in (edge.connection.source, edge.connection.target)
            )
        if selected_id is not None:
            self.node_selected.emit(selected_id)

    def select_node(self, node_id: str) -> None:
        item = self.node_items[node_id]
        self._scene.clearSelection()
        item.setSelected(True)
        self.centerOn(item)

    def refresh(self, node_ids=None) -> None:
        identifiers = self.node_items if node_ids is None else node_ids
        for node_id in identifiers:
            self.node_items[node_id].refresh()
        for edge in self.edge_items:
            edge.update_path()

    def fit_graph(self) -> None:
        bounds = self._scene.itemsBoundingRect().adjusted(-45, -45, 45, 45)
        if bounds.isEmpty():
            return
        self.resetTransform()
        self.fitInView(bounds, Qt.AspectRatioMode.KeepAspectRatio)
        # With the graph below the image, fitting all 3,000+ scene units made
        # node text microscopic. Preserve a readable minimum scale and let the
        # standard QGraphicsView scroll bars expose the remaining graph width.
        if self.transform().m11() < 0.55:
            self.resetTransform()
            self.scale(0.55, 0.55)
            self.centerOn(bounds.left(), bounds.center().y())
        self._zoom_steps = 0

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().showEvent(event)
        if self._fit_pending:
            QTimer.singleShot(0, self._finish_pending_fit)

    def _finish_pending_fit(self) -> None:
        self._fit_pending = False
        self.fit_graph()

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 - Qt override
        direction = 1 if event.angleDelta().y() > 0 else -1
        if direction < 0 and self._zoom_steps <= -8:
            return
        if direction > 0 and self._zoom_steps >= 20:
            return
        factor = 1.18 if direction > 0 else 1 / 1.18
        self.scale(factor, factor)
        self._zoom_steps += direction

    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:  # noqa: N802
        painter.fillRect(rect, QColor("#1c2229"))
        minor = 24
        major = minor * 4
        left = math.floor(rect.left() / minor) * minor
        top = math.floor(rect.top() / minor) * minor

        minor_pen = QPen(QColor("#252d35"), 1)
        minor_pen.setCosmetic(True)
        major_pen = QPen(QColor("#303a44"), 1)
        major_pen.setCosmetic(True)
        x = left
        while x <= rect.right():
            painter.setPen(major_pen if int(x) % major == 0 else minor_pen)
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            x += minor
        y = top
        while y <= rect.bottom():
            painter.setPen(major_pen if int(y) % major == 0 else minor_pen)
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            y += minor
