"""Native Qt graphics canvas for the left-to-right analysis pipeline."""

from __future__ import annotations

import hashlib
import math

from PySide6.QtCore import QPointF, QRectF, QSignalBlocker, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetricsF,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QWheelEvent,
)
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
    QFrame,
    QHBoxLayout,
    QLayout,
    QLabel,
    QMenu,
    QListWidget,
    QListWidgetItem,
    QLineEdit,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
    QSpinBox,
    QStyleOptionGraphicsItem,
    QToolButton,
)

from seedvision.pipeline import NodeStatus, PipelineConnection, PipelineGraph, PipelineNode, category_rank
from seedvision.ui.canvas_controls import style_canvas_control_bar
from seedvision.ui.spin_format import spin_decimals


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


_ROUTE_CLEARANCE = 2.0
_ROUTE_CORNER_RADIUS = 6.0
_ROUTE_LEAD = _ROUTE_CLEARANCE + _ROUTE_CORNER_RADIUS
_ROUTE_BEND_PENALTY = 28.0


def _same_point(first: QPointF, second: QPointF, tolerance: float = 0.01) -> bool:
    return (
        abs(float(first.x()) - float(second.x())) <= tolerance
        and abs(float(first.y()) - float(second.y())) <= tolerance
    )


def _compressed_route_points(
    points: list[QPointF] | tuple[QPointF, ...],
) -> tuple[QPointF, ...]:
    """Drop duplicate and collinear waypoints without changing the route."""

    compact: list[QPointF] = []
    for point in points:
        point = QPointF(point)
        if compact and _same_point(compact[-1], point):
            continue
        compact.append(point)
        while len(compact) >= 3:
            first, middle, last = compact[-3:]
            vertical = (
                abs(float(first.x()) - float(middle.x())) <= 0.01
                and abs(float(middle.x()) - float(last.x())) <= 0.01
            )
            horizontal = (
                abs(float(first.y()) - float(middle.y())) <= 0.01
                and abs(float(middle.y()) - float(last.y())) <= 0.01
            )
            if not (vertical or horizontal):
                break
            compact.pop(-2)
    return tuple(compact)


def _rounded_route_path(
    points: tuple[QPointF, ...], radius: float = _ROUTE_CORNER_RADIUS
) -> QPainterPath:
    """Build a polyline path with bounded quadratic rounding at each bend."""

    if not points:
        return QPainterPath()
    path = QPainterPath(points[0])
    if len(points) == 1:
        return path
    for index in range(1, len(points) - 1):
        previous = points[index - 1]
        corner = points[index]
        following = points[index + 1]
        before_dx = float(previous.x()) - float(corner.x())
        before_dy = float(previous.y()) - float(corner.y())
        after_dx = float(following.x()) - float(corner.x())
        after_dy = float(following.y()) - float(corner.y())
        before_length = math.hypot(before_dx, before_dy)
        after_length = math.hypot(after_dx, after_dy)
        if before_length <= 0.01 or after_length <= 0.01:
            path.lineTo(corner)
            continue
        corner_radius = min(float(radius), before_length * 0.42, after_length * 0.42)
        before = QPointF(
            float(corner.x()) + before_dx * corner_radius / before_length,
            float(corner.y()) + before_dy * corner_radius / before_length,
        )
        after = QPointF(
            float(corner.x()) + after_dx * corner_radius / after_length,
            float(corner.y()) + after_dy * corner_radius / after_length,
        )
        path.lineTo(before)
        path.quadTo(corner, after)
    path.lineTo(points[-1])
    return path


def _default_connection_path(start: QPointF, end: QPointF) -> QPainterPath:
    """Return the established free Bezier used when routing options are off."""

    horizontal = max(60.0, abs(float(end.x()) - float(start.x())) * 0.48)
    path = QPainterPath(start)
    path.cubicTo(
        QPointF(float(start.x()) + horizontal, float(start.y())),
        QPointF(float(end.x()) - horizontal, float(end.y())),
        end,
    )
    return path


def _smooth_bundle_path(points: tuple[QPointF, ...]) -> QPainterPath:
    """Interpolate a bundle route with continuous cubic tangents."""

    points = _compressed_route_points(points)
    if len(points) < 2:
        return QPainterPath(points[0]) if points else QPainterPath()
    path = QPainterPath(points[0])
    for index in range(len(points) - 1):
        previous = points[max(0, index - 1)]
        start = points[index]
        end = points[index + 1]
        following = points[min(len(points) - 1, index + 2)]
        control_1 = QPointF(
            float(start.x()) + (float(end.x()) - float(previous.x())) / 6.0,
            float(start.y()) + (float(end.y()) - float(previous.y())) / 6.0,
        )
        control_2 = QPointF(
            float(end.x()) - (float(following.x()) - float(start.x())) / 6.0,
            float(end.y()) - (float(following.y()) - float(start.y())) / 6.0,
        )
        path.cubicTo(control_1, control_2, end)
    return path


def _stable_colour_fraction(*parts: str) -> float:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") / (
        2**64 - 1
    )


def _wire_colour(connection: PipelineConnection) -> QColor:
    """Give each wire a stable variation within its source node's palette."""

    base_hue = _stable_colour_fraction(connection.source, "source-hue")
    variation = (
        _stable_colour_fraction(
            connection.source_port,
            connection.target,
            connection.target_port,
            connection.data_type,
        )
        - 0.5
    )
    hue = (base_hue + variation * 0.10) % 1.0
    saturation = 0.58 + 0.20 * _stable_colour_fraction(
        connection.source_port, connection.target_port, "saturation"
    )
    value = 0.76 + 0.16 * _stable_colour_fraction(
        connection.target, connection.data_type, "value"
    )
    return QColor.fromHsvF(hue, saturation, value)


def _source_cable_colour(source: str) -> QColor:
    hue = _stable_colour_fraction(source, "source-hue")
    return QColor.fromHsvF(hue, 0.48, 0.48)


def _route_clears_obstacles(
    points: tuple[QPointF, ...], obstacles: tuple[QRectF, ...]
) -> bool:
    """Return whether an orthogonal waypoint route clears every rect interior."""

    for first, second in zip(points, points[1:]):
        first_x, first_y = float(first.x()), float(first.y())
        second_x, second_y = float(second.x()), float(second.y())
        horizontal = abs(first_y - second_y) <= 0.01
        vertical = abs(first_x - second_x) <= 0.01
        if not (horizontal or vertical):
            return False
        for obstacle in obstacles:
            if horizontal:
                if (
                    float(obstacle.top()) + 0.01
                    < first_y
                    < float(obstacle.bottom()) - 0.01
                    and min(first_x, second_x) < float(obstacle.right()) - 0.01
                    and max(first_x, second_x) > float(obstacle.left()) + 0.01
                ):
                    return False
            elif (
                float(obstacle.left()) + 0.01
                < first_x
                < float(obstacle.right()) - 0.01
                and min(first_y, second_y) < float(obstacle.bottom()) - 0.01
                and max(first_y, second_y) > float(obstacle.top()) + 0.01
            ):
                return False
    return True


def _orthogonal_route(
    start: QPointF,
    end: QPointF,
    obstacles: tuple[QRectF, ...],
) -> tuple[QPointF, ...]:
    """Find a deterministic bend-aware rectilinear route around rectangles."""

    start = QPointF(start)
    end = QPointF(end)
    if _same_point(start, end):
        return (start,)
    all_rectangles = tuple(
        (
            float(obstacle.left()),
            float(obstacle.top()),
            float(obstacle.right()),
            float(obstacle.bottom()),
        )
        for obstacle in obstacles
    )
    if any(
        left + 0.01 < float(point.x()) < right - 0.01
        and top + 0.01 < float(point.y()) < bottom - 0.01
        for point in (start, end)
        for left, top, right, bottom in all_rectangles
    ):
        return _compressed_route_points((start, end))

    left = min(float(start.x()), float(end.x()))
    right = max(float(start.x()), float(end.x()))
    rectangles = tuple(
        rectangle
        for rectangle in all_rectangles
        if rectangle[2] >= left and rectangle[0] <= right
    )

    def horizontal_is_clear(y: float, first_x: float, second_x: float) -> bool:
        segment_left = min(first_x, second_x)
        segment_right = max(first_x, second_x)
        return not any(
            top + 0.01 < y < bottom - 0.01
            and segment_left < right_edge - 0.01
            and segment_right > left_edge + 0.01
            for left_edge, top, right_edge, bottom in rectangles
        )

    def vertical_is_clear(x: float, first_y: float, second_y: float) -> bool:
        segment_top = min(first_y, second_y)
        segment_bottom = max(first_y, second_y)
        return not any(
            left_edge + 0.01 < x < right_edge - 0.01
            and segment_top < bottom - 0.01
            and segment_bottom > top + 0.01
            for left_edge, top, right_edge, bottom in rectangles
        )

    def segment_is_clear(first: QPointF, second: QPointF) -> bool:
        if _same_point(first, second):
            return True
        if abs(float(first.y()) - float(second.y())) <= 0.01:
            return horizontal_is_clear(
                float(first.y()), float(first.x()), float(second.x())
            )
        if abs(float(first.x()) - float(second.x())) <= 0.01:
            return vertical_is_clear(
                float(first.x()), float(first.y()), float(second.y())
            )
        return False

    def route_score(points: tuple[QPointF, ...]) -> float:
        distance = sum(
            abs(float(second.x()) - float(first.x()))
            + abs(float(second.y()) - float(first.y()))
            for first, second in zip(points, points[1:])
        )
        return distance + max(0, len(points) - 2) * _ROUTE_BEND_PENALTY

    candidates: list[tuple[QPointF, ...]] = []

    def consider(points: tuple[QPointF, ...]) -> None:
        points = _compressed_route_points(points)
        if all(
            segment_is_clear(first, second)
            for first, second in zip(points, points[1:])
        ):
            candidates.append(points)

    consider(
        (
            start,
            QPointF(float(end.x()), float(start.y())),
            end,
        )
    )
    consider(
        (
            start,
            QPointF(float(start.x()), float(end.y())),
            end,
        )
    )
    lanes = sorted(
        {
            float(start.y()),
            float(end.y()),
            *(top for _, top, _, _ in rectangles),
            *(bottom for _, _, _, bottom in rectangles),
        },
        key=lambda y: (
            abs(y - (float(start.y()) + float(end.y())) * 0.5),
            y,
        ),
    )
    for lane in lanes:
        consider(
            (
                start,
                QPointF(float(start.x()), lane),
                QPointF(float(end.x()), lane),
                end,
            )
        )
    if candidates:
        return min(candidates, key=route_score)

    # A manually overlapped card can block a vertical endpoint leg. Try one
    # extra channel before falling back to the direct display path.
    rectangles = all_rectangles
    lanes = sorted(
        {
            float(start.y()),
            float(end.y()),
            *(top for _, top, _, _ in rectangles),
            *(bottom for _, _, _, bottom in rectangles),
        },
        key=lambda y: (
            abs(y - (float(start.y()) + float(end.y())) * 0.5),
            y,
        ),
    )
    channels = sorted(
        {
            float(start.x()),
            float(end.x()),
            *(left_edge for left_edge, _, _, _ in rectangles),
            *(right_edge for _, _, right_edge, _ in rectangles),
        },
        key=lambda x: (abs(x - (left + right) * 0.5), x),
    )
    start_x, start_y = float(start.x()), float(start.y())
    end_x, end_y = float(end.x()), float(end.y())
    source_order = sorted(channels, key=lambda x: (abs(x - start_x), x))
    target_order = sorted(channels, key=lambda x: (abs(x - end_x), x))
    best_coordinates: tuple[float, float, float] | None = None
    best_score = math.inf
    for lane in lanes:
        source_channels: list[float] = []
        for channel in source_order:
            if horizontal_is_clear(
                start_y, start_x, channel
            ) and vertical_is_clear(channel, start_y, lane):
                source_channels.append(channel)
                if len(source_channels) == 3:
                    break
        target_channels: list[float] = []
        for channel in target_order:
            if vertical_is_clear(
                channel, lane, end_y
            ) and horizontal_is_clear(end_y, channel, end_x):
                target_channels.append(channel)
                if len(target_channels) == 3:
                    break
        for source_channel in source_channels:
            for target_channel in target_channels:
                if not horizontal_is_clear(lane, source_channel, target_channel):
                    continue
                score = (
                    abs(source_channel - start_x)
                    + abs(lane - start_y)
                    + abs(target_channel - source_channel)
                    + abs(end_y - lane)
                    + abs(end_x - target_channel)
                    + 4.0 * _ROUTE_BEND_PENALTY
                )
                if score < best_score:
                    best_coordinates = (source_channel, lane, target_channel)
                    best_score = score
    if best_coordinates is not None:
        source_channel, lane, target_channel = best_coordinates
        return _compressed_route_points(
            (
                start,
                QPointF(source_channel, start_y),
                QPointF(source_channel, lane),
                QPointF(target_channel, lane),
                QPointF(target_channel, end_y),
                end,
            )
        )
    return _compressed_route_points((start, end))


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
        graph: PipelineGraph,
        moved_callback,
        released_callback,
        parameter_callback,
        run_to_callback,
        parent=None,
        *,
        read_only: bool = False,
        reference_view: bool = False,
        editing_locked: bool = False,
    ) -> None:
        super().__init__(parent)
        self.node = node
        self.graph = graph
        self._moved_callback = moved_callback
        self._released_callback = released_callback
        self._parameter_callback = parameter_callback
        self._run_to_callback = run_to_callback
        self._read_only = bool(read_only)
        self._editing_locked = bool(editing_locked)
        self._reference_view = bool(reference_view)
        self.width = (
            250.0
            if self._reference_view
            else 220.0 if node.inline_parameters else self.DEFAULT_WIDTH
        )
        self._header_height = 34.0
        self._header_body_offset = 0.0
        if self._reference_view:
            title_font = QFont()
            title_font.setBold(True)
            title_font.setPointSizeF(10.0)
            title_bounds = QFontMetricsF(title_font).boundingRect(
                QRectF(0, 0, self.width - 24.0, 1000.0),
                int(Qt.TextFlag.TextWordWrap),
                self.node.title,
            )
            self._header_height = max(34.0, math.ceil(title_bounds.height() + 12.0))
            self._header_body_offset = self._header_height - 34.0
        # Viewer-only overlay outputs fold behind one expander row; every
        # dataflow output and every connected port always stays visible.
        self._show_all_outputs = False
        self._inline_editors: dict[str, QWidget] = {}
        self._inline_proxies: dict[str, QGraphicsProxyWidget] = {}
        self._layout_geometry()
        self._adjacent = False
        self.setPos(node.x, node.y)
        self.setFlags(
            self.GraphicsItemFlag.ItemIsMovable
            | self.GraphicsItemFlag.ItemIsSelectable
            | self.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setCacheMode(self.CacheMode.DeviceCoordinateCache)
        self._update_tooltip()
        self.setZValue(2)
        self._build_inline_editors()

    def _output_connected(self, port_id: str) -> bool:
        return any(
            connection.source == self.node.identifier
            and connection.source_port == port_id
            for connection in self.graph.connections
        )

    def visible_output_ports(self) -> tuple[tuple[str, str], ...]:
        if self._show_all_outputs:
            return self.node.output_ports
        return tuple(
            (port_id, label)
            for port_id, label in self.node.output_ports
            if not str(self.node.output_port_types.get(port_id, "")).startswith("ViewerOverlay:")
            or self._output_connected(port_id)
        )

    def hidden_output_count(self) -> int:
        return len(self.node.output_ports) - len(self.visible_output_ports())

    def _has_output_expander(self) -> bool:
        return self._show_all_outputs or self.hidden_output_count() > 0

    def _expander_y(self) -> float:
        return self._port_y_at(len(self.visible_output_ports()))

    def _layout_geometry(self) -> None:
        """Size the card for its visible ports and inline controls."""

        output_rows = len(self.visible_output_ports()) + int(self._has_output_expander())
        named_port_count = max(len(self.node.input_ports), output_rows)
        port_base = self.NAMED_PORT_Y + self._header_body_offset
        port_bottom = (
            port_base + 10.0
            + max(0, named_port_count - 1) * self.ROW_HEIGHT
            if named_port_count
            else 80.0 + self._header_body_offset
        )
        self._controls_separator_y = max(84.0, port_bottom + 4.0)
        self._controls_y = self._controls_separator_y + 5.0
        if self.node.inline_parameters:
            controls_bottom = (
                self._controls_y
                + (len(self.node.inline_parameters) - 1) * self.ROW_HEIGHT
                + 22.0
            )
            self._footer_y = controls_bottom + 4.0
        else:
            self._footer_y = max(84.0, port_bottom + 4.0)
        self.height = max(self.DEFAULT_HEIGHT, self._footer_y + 24.0)
        for index, (key, _) in enumerate(self.node.inline_parameters):
            proxy = self._inline_proxies.get(key)
            if proxy is not None:
                proxy.setPos(
                    self.width - proxy.widget().width() - 12.0,
                    self._controls_y + index * self.ROW_HEIGHT,
                )

    def _relayout(self) -> bool:
        """Re-size after port visibility changes; return whether geometry moved."""

        previous = (self.height, self._footer_y, self._controls_y)
        self.prepareGeometryChange()
        self._layout_geometry()
        return previous != (self.height, self._footer_y, self._controls_y)

    def set_outputs_expanded(self, expanded: bool) -> None:
        expanded = bool(expanded)
        if expanded == self._show_all_outputs:
            return
        self._show_all_outputs = expanded
        self._relayout()
        self.update()
        # Reuse the move path so incident cables follow the resized card.
        self._moved_callback(self.node.identifier)

    def boundingRect(self) -> QRectF:  # noqa: N802 - Qt override
        return QRectF(-8, -8, self.width + 16, self.height + 16)

    def input_anchor(self, port_id: str = "") -> QPointF:
        return self.mapToScene(QPointF(0, self._port_y(self.node.input_ports, port_id)))

    def output_anchor(self, port_id: str = "") -> QPointF:
        return self.mapToScene(
            QPointF(self.width, self._port_y(self.visible_output_ports(), port_id))
        )

    def port_at_scene(
        self, scene_position: QPointF, radius: float = 13.0
    ) -> tuple[str, str] | None:
        """Hit-test named connector circles in scene coordinates."""

        local = self.mapFromScene(scene_position)
        for direction, x, ports in (
            ("input", 0.0, self.node.input_ports),
            ("output", self.width, self.visible_output_ports()),
        ):
            for port_id, _ in ports:
                y = self._port_y(ports, port_id)
                if math.hypot(local.x() - x, local.y() - y) <= radius:
                    return direction, port_id
        return None

    def _update_tooltip(self) -> None:
        inputs = ", ".join(label for _, label in self.node.input_ports) or "None"
        outputs = ", ".join(label for _, label in self.node.output_ports) or "None"
        interaction = (
            "This reference graph is read-only. Node position is presentation only."
            if self._read_only
            else "Drag between sockets to reconnect. Drag a connected input into empty "
            "space, or right-click an edge, to disconnect it."
        )
        self.setToolTip(
            f"{self.node.title}\n\n{self.node.description}\n\n"
            f"Inputs: {inputs}\nOutputs: {outputs}\n\n"
            f"{interaction}"
        )

    def _port_y_at(self, index: int) -> float:
        return (
            PipelineNodeItem.NAMED_PORT_Y
            + self._header_body_offset
            + index * PipelineNodeItem.ROW_HEIGHT
        )

    def _port_y(self, ports: tuple[tuple[str, str], ...], port_id: str) -> float:
        if not ports:
            return PipelineNodeItem.PORT_Y + self._header_body_offset
        index = next(
            (index for index, (identifier, _) in enumerate(ports) if identifier == port_id),
            0,
        )
        return self._port_y_at(index)

    def _build_inline_editors(self) -> None:
        for index, (key, _) in enumerate(self.node.inline_parameters):
            spec = next(item for item in self.node.parameter_specs if item.key == key)
            if self._read_only:
                value = self.node.parameters[key]
                if isinstance(value, bool):
                    rendered = "Yes" if value else "No"
                else:
                    rendered = str(value)
                editor = QLabel(rendered)
                editor.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                editor.setFixedSize(108, 22)
                editor.setStyleSheet("QLabel { color: #edf3f7; padding: 1px 4px; }")
                editor.setToolTip(f"{spec.description}\nFixed reference value.")
                proxy = QGraphicsProxyWidget(self)
                proxy.setWidget(editor)
                proxy.setPos(
                    self.width - editor.width() - 12.0,
                    self._controls_y + index * self.ROW_HEIGHT,
                )
                proxy.setZValue(4)
                self._inline_editors[key] = editor
                self._inline_proxies[key] = proxy
                continue
            if spec.kind == "int":
                editor = QSpinBox()
                editor.setRange(int(spec.minimum), int(spec.maximum))
                editor.setSingleStep(int(spec.step or 1))
                editor.setValue(int(self.node.parameters[key]))
            elif spec.kind == "float":
                editor = QDoubleSpinBox()
                editor.setDecimals(spin_decimals(
                    spec,
                    self.node.parameters[key],
                    self.node.default_parameters.get(key, self.node.parameters[key]),
                ))
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
            editor.setEnabled(self.node.enabled and self.node.implemented and not self._editing_locked)
            proxy = QGraphicsProxyWidget(self)
            proxy.setWidget(editor)
            proxy.setPos(
                self.width - editor.width() - 12.0,
                self._controls_y + index * self.ROW_HEIGHT,
            )
            proxy.setZValue(4)
            self._inline_editors[key] = editor
            self._inline_proxies[key] = proxy

    def refresh(self) -> None:
        # Connecting or disconnecting can reveal or fold a viewer output.
        self._relayout()
        for key, editor in self._inline_editors.items():
            with QSignalBlocker(editor):
                if isinstance(editor, QLabel):
                    value = self.node.parameters[key]
                    editor.setText(
                        ("Yes" if value else "No")
                        if isinstance(value, bool)
                        else str(value)
                    )
                elif isinstance(editor, QAbstractSpinBox):
                    editor.setValue(self.node.parameters[key])
                elif isinstance(editor, QComboBox):
                    editor.setCurrentText(str(self.node.parameters[key]))
                else:
                    editor.setChecked(bool(self.node.parameters[key]))
            if not self._read_only:
                editor.setEnabled(self.node.enabled and self.node.implemented and not self._editing_locked)
        self._update_tooltip()
        self.update()

    def set_adjacent(self, adjacent: bool) -> None:
        adjacent = bool(adjacent)
        if adjacent == self._adjacent:
            return
        self._adjacent = adjacent
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
        if selected:
            border = QColor("#45ddff")
            border_width = 5.0
        elif self._adjacent:
            border = QColor("#bf7cff")
            border_width = 2.4
        else:
            border = QColor("#4d5865")
            border_width = 1.3
        border_pen = QPen(border, border_width)
        border_pen.setCosmetic(True)
        painter.setPen(border_pen)
        painter.setBrush(
            QColor("#123e60")
            if selected and self.node.enabled
            else QColor("#272e36")
            if self.node.enabled
            else QColor("#24282d")
        )
        painter.drawRoundedRect(body, 9, 9)

        header = QRectF(1, 1, self.width - 2, self._header_height)
        status_colour = STATUS_COLOURS[self.node.status]
        if not self.node.enabled:
            status_colour = STATUS_COLOURS[NodeStatus.BYPASSED]
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#006bbb") if selected else status_colour)
        painter.drawRoundedRect(header, 8, 8)
        painter.drawRect(
            QRectF(1, 24, self.width - 2, self._header_height - 23)
        )
        if selected:
            # Keep execution status readable independently of selection.
            painter.setBrush(status_colour)
            painter.drawRect(
                QRectF(1, self._header_height - 2, self.width - 2, 3)
            )

        painter.setPen(QColor("#f6f8fa"))
        title_font = QFont(painter.font())
        title_font.setBold(True)
        title_font.setPointSizeF(10.0)
        painter.setFont(title_font)
        title_rect = QRectF(
            12, 5, self.width - 24, self._header_height - 10
        )
        if self._reference_view:
            painter.drawText(
                title_rect,
                Qt.AlignmentFlag.AlignLeft
                | Qt.AlignmentFlag.AlignVCenter
                | Qt.TextFlag.TextWordWrap,
                self.node.title,
            )
        else:
            painter.drawText(
                QRectF(12, 5, self.width - 24, 26), self.node.title
            )

        body_font = QFont(painter.font())
        body_font.setBold(False)
        body_font.setPointSizeF(8.5)
        painter.setFont(body_font)
        painter.setPen(QColor("#aeb8c3"))
        body_offset = self._header_body_offset
        painter.drawText(
            QRectF(12, 43 + body_offset, self.width - 24, 18),
            self.node.category.upper(),
        )
        painter.setPen(QColor("#e2e7ec"))
        detail = self.node.status_detail or self.node.status.value.capitalize()
        if self.node.inline_parameters:
            painter.drawText(
                QRectF(12, 62 + body_offset, self.width - 24, 18), detail
            )
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
                QRectF(12, 62 + body_offset, self.width - 24, 18),
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
        if self.node.input_ports:
            for port_id, label in self.node.input_ports:
                y = self._port_y(self.node.input_ports, port_id)
                connected = self.graph.connection_for_input(
                    self.node.identifier, port_id
                ) is not None
                painter.setPen(port_pen)
                painter.setBrush(
                    QColor("#57d8ff") if connected else QColor("#ef6b73")
                )
                painter.drawEllipse(QPointF(0, y), 6, 6)
                painter.setPen(
                    QColor("#b9c5cf") if connected else QColor("#ff9da3")
                )
                compact_label = painter.fontMetrics().elidedText(
                    label, Qt.TextElideMode.ElideRight, 90
                )
                painter.drawText(QRectF(10, y - 10, 92, 20), compact_label)
        visible_outputs = self.visible_output_ports()
        if self._has_output_expander():
            hidden = self.hidden_output_count()
            painter.setPen(QColor("#8fb8d8"))
            painter.drawText(
                QRectF(self.width - 150, self._expander_y() - 10, 138, 20),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                f"▸ {hidden} more view{'s' if hidden != 1 else ''}"
                if hidden
                else "▾ Hide views",
            )
        if visible_outputs:
            for port_id, label in visible_outputs:
                y = self._port_y(visible_outputs, port_id)
                connected = self._output_connected(port_id)
                painter.setPen(port_pen)
                painter.setBrush(
                    QColor("#70dfa7") if connected else QColor("#7d8995")
                )
                painter.drawEllipse(QPointF(self.width, y), 6, 6)
                painter.setPen(
                    QColor("#d6dee5") if connected else QColor("#98a4af")
                )
                compact_label = painter.fontMetrics().elidedText(
                    label, Qt.TextElideMode.ElideRight, 90
                )
                painter.drawText(
                    QRectF(self.width - 102, y - 10, 90, 20),
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                    compact_label,
                )

    def itemChange(self, change, value):  # noqa: N802 - Qt override
        result = super().itemChange(change, value)
        if change == self.GraphicsItemChange.ItemPositionHasChanged:
            self.node.x = float(self.pos().x())
            self.node.y = float(self.pos().y())
            self._moved_callback(self.node.identifier)
        elif change == self.GraphicsItemChange.ItemSelectedHasChanged:
            self.setZValue(4.0 if value else 2.0)
        return result

    def output_expander_rect(self) -> QRectF:
        return QRectF(self.width - 150, self._expander_y() - 11, 150, 22)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt override
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._has_output_expander()
            and self.output_expander_rect().contains(event.pos())
        ):
            self.set_outputs_expanded(not self._show_all_outputs)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().mouseReleaseEvent(event)
        self._released_callback(self.node.identifier)

    def contextMenuEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self._read_only:
            event.accept()
            return
        menu = QMenu()
        run_to = menu.addAction("Run to node")
        run_to.setEnabled(self.node.enabled and self.node.implemented)
        selected = menu.exec(event.screenPos())
        if selected is run_to:
            self._run_to_callback(self.node.identifier)
        event.accept()


class PipelineEdgeItem(QGraphicsPathItem):
    """Bezier connection between two node ports."""

    def __init__(
        self,
        connection: PipelineConnection,
        source: PipelineNodeItem,
        target: PipelineNodeItem,
        disconnect_callback,
        *,
        read_only: bool = False,
    ) -> None:
        super().__init__()
        self.connection = connection
        self.source = source
        self.target = target
        self.route_points: tuple[QPointF, ...] = ()
        self.bundle_key: tuple[str, int, int] | None = None
        self.is_bundled = False
        self._disconnect_callback = disconnect_callback
        self._read_only = bool(read_only)
        self.base_colour = _wire_colour(connection)
        self.setFlag(self.GraphicsItemFlag.ItemIsSelectable, True)
        # A connection's stroked hit area can cover much of a dense graph. If
        # it accepts ordinary left presses, QGraphicsView selects the Bezier
        # (and draws its large bounding box) instead of starting hand-drag
        # panning. The canvas handles the deliberate Ctrl-click selection
        # gesture; the item itself only needs right-click context menus.
        self.setAcceptedMouseButtons(Qt.MouseButton.RightButton)
        self.set_highlighted(False)
        self.setZValue(0)
        source_label = dict(source.node.output_ports)[connection.source_port]
        target_label = dict(target.node.input_ports)[connection.target_port]
        self.setToolTip(
            f"{source.node.title}: {source_label}\n-> "
            f"{target.node.title}: {target_label}\n"
            f"Type: {connection.data_type}\n\n"
            + (
                "Reference graph is read-only."
                if self._read_only
                else "Right-click to disconnect. Ctrl-click to select for Delete/Backspace."
            )
        )
        self.update_path()

    def set_highlighted(self, highlighted: bool) -> None:
        """Emphasize every connection incident to the selected node.

        Resting cables are thin and desaturated so the cards stay readable;
        the typed colour appears only on the selected node's connections.
        Cables touching a bypassed node are dimmed further.
        """

        if highlighted:
            colour = self.base_colour.lighter(138)
            width = 4.0
        else:
            colour = QColor.fromHsvF(
                max(0.0, self.base_colour.hsvHueF()), 0.22, 0.66
            )
            width = 1.5
        if not (self.source.node.enabled and self.target.node.enabled):
            colour.setAlphaF(0.28)
        pen = QPen(colour, width)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setZValue(1 if highlighted else 0)

    def contextMenuEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self._read_only:
            event.accept()
            return
        menu = QMenu()
        disconnect = menu.addAction("Disconnect")
        selected = menu.exec(event.screenPos())
        if selected is disconnect:
            self._disconnect_callback(self.connection)
        event.accept()

    def update_path(self) -> None:
        start = self.source.output_anchor(self.connection.source_port)
        end = self.target.input_anchor(self.connection.target_port)
        self.set_route(
            _default_connection_path(start, end),
            route_points=(start, end),
        )

    def set_route(
        self,
        path: QPainterPath,
        *,
        route_points: tuple[QPointF, ...],
        bundle_key: tuple[str, int, int] | None = None,
    ) -> None:
        """Install display geometry while retaining logical-edge identity."""

        self.route_points = tuple(QPointF(point) for point in route_points)
        self.bundle_key = bundle_key
        self.is_bundled = bundle_key is not None
        self.setPath(path)


class PipelineCableItem(QGraphicsPathItem):
    """Non-interactive conduit underlay for one shared connection trunk."""

    def __init__(
        self,
        bundle_key: tuple[str, int, int],
        route_points: tuple[QPointF, ...],
        colour: QColor,
        smooth: bool,
    ) -> None:
        super().__init__()
        self.bundle_key = bundle_key
        self.route_points = tuple(QPointF(point) for point in route_points)
        trunk = QColor(colour)
        trunk.setAlphaF(0.45)
        pen = QPen(trunk, 5.0)
        pen.setCosmetic(True)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        self.setPen(pen)
        self.setPath(
            _smooth_bundle_path(self.route_points)
            if smooth
            else _rounded_route_path(self.route_points)
        )
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.setFlag(self.GraphicsItemFlag.ItemIsSelectable, False)
        self.setZValue(-0.5)


class PipelineCanvas(QGraphicsView):
    """Zoomable, pannable, and movable Qt pipeline-node canvas."""

    node_selected = Signal(str)
    run_to_node_requested = Signal(str)
    unused_node_restored = Signal(str)
    unused_node_shelved = Signal(str)
    parameter_changed = Signal(str, str, object)
    connections_changed = Signal(object)
    connection_error = Signal(str)
    layout_changed = Signal()
    presentation_changed = Signal()
    NODE_GAP = 18.0
    OPTIONAL_TOOLBOX_NODE_IDS = frozenset({"circle_candidates"})

    def __init__(
        self,
        graph: PipelineGraph,
        parent=None,
        *,
        read_only: bool = False,
        reference_view: bool = False,
        initial_focus_node_id: str | None = None,
        preserve_initial_layout: bool = False,
    ) -> None:
        super().__init__(parent)
        self._read_only = bool(read_only)
        if reference_view and not self._read_only:
            raise ValueError("The reference canvas view must be read-only.")
        if initial_focus_node_id is not None and initial_focus_node_id not in graph.nodes:
            raise ValueError("The initial focus node must exist in the graph.")
        self._reference_view = bool(reference_view)
        self._editing_locked = False
        self._initial_focus_node_id = initial_focus_node_id
        self._workflow_controls = None
        self._catalogue_provider = None
        self._preserve_initial_layout = bool(preserve_initial_layout)
        self.graph = graph
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.node_items: dict[str, PipelineNodeItem] = {}
        self.edge_items: list[PipelineEdgeItem] = []
        self.cable_items: list[PipelineCableItem] = []
        self._zoom_steps = 0
        self._fit_pending = True
        self._arranging_nodes = False
        self._connection_drag: tuple[str, str, str] | None = None
        self._connection_selection_click = False
        self._committed_positions: dict[str, tuple[float, float]] = {}
        self.cable_bundling_enabled = False
        self.obstacle_routing_enabled = False
        self._route_refresh_timer = QTimer(self)
        self._route_refresh_timer.setSingleShot(True)
        self._route_refresh_timer.timeout.connect(self._finish_deferred_route_refresh)
        self._create_connection_preview()

        self.setBackgroundBrush(QColor("#1c2229"))
        self._finish_initialization()

    def _create_connection_preview(self) -> None:
        self._connection_preview = QGraphicsPathItem()
        preview_pen = QPen(QColor("#57d8ff"), 2.8, Qt.PenStyle.DashLine)
        preview_pen.setCosmetic(True)
        self._connection_preview.setPen(preview_pen)
        self._connection_preview.setZValue(5)
        self._connection_preview.hide()
        self._scene.addItem(self._connection_preview)

    def _finish_initialization(self) -> None:
        """Kept separate only to make the canvas setup easy to scan."""
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
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setViewportMargins(0, 38, 0, 0)
        self._build_control_bar()
        self._populate(resolve_overlaps=not self._preserve_initial_layout)
        self._scene.selectionChanged.connect(self._selection_changed)

    def rebuild_graph_items(self) -> None:
        """Recreate cards after a structural port-catalogue update."""

        selected = next(
            (
                item.node.identifier
                for item in self._scene.selectedItems()
                if isinstance(item, PipelineNodeItem)
            ),
            None,
        )
        self.node_items.clear()
        self.edge_items.clear()
        self.cable_items.clear()
        self._scene.clear()
        self._create_connection_preview()
        self._populate()
        if selected in self.node_items:
            self.select_node(selected)

    def _build_control_bar(self) -> None:
        self.control_bar = QFrame(self)
        rows = QVBoxLayout(self.control_bar)
        rows.setContentsMargins(8, 4, 8, 4)
        rows.setSpacing(4)
        rows.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
        controls = QHBoxLayout()
        rows.addLayout(controls)
        self._controls_layout = controls
        self._workflow_stacked = False
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(5)
        controls.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
        self.auto_arrange_button = QToolButton(self.control_bar)
        self.auto_arrange_button.setText("Auto arrange")
        self.auto_arrange_button.setToolTip(
            "Reposition nodes into non-overlapping dependency columns while "
            "reducing connection crossings. Manual overlaps remain allowed."
        )
        self.auto_arrange_button.clicked.connect(self.auto_arrange)
        self.move_to_unused_button = QToolButton(self.control_bar)
        self.move_to_unused_button.setText("Move to unused")
        self.move_to_unused_button.setToolTip(
            "Move the selected optional node and its connections back to the "
            "Unused nodes toolbox."
        )
        self.move_to_unused_button.setEnabled(False)
        self.move_to_unused_button.clicked.connect(
            self._shelve_selected_optional_node
        )
        self.unused_nodes_button = QToolButton(self.control_bar)
        self.unused_nodes_button.setPopupMode(
            QToolButton.ToolButtonPopupMode.InstantPopup
        )
        self.unused_nodes_button.setToolTip(
            "Nodes preserved for possible future use but excluded from the "
            "current calculation graph."
        )
        self.unused_nodes_menu = QMenu(self.unused_nodes_button)
        self.unused_nodes_button.setMenu(self.unused_nodes_menu)
        self._refresh_unused_node_toolbox()
        self.bundle_cables_button = QToolButton(self.control_bar)
        self.bundle_cables_button.setText("Bundle cables")
        self.bundle_cables_button.setCheckable(True)
        self.bundle_cables_button.setChecked(self.cable_bundling_enabled)
        self.bundle_cables_button.setToolTip(
            "Bundle compatible connections leaving the same source node into "
            "a shared cable trunk, then branch the individual wires near their "
            "destinations. This changes graph display only."
        )
        self.bundle_cables_button.toggled.connect(
            self.set_cable_bundling_enabled
        )
        self.route_around_nodes_button = QToolButton(self.control_bar)
        self.route_around_nodes_button.setText("Route around nodes")
        self.route_around_nodes_button.setCheckable(True)
        self.route_around_nodes_button.setChecked(self.obstacle_routing_enabled)
        self.route_around_nodes_button.setToolTip(
            "Route connections around node cards so their source and destination "
            "remain legible. Routes retain rounded corners where space allows. "
            "This changes graph display only."
        )
        self.route_around_nodes_button.toggled.connect(
            self.set_obstacle_routing_enabled
        )
        self.zoom_out_button = QToolButton(self.control_bar)
        self.zoom_out_button.setText("−")
        self.zoom_out_button.setToolTip("Zoom out")
        self.zoom_out_button.clicked.connect(self.zoom_out)
        self.zoom_label = QLabel("100%", self.control_bar)
        self.zoom_label.setMinimumWidth(48)
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.zoom_in_button = QToolButton(self.control_bar)
        self.zoom_in_button.setText("+")
        self.zoom_in_button.setToolTip("Zoom in")
        self.zoom_in_button.clicked.connect(self.zoom_in)
        self.fit_button = QToolButton(self.control_bar)
        self.fit_button.setText("Fit")
        self.fit_button.setToolTip("Fit the complete pipeline")
        self.fit_button.clicked.connect(self.fit_graph)
        self.actual_size_button = QToolButton(self.control_bar)
        self.actual_size_button.setText("100%")
        self.actual_size_button.setToolTip("Show pipeline at actual size")
        self.actual_size_button.clicked.connect(self.actual_size)
        self._editing_buttons = (
            self.auto_arrange_button, self.move_to_unused_button,
            self.unused_nodes_button, self.bundle_cables_button,
            self.route_around_nodes_button,
        )
        self.tools_button = QToolButton(self.control_bar)
        self.tools_button.setText("Tools")
        self.tools_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.tools_menu = QMenu(self.tools_button)
        self.tools_button.setMenu(self.tools_menu)
        self.tools_menu.aboutToShow.connect(self._populate_tools_menu)
        controls.addWidget(self.tools_button)
        self.tools_button.hide()
        controls.addWidget(self.auto_arrange_button)
        controls.addWidget(self.move_to_unused_button)
        controls.addWidget(self.unused_nodes_button)
        controls.addWidget(self.bundle_cables_button)
        controls.addWidget(self.route_around_nodes_button)
        controls.addStretch(1)
        controls.addWidget(self.zoom_out_button)
        controls.addWidget(self.zoom_label)
        controls.addWidget(self.zoom_in_button)
        controls.addWidget(self.fit_button)
        controls.addWidget(self.actual_size_button)
        style_canvas_control_bar(self.control_bar, "pipelineControlBar")
        self.control_bar.raise_()
        if self.read_only:
            for button in (self.move_to_unused_button, self.unused_nodes_button):
                button.hide()
                button.setToolTip("")

    def set_workflow_controls(self, widget) -> None:
        """Put the active workflow controls beside the existing canvas tools."""
        self._workflow_controls = widget
        widget.setParent(self.control_bar)
        self._controls_layout.insertWidget(0, widget)
        self._layout_control_bar()

    @property
    def editing_locked(self) -> bool:
        return self._editing_locked

    def set_editing_locked(self, locked: bool) -> None:
        """Lock analytical edits while retaining layout, selection and execution."""
        self._editing_locked = bool(locked)
        self._connection_drag = None
        self._connection_preview.hide()
        for item in self.node_items.values():
            item._editing_locked = self._editing_locked
            item.refresh()
        for item in self.edge_items:
            item._read_only = self.read_only or self._editing_locked
        self._refresh_unused_node_toolbox()

    def set_node_catalogue(self, provider, add_callback) -> None:
        """A searchable, colour-coded picker for independent catalogue instances."""
        self._catalogue_provider = provider
        self._catalogue_add = add_callback
        self.add_node_menu = QMenu(self.unused_nodes_button)
        panel = QWidget(self.add_node_menu)
        box = QVBoxLayout(panel)
        box.setContentsMargins(7, 7, 7, 7)
        self.node_search = QLineEdit(panel)
        self.node_search.setPlaceholderText('Find node…')
        self.node_search.setClearButtonEnabled(True)
        box.addWidget(self.node_search)
        legend = QLabel('✓ already in the pipeline (adds another instance)', panel)
        legend.setStyleSheet('color: #46515d;')
        box.addWidget(legend)
        self.node_catalogue_list = QListWidget(panel)
        self.node_catalogue_list.setMinimumSize(320, 280)
        self.node_catalogue_list.setMaximumHeight(460)
        box.addWidget(self.node_catalogue_list)
        action = QWidgetAction(self.add_node_menu)
        action.setDefaultWidget(panel)
        self.add_node_menu.addAction(action)
        self.unused_nodes_button.setMenu(self.add_node_menu)
        self.add_node_button = self.unused_nodes_button
        self.node_search.textChanged.connect(self._filter_node_catalogue)
        self.node_search.returnPressed.connect(self._add_first_catalogue_match)
        self.node_catalogue_list.itemClicked.connect(self._add_catalogue_item)
        self.node_catalogue_list.itemActivated.connect(self._add_catalogue_item)
        self.add_node_menu.aboutToShow.connect(self._populate_node_catalogue)
        self._refresh_unused_node_toolbox()
        self.unused_nodes_button.setToolTip('Add a catalogue node with its required inputs. Selecting an existing node adds another independent instance.')

    CATALOGUE_HEADER_ROLE = Qt.ItemDataRole.UserRole + 2

    def _populate_node_catalogue(self) -> None:
        self.node_catalogue_list.clear()
        grouped: dict[str, list[dict]] = {}
        for entry in self._catalogue_provider():
            grouped.setdefault(entry.get('category', '') or 'Other', []).append(entry)
        for category, entries in sorted(grouped.items(), key=lambda item: category_rank(item[0])):
            # Non-selectable category headings keep 120+ entries scannable.
            header = QListWidgetItem(category.upper())
            header.setData(self.CATALOGUE_HEADER_ROLE, category)
            header.setFlags(Qt.ItemFlag.NoItemFlags)
            font = header.font()
            font.setBold(True)
            header.setFont(font)
            header.setForeground(QColor('#46515d'))
            self.node_catalogue_list.addItem(header)
            for entry in entries:
                count = entry.get('count', int(entry['present']))
                item = QListWidgetItem(('✓ ' if entry['present'] else '') + entry['title'])
                item.setData(Qt.ItemDataRole.UserRole, entry['key'])
                item.setData(Qt.ItemDataRole.UserRole + 1, category)
                item.setForeground(QColor('#1768ad' if entry['present'] else '#218044'))
                item.setToolTip(entry['description'] + '\n' +
                    (f'{count} instance(s) present. Add another independent instance.' if count else 'Add an independent instance.'))
                self.node_catalogue_list.addItem(item)
        self._filter_node_catalogue(self.node_search.text())
        self.node_search.setFocus()

    def catalogue_entry_items(self) -> list[QListWidgetItem]:
        """Selectable catalogue rows, excluding category headings."""
        return [
            self.node_catalogue_list.item(index)
            for index in range(self.node_catalogue_list.count())
            if self.node_catalogue_list.item(index).data(self.CATALOGUE_HEADER_ROLE) is None
        ]

    def _filter_node_catalogue(self, text) -> None:
        needle = text.casefold().strip()
        visible_categories = set()
        for item in self.catalogue_entry_items():
            category = str(item.data(Qt.ItemDataRole.UserRole + 1))
            hidden = needle not in (item.text() + ' ' + category).casefold()
            item.setHidden(hidden)
            if not hidden:
                visible_categories.add(category)
        for index in range(self.node_catalogue_list.count()):
            item = self.node_catalogue_list.item(index)
            category = item.data(self.CATALOGUE_HEADER_ROLE)
            if category is not None:
                item.setHidden(category not in visible_categories)

    def _add_catalogue_item(self, item) -> None:
        if self.read_only or self._editing_locked:
            return
        if item.data(self.CATALOGUE_HEADER_ROLE) is not None:
            return
        if not self.add_node_menu.isVisible():
            return
        self.add_node_menu.close()
        self._catalogue_add(item.data(Qt.ItemDataRole.UserRole))

    def _add_first_catalogue_match(self) -> None:
        for item in self.catalogue_entry_items():
            if not item.isHidden():
                self._add_catalogue_item(item)
                return

    def _populate_tools_menu(self) -> None:
        """Keep all editing commands accessible when the pane is narrow."""
        self.tools_menu.clear()
        for button in self._editing_buttons:
            if self.read_only and button in (
                self.move_to_unused_button,
                self.unused_nodes_button,
            ):
                continue
            if button is self.unused_nodes_button:
                menu = button.menu()
                menu.setTitle(button.text())
                self.tools_menu.addMenu(menu).setEnabled(button.isEnabled())
                continue
            action = self.tools_menu.addAction(button.text())
            action.setEnabled(button.isEnabled())
            action.setCheckable(button.isCheckable())
            action.setChecked(button.isChecked())
            action.triggered.connect(lambda checked=False, button=button: button.click())

    def _layout_control_bar(self) -> None:
        width = max(120, self.viewport().width())
        if self._workflow_controls is not None:
            adjust = getattr(self._workflow_controls, 'adjust_for_width', None)
            if adjust is not None:
                adjust(width - 16)
        zoom_controls = (self.zoom_out_button, self.zoom_label,
                         self.zoom_in_button, self.fit_button, self.actual_size_button)
        editing_buttons = tuple(
            button
            for button in self._editing_buttons
            if not self.read_only
            or button not in (self.move_to_unused_button, self.unused_nodes_button)
        )
        required = (self._workflow_controls.sizeHint().width() if self._workflow_controls is not None else 0) + sum(button.sizeHint().width() for button in (
            *editing_buttons, *zoom_controls
        )) + 80
        compact = width < required
        for button in self._editing_buttons:
            visible = not compact or (button is self.unused_nodes_button and self._catalogue_provider is not None)
            if self.read_only and button in (
                self.move_to_unused_button,
                self.unused_nodes_button,
            ):
                visible = False
            button.setVisible(visible)
        self.tools_button.setVisible(compact)
        minimum_row = sum(button.sizeHint().width() for button in zoom_controls) + self.tools_button.sizeHint().width() + 60
        if self._catalogue_provider is not None:
            minimum_row += self.unused_nodes_button.sizeHint().width() + 5
        stacked = self._workflow_controls is not None and width < self._workflow_controls.sizeHint().width() + minimum_row
        if stacked != self._workflow_stacked:
            self._controls_layout.removeWidget(self._workflow_controls)
            self.control_bar.layout().removeWidget(self._workflow_controls)
            if stacked:
                self.control_bar.layout().insertWidget(0, self._workflow_controls)
            else:
                self._controls_layout.insertWidget(0, self._workflow_controls)
            self._workflow_stacked = stacked
        height = 68 if stacked else 38
        if self.viewportMargins().top() != height:
            self.setViewportMargins(0, height, 0, 0)
        self.control_bar.setGeometry(self.viewport().x(), 0, width, height)
        self.control_bar.raise_()

    def _populate(self, *, resolve_overlaps: bool = True) -> None:
        for node in self.graph.nodes.values():
            item = PipelineNodeItem(
                node,
                self.graph,
                self._node_moved,
                self._node_released,
                self._inline_parameter_changed,
                self._request_run_to_node,
                read_only=self.read_only,
                reference_view=self._reference_view,
                editing_locked=self._editing_locked,
            )
            self.node_items[node.identifier] = item
            self._scene.addItem(item)
        if resolve_overlaps:
            self._resolve_all_overlaps()
        for connection in self.graph.connections:
            edge = PipelineEdgeItem(
                connection,
                self.node_items[connection.source],
                self.node_items[connection.target],
                self._disconnect_connection,
                read_only=self.read_only or self._editing_locked,
            )
            self.edge_items.append(edge)
            self._scene.addItem(edge)
        self._update_edge_paths()
        self._update_scene_rect()
        self._committed_positions = {
            node_id: (float(item.pos().x()), float(item.pos().y()))
            for node_id, item in self.node_items.items()
        }

    def rebuild_from_graph(self, preferred_node_id: str | None = None) -> str | None:
        """Rebuild scene items after an atomic graph topology replacement.

        Analytical settings profiles can move nodes between the active graph and
        the unused-node toolbox and can replace authored connections in one core
        transaction. Reusing the existing scene items after that transaction
        would leave stale cards, ports, or edges. This narrow rebuild keeps the
        canvas-only cable options intact and restores the exact positions retained
        by the graph instead of emitting any parameter or connection callbacks.

        The returned identifier is the surviving node selected after the rebuild,
        or ``None`` when the active graph is empty.
        """

        selected_ids = [
            item.node.identifier
            for item in self._scene.selectedItems()
            if isinstance(item, PipelineNodeItem)
        ]
        candidate_ids = (
            preferred_node_id,
            selected_ids[0] if selected_ids else None,
            next(iter(self.graph.nodes), None),
        )
        selected_id = next(
            (
                node_id
                for node_id in candidate_ids
                if node_id is not None and node_id in self.graph.nodes
            ),
            None,
        )

        self._route_refresh_timer.stop()
        self._connection_drag = None
        self._connection_selection_click = False
        self.node_items.clear()
        self.edge_items.clear()
        self.cable_items.clear()
        self._scene.clear()
        self._create_connection_preview()
        self._populate(resolve_overlaps=False)
        self._refresh_unused_node_toolbox()
        self.move_to_unused_button.setEnabled(False)
        if selected_id is not None:
            self.select_node(selected_id)
        return selected_id

    def _rebuild_edges(self) -> None:
        self._clear_cable_items()
        for edge in self.edge_items:
            self._scene.removeItem(edge)
        self.edge_items.clear()
        for connection in self.graph.connections:
            edge = PipelineEdgeItem(
                connection,
                self.node_items[connection.source],
                self.node_items[connection.target],
                self._disconnect_connection,
                read_only=self.read_only or self._editing_locked,
            )
            self.edge_items.append(edge)
            self._scene.addItem(edge)
        # Refresh cards first: a new or removed connection can change which
        # outputs are visible, and edge anchors depend on that geometry.
        for item in self.node_items.values():
            item.refresh()
        self._update_edge_paths()
        self._update_scene_rect()

    def set_cable_bundling_enabled(self, enabled: bool) -> None:
        """Enable or disable source/direction cable grouping for display."""

        enabled = bool(enabled)
        changed = enabled != self.cable_bundling_enabled
        self.cable_bundling_enabled = enabled
        if (
            hasattr(self, "bundle_cables_button")
            and self.bundle_cables_button.isChecked() != enabled
        ):
            with QSignalBlocker(self.bundle_cables_button):
                self.bundle_cables_button.setChecked(enabled)
        if changed:
            self._update_edge_paths()
            self._update_scene_rect()
            self.presentation_changed.emit()

    def set_obstacle_routing_enabled(self, enabled: bool) -> None:
        """Enable or disable connection routing around other node cards."""

        enabled = bool(enabled)
        changed = enabled != self.obstacle_routing_enabled
        self.obstacle_routing_enabled = enabled
        if (
            hasattr(self, "route_around_nodes_button")
            and self.route_around_nodes_button.isChecked() != enabled
        ):
            with QSignalBlocker(self.route_around_nodes_button):
                self.route_around_nodes_button.setChecked(enabled)
        if changed:
            self._update_edge_paths()
            self._update_scene_rect()
            self.presentation_changed.emit()

    def _clear_cable_items(self) -> None:
        for cable in self.cable_items:
            self._scene.removeItem(cable)
        self.cable_items.clear()

    def _node_obstacles(self, excluded: frozenset[str]) -> tuple[QRectF, ...]:
        # Reserve the rounding radius in addition to the visible clearance;
        # otherwise a rounded corner could cut back through a card even when
        # its rectilinear control segments only touched the expanded boundary.
        margin = _ROUTE_CLEARANCE + _ROUTE_CORNER_RADIUS
        return tuple(
            item.mapRectToScene(QRectF(0.0, 0.0, item.width, item.height)).adjusted(
                -margin, -margin, margin, margin
            )
            for node_id, item in self.node_items.items()
            if node_id not in excluded
        )

    @staticmethod
    def _bundle_key_for_edge(
        edge: PipelineEdgeItem,
        start: QPointF,
        end: QPointF,
    ) -> tuple[str, int, int]:
        dx = float(end.x()) - float(start.x())
        dy = float(end.y()) - float(start.y())
        side = 1 if dx >= 0.0 else -1
        angle = math.atan2(dy, max(1.0, abs(dx)))
        direction_bucket = int(round(angle / math.radians(30.0)))
        return edge.connection.source, side, direction_bucket

    def _single_edge_route(
        self,
        edge: PipelineEdgeItem,
        start: QPointF,
        end: QPointF,
    ) -> tuple[QPointF, ...]:
        if not self.obstacle_routing_enabled:
            return (QPointF(start), QPointF(end))
        obstacles = self._node_obstacles(
            frozenset((edge.connection.source, edge.connection.target))
        )
        source_exit = QPointF(float(start.x()) + _ROUTE_LEAD, float(start.y()))
        target_entry = QPointF(float(end.x()) - _ROUTE_LEAD, float(end.y()))
        middle = _orthogonal_route(source_exit, target_entry, obstacles)
        return _compressed_route_points((start, *middle, end))

    def _install_bundle(
        self,
        bundle_key: tuple[str, int, int],
        edges: tuple[PipelineEdgeItem, ...],
        endpoints: dict[PipelineEdgeItem, tuple[QPointF, QPointF]],
    ) -> bool:
        """Install a common trunk plus independently interactive branches."""

        _, side, _ = bundle_key
        starts = [endpoints[edge][0] for edge in edges]
        ends = [endpoints[edge][1] for edge in edges]
        source_x = sum(float(point.x()) for point in starts) / len(starts)
        source_y = sum(float(point.y()) for point in starts) / len(starts)
        # Inputs are always on their cards' left edges, so a branch junction is
        # placed to the left of the leftmost destination regardless of whether
        # the destinations were manually dragged ahead of or behind the source.
        nearest_target_x = min(float(point.x()) for point in ends)
        split_margin = _ROUTE_CLEARANCE + _ROUTE_CORNER_RADIUS + 4.0
        # Output sockets always live on the right edge of a card, even when a
        # manually positioned target is to its left. Every cable first exits
        # rightward; a reverse-facing trunk then wraps around the source.
        origin = QPointF(source_x + _ROUTE_LEAD, source_y)
        split = QPointF(
            nearest_target_x - split_margin,
            sum(float(point.y()) for point in ends) / len(ends),
        )
        if side * (float(split.x()) - float(origin.x())) < 14.0:
            return False

        if self.obstacle_routing_enabled:
            trunk_obstacles = self._node_obstacles(
                frozenset((bundle_key[0],)) if side > 0 else frozenset()
            )
            desired_y = float(split.y())
            split_x = float(split.x())
            spanning = tuple(
                obstacle
                for obstacle in trunk_obstacles
                if float(obstacle.left()) - 0.01
                <= split_x
                <= float(obstacle.right()) + 0.01
            )
            containing = tuple(
                boundary
                for obstacle in spanning
                if float(obstacle.top()) < desired_y < float(obstacle.bottom())
                for boundary in (float(obstacle.top()), float(obstacle.bottom()))
            )
            nearby = sorted(
                {
                    boundary
                    for obstacle in spanning
                    for boundary in (
                        float(obstacle.top()),
                        float(obstacle.bottom()),
                    )
                },
                key=lambda boundary: (abs(boundary - desired_y), boundary),
            )
            split_candidates = tuple(
                dict.fromkeys(
                    (desired_y, float(origin.y()), *containing, *nearby[:4])
                )
            )
            selected_trunk: tuple[QPointF, tuple[QPointF, ...]] | None = None
            for split_y in split_candidates:
                candidate_split = QPointF(float(split.x()), split_y)
                candidate_points = _compressed_route_points(
                    _orthogonal_route(origin, candidate_split, trunk_obstacles)
                )
                if len(candidate_points) < 2 or not _route_clears_obstacles(
                    candidate_points, trunk_obstacles
                ):
                    continue
                selected_trunk = candidate_split, candidate_points
                break
            if selected_trunk is None:
                return False
            split, trunk_points = selected_trunk
        else:
            trunk_points = (origin, split)
        trunk_points = _compressed_route_points(trunk_points)
        if len(trunk_points) < 2:
            return False

        planned_routes: list[tuple[PipelineEdgeItem, tuple[QPointF, ...]]] = []
        for edge in edges:
            start, end = endpoints[edge]
            if self.obstacle_routing_enabled:
                obstacles = self._node_obstacles(
                    frozenset((edge.connection.source, edge.connection.target))
                )
                entry_points = _orthogonal_route(start, origin, obstacles)
                target_entry = QPointF(
                    float(end.x()) - _ROUTE_LEAD,
                    float(end.y()),
                )
                branch_points = _orthogonal_route(split, target_entry, obstacles)
                points = _compressed_route_points(
                    (*entry_points, *trunk_points, *branch_points, end)
                )
                if not _route_clears_obstacles(points, obstacles):
                    return False
            else:
                points = _compressed_route_points((start, *trunk_points, end))
            planned_routes.append((edge, points))

        for edge, points in planned_routes:
            edge.set_route(
                (
                    _rounded_route_path(points)
                    if self.obstacle_routing_enabled
                    else _smooth_bundle_path(points)
                ),
                route_points=points,
                bundle_key=bundle_key,
            )

        cable = PipelineCableItem(
            bundle_key,
            trunk_points,
            _source_cable_colour(bundle_key[0]),
            not self.obstacle_routing_enabled,
        )
        cable.setToolTip(
            f"{len(edges)} connections bundled from "
            f"{self.node_items[bundle_key[0]].node.title}"
        )
        self.cable_items.append(cable)
        self._scene.addItem(cable)
        return True

    def _update_edge_paths(self) -> None:
        """Recompute default, obstacle-aware, and bundled display routes."""

        self._clear_cable_items()
        endpoints = {
            edge: (
                edge.source.output_anchor(edge.connection.source_port),
                edge.target.input_anchor(edge.connection.target_port),
            )
            for edge in self.edge_items
        }
        bundled: set[PipelineEdgeItem] = set()
        if self.cable_bundling_enabled:
            candidates: dict[tuple[str, int, int], list[PipelineEdgeItem]] = {}
            for edge, (start, end) in endpoints.items():
                key = self._bundle_key_for_edge(edge, start, end)
                candidates.setdefault(key, []).append(edge)
            for bundle_key in sorted(candidates):
                group = tuple(
                    sorted(
                        candidates[bundle_key],
                        key=lambda item: (
                            item.connection.target,
                            item.connection.target_port,
                            item.connection.source_port,
                        ),
                    )
                )
                if len(group) < 2:
                    continue
                if self._install_bundle(bundle_key, group, endpoints):
                    bundled.update(group)

        for edge, (start, end) in endpoints.items():
            if edge in bundled:
                continue
            points = self._single_edge_route(edge, start, end)
            path = (
                _rounded_route_path(points)
                if self.obstacle_routing_enabled
                else _default_connection_path(start, end)
            )
            edge.set_route(path, route_points=points)

    def _disconnect_connection(self, connection: PipelineConnection) -> None:
        if self.read_only or self._editing_locked:
            return
        try:
            affected = self.graph.disconnect(connection)
        except ValueError as error:
            self.connection_error.emit(str(error))
            return
        self._rebuild_edges()
        self.connections_changed.emit(affected)

    def _port_at(
        self, scene_position: QPointF
    ) -> tuple[PipelineNodeItem, str, str] | None:
        for item in reversed(list(self.node_items.values())):
            port = item.port_at_scene(scene_position)
            if port is not None:
                direction, port_id = port
                return item, direction, port_id
        return None

    @staticmethod
    def _connection_path(start: QPointF, end: QPointF) -> QPainterPath:
        return _default_connection_path(start, end)

    def _drag_endpoints(self, cursor: QPointF) -> tuple[QPointF, QPointF]:
        assert self._connection_drag is not None
        node_id, direction, port_id = self._connection_drag
        item = self.node_items[node_id]
        if direction == "output":
            return item.output_anchor(port_id), cursor
        return cursor, item.input_anchor(port_id)

    def _finish_connection_drag(self, scene_position: QPointF) -> None:
        if self.read_only or self._editing_locked:
            self._connection_drag = None
            self._connection_preview.hide()
            return
        assert self._connection_drag is not None
        start_node, start_direction, start_port = self._connection_drag
        destination = self._port_at(scene_position)
        self._connection_drag = None
        self._connection_preview.hide()
        if destination is None:
            if start_direction == "input":
                existing = self.graph.connection_for_input(start_node, start_port)
                if existing is not None:
                    self._disconnect_connection(existing)
            return
        item, end_direction, end_port = destination
        if end_direction == start_direction:
            self.connection_error.emit("Join an output socket to an input socket.")
            return
        if start_direction == "output":
            source, source_port = start_node, start_port
            target, target_port = item.node.identifier, end_port
        else:
            source, source_port = item.node.identifier, end_port
            target, target_port = start_node, start_port
        try:
            affected = self.graph.connect(
                source, source_port, target, target_port
            )
        except ValueError as error:
            self.connection_error.emit(str(error))
            return
        if affected:
            self._rebuild_edges()
            self.connections_changed.emit(affected)

    def _node_moved(self, node_id: str) -> None:
        if self._arranging_nodes:
            return
        incident = [
            edge
            for edge in self.edge_items
            if node_id in (edge.connection.source, edge.connection.target)
        ]
        for edge in incident:
            start = edge.source.output_anchor(edge.connection.source_port)
            end = edge.target.input_anchor(edge.connection.target_port)
            # Keep socket attachment fluid during the drag; the exact routed
            # path is coalesced below and forced once on mouse release.
            edge.set_route(
                _default_connection_path(start, end),
                route_points=(start, end),
            )
        if self.cable_bundling_enabled or self.obstacle_routing_enabled:
            # A full obstacle/bundle solve is deliberately deferred during a
            # drag. Incident wires remain attached immediately, node cards sit
            # above non-incident wires, and releasing performs the exact solve.
            for cable in self.cable_items:
                cable.hide()
            self._route_refresh_timer.start(140)
        self._update_scene_rect()

    def _node_released(self, node_id: str) -> None:
        """Keep the user's exact drop position, including intentional overlaps."""

        self._route_refresh_timer.stop()
        self._update_edge_paths()
        self._update_scene_rect()
        item = self.node_items.get(node_id)
        if item is None:
            return
        position = (float(item.pos().x()), float(item.pos().y()))
        previous = self._committed_positions.get(node_id)
        self._committed_positions[node_id] = position
        if previous is not None and position != previous:
            self.layout_changed.emit()

    def _finish_deferred_route_refresh(self) -> None:
        self._update_edge_paths()
        self._update_scene_rect()

    def auto_arrange(self) -> None:
        """Apply a compact Sugiyama-style layout with barycentric ordering."""

        topological = self.graph.topological_order()
        ranks: dict[str, int] = {}
        for node_id in topological:
            upstream = self.graph.upstream(node_id)
            ranks[node_id] = (
                0 if not upstream else max(ranks[parent] + 1 for parent in upstream)
            )
        columns: dict[int, list[str]] = {}
        for node_id in topological:
            columns.setdefault(ranks[node_id], []).append(node_id)
        for node_ids in columns.values():
            node_ids.sort(key=lambda identifier: float(self.node_items[identifier].pos().y()))

        def positions() -> dict[str, float]:
            return {
                node_id: float(index)
                for node_ids in columns.values()
                for index, node_id in enumerate(node_ids)
            }

        for _ in range(5):
            ordering = positions()
            for rank in sorted(columns):
                if rank == 0:
                    continue
                columns[rank].sort(
                    key=lambda node_id: (
                        sum(ordering[parent] for parent in self.graph.upstream(node_id))
                        / max(1, len(self.graph.upstream(node_id))),
                        ordering[node_id],
                    )
                )
            ordering = positions()
            for rank in sorted(columns, reverse=True):
                children_by_node = {
                    node_id: self.graph.downstream(node_id)
                    for node_id in columns[rank]
                }
                columns[rank].sort(
                    key=lambda node_id: (
                        sum(ordering[child] for child in children_by_node[node_id])
                        / max(1, len(children_by_node[node_id]))
                        if children_by_node[node_id]
                        else ordering[node_id],
                        ordering[node_id],
                    )
                )

        self._arranging_nodes = True
        try:
            for rank, node_ids in columns.items():
                total_height = sum(
                    self.node_items[node_id].height for node_id in node_ids
                ) + self.NODE_GAP * max(0, len(node_ids) - 1)
                y = -total_height * 0.5
                for node_id in node_ids:
                    item = self.node_items[node_id]
                    item.setPos(rank * 270.0, y)
                    y += item.height + self.NODE_GAP
        finally:
            self._arranging_nodes = False
        self._update_edge_paths()
        self._update_scene_rect()
        positions_after = {
            node_id: (float(item.pos().x()), float(item.pos().y()))
            for node_id, item in self.node_items.items()
        }
        if positions_after != self._committed_positions:
            self._committed_positions = positions_after
            self.layout_changed.emit()

    def _refresh_unused_node_toolbox(self) -> None:
        self.unused_nodes_menu.clear()
        count = len(self.graph.unused_nodes)
        self.unused_nodes_button.setText('Add node')
        self.unused_nodes_button.setEnabled((bool(count) or self._catalogue_provider is not None) and not self.read_only and not self._editing_locked)
        if self.read_only:
            self.unused_nodes_button.setToolTip("")
            return
        if self._editing_locked:
            self.unused_nodes_button.setToolTip('Unlock the pipeline to add nodes.')
        elif self._catalogue_provider is not None:
            self.unused_nodes_button.setToolTip('Add a catalogue node with its required inputs. Selecting an existing node adds another independent instance.')
        # The button reads "Add node", so entries name only the node.
        for node in self.graph.unused_nodes.values():
            action = self.unused_nodes_menu.addAction(node.title)
            action.setToolTip(node.description)
            action.setStatusTip(node.description)
            action.triggered.connect(
                lambda _checked=False, node_id=node.identifier: (
                    self._restore_unused_node(node_id)
                )
            )

    def _restore_unused_node(self, node_id: str) -> None:
        if self.read_only or self._editing_locked:
            return
        self.graph.restore_unused_node(node_id)
        self.node_items.clear()
        self.edge_items.clear()
        self.cable_items.clear()
        self._scene.clear()
        self._create_connection_preview()
        self._populate()
        self._refresh_unused_node_toolbox()
        self.select_node(node_id)
        self.unused_node_restored.emit(node_id)

    def _shelve_selected_optional_node(self) -> None:
        if self.read_only or self._editing_locked:
            return
        selected = [
            item
            for item in self._scene.selectedItems()
            if isinstance(item, PipelineNodeItem)
        ]
        if not selected:
            return
        node_id = selected[0].node.identifier
        if node_id not in self.OPTIONAL_TOOLBOX_NODE_IDS:
            return
        self.graph.shelve_node(node_id)
        shelved = self.graph.node(node_id)
        shelved.enabled = False
        shelved.status = NodeStatus.BYPASSED
        shelved.status_detail = "Disabled in the unused-node toolbox"
        self.node_items.clear()
        self.edge_items.clear()
        self.cable_items.clear()
        self._scene.clear()
        self._create_connection_preview()
        self._populate()
        self._refresh_unused_node_toolbox()
        self.move_to_unused_button.setEnabled(False)
        self.unused_node_shelved.emit(node_id)

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
        if self.read_only or self._editing_locked:
            return
        self.parameter_changed.emit(node_id, key, value)

    @property
    def read_only(self) -> bool:
        """Whether this canvas is a fixed reference view."""

        return self._read_only

    def _request_run_to_node(self, node_id: str) -> None:
        if not self.read_only:
            self.run_to_node_requested.emit(node_id)

    def _update_scene_rect(self) -> None:
        bounds = self._scene.itemsBoundingRect().adjusted(-90, -90, 90, 90)
        self._scene.setSceneRect(bounds)

    def _selection_changed(self) -> None:
        selected = [
            item for item in self._scene.selectedItems() if isinstance(item, PipelineNodeItem)
        ]
        selected_id = selected[0].node.identifier if selected else None
        adjacent_ids = set()
        if selected_id is not None:
            adjacent_ids.update(self.graph.upstream(selected_id))
            adjacent_ids.update(self.graph.downstream(selected_id))
        for node_id, item in self.node_items.items():
            item.set_adjacent(node_id in adjacent_ids)
        for edge in self.edge_items:
            edge.set_highlighted(
                selected_id is not None
                and selected_id
                in (edge.connection.source, edge.connection.target)
            )
        self.move_to_unused_button.setEnabled(
            selected_id in self.OPTIONAL_TOOLBOX_NODE_IDS
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
            item = self.node_items.get(node_id)
            if item is not None:
                item.refresh()
        self._update_edge_paths()
        # Enabling or bypassing a node changes how its cables are dimmed.
        selected = [
            item for item in self._scene.selectedItems() if isinstance(item, PipelineNodeItem)
        ]
        selected_id = selected[0].node.identifier if selected else None
        for edge in self.edge_items:
            edge.set_highlighted(
                selected_id is not None
                and selected_id in (edge.connection.source, edge.connection.target)
            )

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            scene_position = self.mapToScene(event.position().toPoint())
            hit = self._port_at(scene_position)
            if hit is not None:
                if self.read_only or self._editing_locked:
                    if not event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                        self._scene.clearSelection()
                    hit[0].setSelected(True)
                    event.accept()
                    return
                item, direction, port_id = hit
                self._connection_drag = (item.node.identifier, direction, port_id)
                start, end = self._drag_endpoints(scene_position)
                self._connection_preview.setPath(
                    self._connection_path(start, end)
                )
                self._connection_preview.show()
                event.accept()
                return
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                graphics_item = self.itemAt(event.position().toPoint())
                if isinstance(graphics_item, PipelineEdgeItem):
                    if not (
                        event.modifiers() & Qt.KeyboardModifier.ShiftModifier
                    ):
                        self._scene.clearSelection()
                    graphics_item.setSelected(True)
                    self._connection_selection_click = True
                    self.setFocus(Qt.FocusReason.MouseFocusReason)
                    event.accept()
                    return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._connection_drag is not None:
            scene_position = self.mapToScene(event.position().toPoint())
            start, end = self._drag_endpoints(scene_position)
            self._connection_preview.setPath(self._connection_path(start, end))
            colour = QColor("#57d8ff")
            destination = self._port_at(scene_position)
            if destination is not None:
                item, end_direction, end_port = destination
                start_node, start_direction, start_port = self._connection_drag
                if end_direction == start_direction:
                    colour = QColor("#ef6b73")
                else:
                    if start_direction == "output":
                        endpoints = (
                            start_node,
                            start_port,
                            item.node.identifier,
                            end_port,
                        )
                    else:
                        endpoints = (
                            item.node.identifier,
                            end_port,
                            start_node,
                            start_port,
                        )
                    colour = (
                        QColor("#70dfa7")
                        if self.graph.connection_template(*endpoints) is not None
                        else QColor("#ef6b73")
                    )
            pen = QPen(colour, 2.8, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            self._connection_preview.setPen(pen)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if (
            self._connection_selection_click
            and event.button() == Qt.MouseButton.LeftButton
        ):
            self._connection_selection_click = False
            event.accept()
            return
        if (
            self._connection_drag is not None
            and event.button() == Qt.MouseButton.LeftButton
        ):
            self._finish_connection_drag(
                self.mapToScene(event.position().toPoint())
            )
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape and self._connection_drag is not None:
            self._connection_drag = None
            self._connection_preview.hide()
            event.accept()
            return
        if (self.read_only or self._editing_locked) and event.key() in (
            Qt.Key.Key_Delete,
            Qt.Key.Key_Backspace,
        ):
            event.accept()
            return
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            selected_connections = [
                item.connection
                for item in self._scene.selectedItems()
                if isinstance(item, PipelineEdgeItem)
            ]
            if selected_connections:
                affected: set[str] = set()
                for connection in selected_connections:
                    affected.update(self.graph.disconnect(connection))
                self._rebuild_edges()
                self.connections_changed.emit(tuple(affected))
                event.accept()
                return
        super().keyPressEvent(event)


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
        self._update_zoom_indicator()

    def zoom_in(self) -> None:
        self._zoom_by(1.18)

    def zoom_out(self) -> None:
        self._zoom_by(1.0 / 1.18)

    def actual_size(self) -> None:
        centre = self.mapToScene(self.viewport().rect().center())
        self.resetTransform()
        self.centerOn(centre)
        self._zoom_steps = 0
        self._update_zoom_indicator()

    def _zoom_by(self, factor: float) -> None:
        current = float(self.transform().m11())
        target = current * float(factor)
        if not 0.12 <= target <= 5.0:
            return
        self.scale(float(factor), float(factor))
        self._update_zoom_indicator()

    def _update_zoom_indicator(self) -> None:
        self.zoom_label.setText(f"{self.transform().m11() * 100.0:.0f}%")

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().showEvent(event)
        if self._fit_pending:
            QTimer.singleShot(0, self._finish_pending_fit)

    def _finish_pending_fit(self) -> None:
        self._fit_pending = False
        self.fit_graph()
        if self._reference_view and self._initial_focus_node_id in self.node_items:
            item = self.node_items[self._initial_focus_node_id]
            self.resetTransform()
            self.centerOn(item.sceneBoundingRect().center())
            self._zoom_steps = 0
            self._update_zoom_indicator()

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 - Qt override
        direction = 1 if event.angleDelta().y() > 0 else -1
        self._zoom_by(1.18 if direction > 0 else 1 / 1.18)
        self._zoom_steps += direction

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._layout_control_bar()

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
