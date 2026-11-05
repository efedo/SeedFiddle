"""Shared styling for controls drawn over the dark image and graph canvases."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPalette, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QFrame, QLabel, QToolButton


CANVAS_CONTROL_BACKGROUND = "#252d35"
CANVAS_BUTTON_BACKGROUND = "#364452"
CANVAS_BUTTON_TEXT = "#f2f6fa"
CANVAS_DISABLED_BUTTON_BACKGROUND = "#303a44"
CANVAS_DISABLED_BUTTON_TEXT = "#aeb9c4"
COMPACT_ICON_SIZE = QSize(16, 16)
CHECK_ICON_SIZE = QSize(12, 12)
# Icons are painted at this multiple of their logical size to stay sharp on
# high-density screens.
_ICON_SCALE = 3


def _icon_pixmap(size: QSize) -> tuple[QPixmap, QPainter]:
    pixmap = QPixmap(size * _ICON_SCALE)
    pixmap.setDevicePixelRatio(_ICON_SCALE)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    return pixmap, painter


def _polygon(*points: tuple[float, float]) -> QPainterPath:
    path = QPainterPath()
    path.addPolygon(QPolygonF([QPointF(x, y) for x, y in points]))
    path.closeSubpath()
    return path


def _flat_pixmap(shape: QPainterPath, colour: str) -> QPixmap:
    pixmap, painter = _icon_pixmap(COMPACT_ICON_SIZE)
    painter.fillPath(shape, QColor(colour))
    painter.end()
    return pixmap


def _flat_icon(shape: QPainterPath) -> QIcon:
    """One flat colour with no outlines, from a shape on the 16-unit compact grid."""

    icon = QIcon()
    icon.addPixmap(_flat_pixmap(shape, CANVAS_BUTTON_TEXT), QIcon.Mode.Normal)
    icon.addPixmap(_flat_pixmap(shape, CANVAS_DISABLED_BUTTON_TEXT), QIcon.Mode.Disabled)
    return icon


def rename_icon() -> QIcon:
    """A pencil, its cap split off by a gap, writing towards a wedge on the baseline."""

    shape = _polygon((0.5, 14.7), (0.5, 11.5), (8.0, 4.0), (11.2, 7.2), (3.7, 14.7))
    shape.addPath(_polygon((10.45, 11.4), (15.5, 11.4), (15.5, 14.7), (7.15, 14.7)))
    shape.moveTo(8.88, 3.12)
    shape.lineTo(10.3, 1.7)
    shape.quadTo(11.0, 1.0, 11.7, 1.7)
    shape.lineTo(13.5, 3.5)
    shape.quadTo(14.2, 4.2, 13.5, 4.9)
    shape.lineTo(12.08, 6.32)
    shape.closeSubpath()
    return _flat_icon(shape)


def load_icon() -> QIcon:
    """An open folder: a tabbed back panel behind a slanted front flap."""

    shape = _polygon((1.0, 2.8), (6.0, 2.8), (7.6, 4.6), (14.0, 4.6), (14.0, 6.2), (3.3, 6.2), (1.0, 11.6))
    shape.addPath(_polygon((4.2, 7.3), (15.7, 7.3), (13.2, 13.2), (1.7, 13.2)))
    return _flat_icon(shape)


def save_icon() -> QIcon:
    """A floppy disk: a clipped square with a shutter window and hub cut out of it."""

    shape = _polygon((1.5, 1.5), (11.5, 1.5), (14.5, 4.5), (14.5, 14.5), (1.5, 14.5))
    shape.addRect(QRectF(3.5, 3.4, 6.5, 2.9))
    shape.addEllipse(QPointF(8.0, 10.6), 2.2, 2.2)
    return _flat_icon(shape)


def add_icon() -> QIcon:
    """A plus sign, drawn as one outline so its arms do not cancel where they cross."""

    return _flat_icon(_polygon(
        (6.5, 1.5), (9.5, 1.5), (9.5, 6.5), (14.5, 6.5), (14.5, 9.5), (9.5, 9.5),
        (9.5, 14.5), (6.5, 14.5), (6.5, 9.5), (1.5, 9.5), (1.5, 6.5), (6.5, 6.5),
    ))


def check_icon(colour: str) -> QIcon:
    """A check mark that occupies a fixed column beside list text."""

    pixmap, painter = _icon_pixmap(CHECK_ICON_SIZE)
    pen = QPen(QColor(colour), 1.7)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.drawPolyline([QPointF(2.0, 6.5), QPointF(4.75, 9.25), QPointF(10.0, 2.75)])
    painter.end()
    return QIcon(pixmap)


def blank_icon(size: QSize) -> QIcon:
    """An empty icon, so a row without a mark lines up with marked rows."""

    pixmap = QPixmap(size)
    pixmap.fill(Qt.GlobalColor.transparent)
    return QIcon(pixmap)


def _canvas_button_rules(selector: str) -> str:
    """The colours of a canvas button in each of its states."""

    return f"""
        {selector} {{
            color: {CANVAS_BUTTON_TEXT};
            background-color: {CANVAS_BUTTON_BACKGROUND};
            border: 1px solid #647383;
            border-radius: 3px;
            padding: 2px 7px;
        }}
        {selector}:hover {{
            background-color: #435465;
            border-color: #8295a8;
        }}
        {selector}:pressed {{
            background-color: #202a33;
            border-color: #718395;
        }}
        {selector}:checked {{
            color: #ffffff;
            background-color: #245f75;
            border-color: #57d8ff;
        }}
        {selector}:checked:hover {{
            background-color: #2c7189;
            border-color: #83e4ff;
        }}
        {selector}:disabled {{
            color: {CANVAS_DISABLED_BUTTON_TEXT};
            background-color: {CANVAS_DISABLED_BUTTON_BACKGROUND};
            border-color: #4b5865;
        }}
        """


def style_canvas_icon_button(button: QToolButton) -> None:
    """Make a button outside a canvas bar match the compact icon buttons inside one."""

    button.setIconSize(COMPACT_ICON_SIZE)
    button.setStyleSheet(_canvas_button_rules("QToolButton") + "QToolButton { padding: 2px 3px; }")


def style_canvas_control_bar(frame: QFrame, object_name: str) -> None:
    """Give canvas controls explicit, readable enabled and disabled colours."""

    frame.setObjectName(object_name)
    frame.setStyleSheet(
        f"""
        QFrame#{object_name} {{
            background-color: {CANVAS_CONTROL_BACKGROUND};
            border-bottom: 1px solid #3d4853;
        }}
        {_canvas_button_rules(f"QFrame#{object_name} QToolButton")}
        QFrame#{object_name} QLabel {{
            color: #d7e0e8;
            background-color: transparent;
            border: 0;
        }}
        """
    )

    # Keep the same contrast if a platform style consults the palette for a
    # sub-control instead of the stylesheet (notably disabled tool buttons).
    palette = frame.palette()
    for group in (QPalette.ColorGroup.Active, QPalette.ColorGroup.Inactive):
        palette.setColor(
            group,
            QPalette.ColorRole.Button,
            QColor(CANVAS_BUTTON_BACKGROUND),
        )
        palette.setColor(
            group,
            QPalette.ColorRole.ButtonText,
            QColor(CANVAS_BUTTON_TEXT),
        )
        palette.setColor(group, QPalette.ColorRole.WindowText, QColor("#d7e0e8"))
    palette.setColor(
        QPalette.ColorGroup.Disabled,
        QPalette.ColorRole.Button,
        QColor(CANVAS_DISABLED_BUTTON_BACKGROUND),
    )
    palette.setColor(
        QPalette.ColorGroup.Disabled,
        QPalette.ColorRole.ButtonText,
        QColor(CANVAS_DISABLED_BUTTON_TEXT),
    )
    palette.setColor(
        QPalette.ColorGroup.Disabled,
        QPalette.ColorRole.WindowText,
        QColor(CANVAS_DISABLED_BUTTON_TEXT),
    )
    frame.setPalette(palette)
    for button in frame.findChildren(QToolButton):
        button.setPalette(palette)
    for label in frame.findChildren(QLabel):
        label.setPalette(palette)
