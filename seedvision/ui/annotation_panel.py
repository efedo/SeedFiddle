"""Floating annotation palette owned by the application window."""

from PySide6.QtCore import QEvent, QPointF, QSize, Qt, Signal
from PySide6.QtGui import QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractButton,
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from seedvision.ui.deferred import run_soon

# QWIDGETSIZE_MAX: the largest size a widget may be given.
_UNBOUNDED = (1 << 24) - 1


class CurrentPageStack(QStackedWidget):
    """Hidden assisted-fill pages must not reserve space on the brush page."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.currentChanged.connect(lambda _index: self.updateGeometry())
        self.currentChanged.connect(self._fit_current_page)

    def _fit_current_page(self, *_unused):
        page = self.currentWidget()
        if page is not None:
            height = page.heightForWidth(self.width()) if page.hasHeightForWidth() else page.sizeHint().height()
            self.setFixedHeight(max(page.minimumSizeHint().height(), height, 1))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_current_page()

    def sizeHint(self):
        page = self.currentWidget()
        return page.sizeHint() if page is not None else QSize(0, 0)

    def minimumSizeHint(self):
        page = self.currentWidget()
        return page.minimumSizeHint() if page is not None else QSize(0, 0)

    def hasHeightForWidth(self):
        return self.currentWidget() is not None and self.currentWidget().hasHeightForWidth()

    def heightForWidth(self, width):
        page = self.currentWidget()
        return page.heightForWidth(width) if page is not None else 0


def even_buttons(*buttons: QAbstractButton) -> None:
    """Give buttons sharing equally stretched grid columns one width.

    Their size hints are ignored, so a longer label does not widen its column,
    and their common minimum fits the longest label, so none is clipped.
    """

    widest = 0
    for button in buttons:
        # Apply any style-sheet font (a bold primary button) before measuring.
        button.ensurePolished()
        style = button.style()
        padding = 2 * (
            style.pixelMetric(QStyle.PixelMetric.PM_ButtonMargin, None, button)
            + style.pixelMetric(QStyle.PixelMetric.PM_DefaultFrameWidth, None, button)
        )
        widest = max(
            widest, button.fontMetrics().horizontalAdvance(button.text()) + padding + 2
        )
    for button in buttons:
        button.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        button.setMinimumWidth(widest)


class AnnotationPanel(QFrame):
    """A floating palette: a title bar that drags it, over contents that scroll.

    The panel is never taller than its contents, so a short editor leaves no
    empty space below its controls, and never narrower than they need. Its
    owner places it; ``contents_resized`` asks the owner to place it again.
    """

    contents_resized = Signal()
    close_requested = Signal()

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.user_size = None
        self._collapsed = False
        self._fitted = None
        self._fit_scheduled = False
        self.setObjectName("annotationPanel")
        self.setStyleSheet(
            "QFrame#annotationPanel { background: palette(window); "
            "border: 1px solid palette(dark); border-radius: 4px; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.title_bar = _TitleBar(title, self)
        self.title_bar.collapse_button.clicked.connect(
            lambda: self.set_collapsed(not self._collapsed)
        )
        self.title_bar.close_button.clicked.connect(self.close_requested)
        layout.addWidget(self.title_bar)
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        layout.addWidget(self.scroll_area, 1)
        self.resize_grip = _ResizeGrip(self)
        self.setMinimumSize(300, self.title_bar.sizeHint().height())

    def set_title(self, title: str) -> None:
        self.title_bar.title_label.setText(title)

    def title(self) -> str:
        return self.title_bar.title_label.text()

    def add_title_button(self, text: str) -> QToolButton:
        """Add a flat command button to the title bar, before collapse and close."""

        return self.title_bar.add_button(text)

    def set_contents(self, widget: QWidget) -> None:
        self.scroll_area.setWidget(widget)
        widget.installEventFilter(self)
        self.fit_to_contents()

    def is_collapsed(self) -> bool:
        return self._collapsed

    def set_collapsed(self, collapsed: bool) -> None:
        """Roll the panel up to its title bar, or back down."""

        collapsed = bool(collapsed)
        if collapsed == self._collapsed:
            return
        self._collapsed = collapsed
        self.scroll_area.setVisible(not collapsed)
        self.resize_grip.setVisible(not collapsed)
        self.title_bar.set_collapsed(collapsed)
        self.fit_to_contents()

    def eventFilter(self, watched, event):  # noqa: N802 - Qt override
        # Every size change inside the contents ends in a layout request here.
        # Fitting now, before the request is handled, resizes the panel before
        # it next paints.
        if (
            event.type() == QEvent.Type.LayoutRequest
            and watched is self.scroll_area.widget()
        ):
            self.fit_to_contents()
        return super().eventFilter(watched, event)

    def fit_to_contents(self) -> None:
        """Bound the panel's size by its contents and ask to be placed again."""

        frame = self.size() - self.contentsRect().size()
        title_height = self.title_bar.sizeHint().height()
        contents = self.scroll_area.widget()
        if self._collapsed or contents is None:
            natural = floor = frame.height() + title_height
            minimum_width = 300
        else:
            margin = 2 * self.scroll_area.frameWidth()
            width = self.contentsRect().width() - margin
            height = (
                contents.heightForWidth(width)
                if contents.hasHeightForWidth()
                else -1
            )
            if height < 0:
                height = contents.sizeHint().height()
            height = max(height, contents.minimumSizeHint().height()) + margin
            natural = frame.height() + title_height + height
            # Keep a few rows in view when the user shrinks the panel.
            floor = frame.height() + title_height + min(height, 96)
            scroll_bar = self.scroll_area.verticalScrollBar().sizeHint().width()
            minimum_width = max(
                300,
                self.title_bar.minimumSizeHint().width(),
                contents.minimumSizeHint().width() + margin + scroll_bar,
            ) + frame.width()
        fitted = (natural, floor, minimum_width)
        if fitted == self._fitted:
            return
        self._fitted = fitted
        self.setMinimumHeight(0)
        self.setMaximumHeight(natural)
        self.setMinimumHeight(floor)
        self.setMinimumWidth(minimum_width)
        self.contents_resized.emit()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._place_grip()
        if event.oldSize().width() != event.size().width():
            # The contents wrap differently at a new width. Refit after this
            # event: resizing during it, while the panel is being shown, would
            # never deliver the second resize event.
            self._schedule_fit()

    def showEvent(self, event):
        super().showEvent(event)
        self._place_grip()

    def _place_grip(self) -> None:
        self.resize_grip.move(
            self.width() - self.resize_grip.width() - 1,
            self.height() - self.resize_grip.height() - 1,
        )
        self.resize_grip.raise_()

    def _schedule_fit(self) -> None:
        if not self._fit_scheduled:
            self._fit_scheduled = True
            run_soon(self, self._scheduled_fit)

    def _scheduled_fit(self) -> None:
        self._fit_scheduled = False
        self.fit_to_contents()


class _TitleBar(QFrame):
    """The panel's title strip: dragging it moves the panel."""

    def __init__(self, title: str, panel: AnnotationPanel):
        super().__init__(panel)
        self.setObjectName("annotationPanelTitleBar")
        self._collapsed = False
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 2, 3, 2)
        layout.setSpacing(2)
        self.title_label = QLabel(title, self)
        font = self.title_label.font()
        font.setBold(True)
        self.title_label.setFont(font)
        # Presses on the title reach the bar, which drags the panel.
        self.title_label.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents
        )
        layout.addWidget(self.title_label, 1)
        self.collapse_button = self._tool_button()
        self.close_button = self._tool_button()
        self.close_button.setToolTip(
            "Finish annotating, as turning off Materials or Seeds on the toolbar "
            "does. Unapplied edits stay as a draft."
        )
        for button in (self.collapse_button, self.close_button):
            button.setFixedSize(22, 22)
            button.setIconSize(QSize(_ICON_SIZE, _ICON_SIZE))
            layout.addWidget(button)
        self.set_collapsed(False)

    def _tool_button(self) -> QToolButton:
        button = QToolButton(self)
        button.setAutoRaise(True)
        # The bar shows a hand to drag with; its buttons are pressed normally.
        button.setCursor(Qt.CursorShape.ArrowCursor)
        return button

    def add_button(self, text: str) -> QToolButton:
        button = self._tool_button()
        button.setText(text)
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        button.setFixedHeight(22)
        layout = self.layout()
        layout.insertWidget(layout.indexOf(self.collapse_button), button)
        return button

    def changeEvent(self, event):  # noqa: N802 - Qt override
        super().changeEvent(event)
        if event.type() == QEvent.Type.PaletteChange and hasattr(self, "close_button"):
            # Redraw the icons in the new text colour (a light or dark theme).
            # Only the icons: a style sheet set here would change the palette
            # again.
            self._draw_icons()

    def _draw_icons(self) -> None:
        self.close_button.setIcon(_line_icon(self, "close"))
        self.collapse_button.setIcon(
            _line_icon(self, "expand" if self._collapsed else "collapse")
        )

    def set_collapsed(self, collapsed: bool) -> None:
        self._collapsed = collapsed
        self._draw_icons()
        self.collapse_button.setToolTip(
            "Show the annotation tools again."
            if collapsed
            else "Roll the annotation tools up to this title bar."
        )
        self._set_rule(not collapsed)

    def _set_rule(self, visible: bool) -> None:
        # Rolled up, the bar is the whole panel, so it draws no rule beneath.
        rule = "1px solid palette(mid)" if visible else "none"
        radius = "0px" if visible else "3px"
        self.setStyleSheet(
            "QFrame#annotationPanelTitleBar { background: palette(midlight); "
            f"border: none; border-bottom: {rule}; "
            "border-top-left-radius: 3px; border-top-right-radius: 3px; "
            f"border-bottom-left-radius: {radius}; "
            f"border-bottom-right-radius: {radius}; }}"
        )


_ICON_SIZE = 12


def _line_icon(widget: QWidget, kind: str) -> QIcon:
    """A small line-drawn title-bar icon in the widget's text colour.

    ``kind`` is ``close`` (a cross), ``collapse`` (a chevron pointing up) or
    ``expand`` (a chevron pointing down). Disabled states are derived by QIcon.
    """

    scale = 3
    pixmap = QPixmap(_ICON_SIZE * scale, _ICON_SIZE * scale)
    pixmap.setDevicePixelRatio(scale)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(widget.palette().windowText().color(), 1.5)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    low, high, middle = 2.5, _ICON_SIZE - 2.5, _ICON_SIZE / 2
    if kind == "close":
        painter.drawLine(QPointF(low, low), QPointF(high, high))
        painter.drawLine(QPointF(high, low), QPointF(low, high))
    else:
        tip, base = (4.0, 8.0) if kind == "collapse" else (8.0, 4.0)
        painter.drawPolyline(
            [QPointF(low, base), QPointF(middle, tip), QPointF(high, base)]
        )
    painter.end()
    return QIcon(pixmap)


class _ResizeGrip(QWidget):
    """Corner handle that resizes the panel; the panel remembers the size."""

    def __init__(self, panel):
        super().__init__(panel)
        self.setFixedSize(14, 14)
        self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        self.setToolTip("Drag to resize annotation tools")
        self._origin = None

    def paintEvent(self, event):  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(self.palette().dark().color(), 1.2)
        painter.setPen(pen)
        right, bottom = self.width() - 2.5, self.height() - 2.5
        for offset in (3.0, 7.0, 11.0):
            painter.drawLine(
                QPointF(right - offset, bottom), QPointF(right, bottom - offset)
            )

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._origin = event.globalPosition().toPoint()
            self._size = self.parentWidget().size()
            event.accept()

    def mouseMoveEvent(self, event):
        if self._origin is None:
            return
        panel = self.parentWidget()
        delta = event.globalPosition().toPoint() - self._origin
        available = panel.parentWidget().size() - QSize(panel.x(), panel.y())
        size = (self._size + QSize(delta.x(), delta.y())).expandedTo(
            panel.minimumSize()).boundedTo(available)
        panel.resize(size)
        # A panel pulled to its full height keeps following its contents.
        height = (
            _UNBOUNDED
            if size.height() >= panel.maximumHeight()
            else panel.height()
        )
        panel.user_size = QSize(panel.width(), height)
        event.accept()

    def mouseReleaseEvent(self, event):
        self._origin = None
        event.accept()
