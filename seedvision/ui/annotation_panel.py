"""Resizable in-view annotation tool window (no separate application window)."""

from PySide6.QtCore import QPoint, QSize, Qt
from PySide6.QtWidgets import QFrame, QLabel, QStackedWidget


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


class AnnotationPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.user_size = None
        self.setMinimumSize(300, 180)
        self.resize_grip = _ResizeGrip(self)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.resize_grip.move(self.width() - 18, self.height() - 18)
        self.resize_grip.raise_()


class _ResizeGrip(QLabel):
    def __init__(self, panel):
        super().__init__("◢", panel)
        self.setFixedSize(18, 18)
        self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        self.setToolTip("Drag to resize annotation tools")
        self._origin = None

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
        panel.user_size = size
        panel.resize(size)
        event.accept()

    def mouseReleaseEvent(self, event):
        self._origin = None
        event.accept()
