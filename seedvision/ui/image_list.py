"""Images list rows: background thumbnails, analysis status, species and the open image.

A row asks for its thumbnail when it is painted, so hidden rows cost nothing.
Thumbnails decode on their own low-priority thread: the analysis pool must not
see them, because its activity count means "analysis is busy" to the rest of the
window. They live only in a small in-memory cache and never reach a project file.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import os

from PySide6.QtCore import (
    QEvent, QObject, QPoint, QRect, QRunnable, QSize, Qt, QThread, QThreadPool, Signal,
)
from PySide6.QtGui import (
    QColor, QFontMetrics, QIcon, QImage, QImageIOHandler, QImageReader, QPainter,
    QPalette, QPen,
)
from PySide6.QtWidgets import QStyle, QStyledItemDelegate, QToolTip

THUMBNAIL_SIZE = QSize(48, 32)
# True on the row whose image is in the viewer, which is the project's selected
# image. Plain list selection only moves the cursor until a row is activated.
IMAGE_OPEN_ROLE = Qt.ItemDataRole.UserRole + 1
# The display name of the species recorded for the row's image.
IMAGE_SPECIES_ROLE = Qt.ItemDataRole.UserRole + 2

_ROW_PADDING = 3
_TILE_LEFT = 3
_TEXT_GAP = 5
_REMOVE_WIDTH = 20
_BADGE_SIZE = 16
_NAME_BREAKS = "._- "
# A species column needs room for the longest species beside an ordinary file name.
_SPECIES_COLUMN_SAMPLE = "Lupinus polyphyllus"
_NAME_COLUMN_SAMPLE = "IMG_0000c.JPG"
_COLUMN_GAP = 8


def read_thumbnail(path: str, box: QSize) -> tuple[QImage, str]:
    """Decode one image to fit ``box``, or return a null image and the reason.

    Scaling is requested from the reader, not applied afterwards, because the
    JPEG decoder then skips most of a full-resolution decode.
    """

    reader = QImageReader(path)
    reader.setAutoTransform(True)
    source = reader.size()
    if not source.isEmpty():
        # The reader scales before it applies the EXIF rotation.
        rotated = reader.transformation() & QImageIOHandler.Transformation.TransformationRotate90
        fitted = source.scaled(
            box.transposed() if rotated else box, Qt.AspectRatioMode.KeepAspectRatio
        )
        reader.setScaledSize(fitted.boundedTo(source).expandedTo(QSize(1, 1)))
    image = reader.read()
    if image.isNull():
        return QImage(), reader.errorString() or "Qt could not decode the image."
    if image.width() > box.width() or image.height() > box.height():
        image = image.scaled(
            box, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
    return image.convertToFormat(
        QImage.Format.Format_ARGB32_Premultiplied if image.hasAlphaChannel()
        else QImage.Format.Format_RGB32
    ), ""


def _file_signature(path: str) -> tuple[int, int] | None:
    try:
        status = os.stat(path)
    except OSError:
        return None
    return status.st_mtime_ns, status.st_size


@dataclass(frozen=True, slots=True)
class Thumbnail:
    """One decoded result; a null image means ``error`` says why there is none."""

    image: QImage
    error: str
    signature: tuple[int, int] | None


class _ThumbnailSignals(QObject):
    finished = Signal(str, int, object)


class _ThumbnailJob(QRunnable):
    def __init__(self, path, request, box, scale, wanted, signals) -> None:
        super().__init__()
        self.path = path
        self.request = request
        self.box = box
        self.scale = scale
        self.wanted = wanted
        self.signals = signals

    def run(self) -> None:
        # A row removed or a project replaced while this waited needs no decode.
        if self.wanted.get(self.path) != self.request:
            return
        signature = _file_signature(self.path)
        image, error = read_thumbnail(self.path, self.box)
        image.setDevicePixelRatio(self.scale)
        try:
            self.signals.finished.emit(
                self.path, self.request, Thumbnail(image, error, signature)
            )
        except RuntimeError:
            pass  # The owning window was destroyed during this decode.


class ImageThumbnailLoader(QObject):
    """Decode thumbnails off the GUI thread and keep the most recently painted."""

    changed = Signal()

    # About 6 kB each at 100% display scale, so a very large project stays small.
    CACHE_LIMIT = 512
    STOP_WAIT_MS = 2000

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._pool = QThreadPool(self)
        # A decode takes about 50 ms, so one thread keeps up and cannot compete
        # with an image being opened for the disk.
        self._pool.setMaxThreadCount(1)
        self._pool.setThreadPriority(QThread.Priority.LowPriority)
        self._entries: OrderedDict[str, Thumbnail] = OrderedDict()
        self._wanted: dict[str, int] = {}
        self._next_request = 0
        self._signals = _ThumbnailSignals()
        self._signals.finished.connect(self._finished)

    def get(self, path: str, scale: float = 1.0) -> Thumbnail | None:
        """Return the cached result, or None after queuing a decode for it."""

        entry = self._entries.get(path)
        if entry is not None:
            self._entries.move_to_end(path)
            return entry
        if path not in self._wanted:
            self._next_request += 1
            self._wanted[path] = self._next_request
            # Newest first: after a long scroll, the rows now on screen go ahead
            # of those scrolled past.
            self._pool.start(
                _ThumbnailJob(
                    path, self._next_request, THUMBNAIL_SIZE * scale, scale,
                    self._wanted, self._signals,
                ),
                self._next_request,
            )
        return None

    def error(self, path: str) -> str:
        entry = self._entries.get(path)
        return "" if entry is None else entry.error

    def refresh(self, path: str) -> None:
        """Drop a thumbnail whose source file has changed since it was decoded."""

        entry = self._entries.get(path)
        if entry is not None and entry.signature != _file_signature(path):
            self.forget(path)
            self.changed.emit()

    def forget(self, path: str) -> None:
        self._entries.pop(path, None)
        self._wanted.pop(path, None)

    def clear(self) -> None:
        self._wanted.clear()
        self._entries.clear()
        self._pool.clear()

    def stop(self) -> None:
        """Abandon queued work and let the decode in flight close its file."""

        self.clear()
        self._pool.waitForDone(self.STOP_WAIT_MS)

    def _finished(self, path: str, request: int, thumbnail: Thumbnail) -> None:
        if self._wanted.get(path) != request:
            return
        del self._wanted[path]
        self._entries[path] = thumbnail
        while len(self._entries) > self.CACHE_LIMIT:
            self._entries.popitem(last=False)
        self.changed.emit()


def name_lines(name: str, metrics: QFontMetrics, width: int) -> tuple[str, ...]:
    """Fit a file name on two lines at most, keeping the end that tells names apart."""

    if metrics.horizontalAdvance(name) <= width:
        return (name,)
    count = len(name) - 1
    while count > 1 and metrics.horizontalAdvance(name[:count]) > width:
        count -= 1
    for candidate in range(count, count // 2, -1):
        if name[candidate - 1] in _NAME_BREAKS:
            count = candidate
            break
    return name[:count], metrics.elidedText(name[count:], Qt.TextElideMode.ElideMiddle, width)


def species_text(species: str, metrics: QFontMetrics, width: int) -> str:
    """Fit a species name, abbreviating its genus first.

    Truncating alone would leave "Lupinus m…" for two different species.
    """

    if metrics.horizontalAdvance(species) <= width:
        return species
    genus, _, epithet = species.partition(" ")
    if epithet:
        species = f"{genus[0]}. {epithet}"
    return metrics.elidedText(species, Qt.TextElideMode.ElideRight, width)


@dataclass(frozen=True, slots=True)
class RowText:
    """Where one row's name and species go; no species means ``species_rect`` is None."""

    name_rect: QRect
    name: tuple[str, ...]
    species_rect: QRect | None = None
    species: str = ""


def row_text(area: QRect, name: str, species: str | None, metrics: QFontMetrics) -> RowText:
    """Give the species its own column in a wide row, or a second line in a narrow one."""

    if not species:
        return RowText(area, name_lines(name, metrics, area.width()))
    column = metrics.horizontalAdvance(_SPECIES_COLUMN_SAMPLE)
    needed = column + _COLUMN_GAP + metrics.horizontalAdvance(_NAME_COLUMN_SAMPLE)
    if area.width() >= needed:
        name_rect = area.adjusted(0, 0, -(column + _COLUMN_GAP), 0)
        species_rect = QRect(area.right() - column + 1, area.top(), column, area.height())
        return RowText(name_rect, name_lines(name, metrics, name_rect.width()),
                       species_rect, species_text(species, metrics, column))
    line = metrics.lineSpacing()
    name_rect = QRect(area.left(), area.center().y() - line + 1, area.width(), line)
    return RowText(
        name_rect, (metrics.elidedText(name, Qt.TextElideMode.ElideMiddle, area.width()),),
        name_rect.translated(0, line), species_text(species, metrics, area.width()),
    )


class ImageListDelegate(QStyledItemDelegate):
    """Paint a thumbnail, status badge, name, species and removal target on each row."""

    remove_requested = Signal(object)

    def __init__(self, thumbnails: ImageThumbnailLoader, view) -> None:
        super().__init__(view)
        self._thumbnails = thumbnails
        self._view = view
        self._show_species = True
        thumbnails.changed.connect(view.viewport().update)

    def set_show_species(self, visible: bool) -> None:
        self._show_species = bool(visible)
        self._view.viewport().update()

    @staticmethod
    def remove_rect(rect: QRect) -> QRect:
        target = QRect(0, 0, 18, 22)
        target.moveCenter(QPoint(rect.right() - 9, rect.center().y()))
        return target

    @staticmethod
    def tile_rect(rect: QRect) -> QRect:
        tile = QRect(QPoint(0, 0), THUMBNAIL_SIZE)
        tile.moveCenter(QPoint(rect.left() + _TILE_LEFT + tile.width() // 2, rect.center().y()))
        return tile

    @staticmethod
    def text_rect(rect: QRect) -> QRect:
        left = rect.left() + _TILE_LEFT + THUMBNAIL_SIZE.width() + _TEXT_GAP
        return QRect(left, rect.top(), max(rect.right() - _REMOVE_WIDTH - left, 1), rect.height())

    def sizeHint(self, option, index) -> QSize:  # noqa: N802 - Qt override
        height = max(
            THUMBNAIL_SIZE.height() + 2 * _ROW_PADDING,
            2 * option.fontMetrics.lineSpacing() + 4,
        )
        return QSize(_TILE_LEFT + THUMBNAIL_SIZE.width() + _TEXT_GAP + _REMOVE_WIDTH, height)

    def paint(self, painter, option, index) -> None:
        painter.save()
        group = QPalette.ColorGroup.Active
        accent = option.palette.color(group, QPalette.ColorRole.Highlight)
        text = option.palette.color(group, QPalette.ColorRole.Text)
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)
        # The view has already painted the alternating base. Active colours keep
        # the open row just as clear when another panel has keyboard focus.
        if index.data(IMAGE_OPEN_ROLE):
            painter.fillRect(option.rect, accent)
            text = option.palette.color(group, QPalette.ColorRole.HighlightedText)
        elif selected or hovered:
            tint = QColor(accent)
            tint.setAlphaF(.24 if selected else .12)
            painter.fillRect(option.rect, tint)
            if selected:
                painter.setPen(QPen(accent, 1))
                painter.drawRect(option.rect.adjusted(0, 0, -1, -1))
        self._paint_tile(painter, option, index)
        layout = row_text(
            self.text_rect(option.rect),
            index.data(Qt.ItemDataRole.DisplayRole) or "",
            index.data(IMAGE_SPECIES_ROLE) if self._show_species else None,
            option.fontMetrics,
        )
        alignment = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        painter.setFont(option.font)
        painter.setPen(text)
        painter.drawText(layout.name_rect, alignment, "\n".join(layout.name))
        if layout.species_rect is not None:
            secondary = QColor(text)
            secondary.setAlphaF(.72)
            painter.setPen(secondary)
            painter.drawText(layout.species_rect, alignment, layout.species)
        # Draw the removal target only where the pointer or selection is, so a
        # long list does not show a column of identical close buttons.
        if selected or hovered:
            # Drawn in the row's text colour, which stays legible on the highlight.
            cross = QRect(0, 0, 9, 9)
            cross.moveCenter(self.remove_rect(option.rect).center())
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(QPen(text, 1.5))
            painter.drawLine(cross.topLeft(), cross.bottomRight())
            painter.drawLine(cross.bottomLeft(), cross.topRight())
        painter.restore()

    def _paint_tile(self, painter, option, index) -> None:
        tile = self.tile_rect(option.rect)
        path = index.data(Qt.ItemDataRole.UserRole)
        thumbnail = None if path is None else self._thumbnails.get(
            path, option.widget.devicePixelRatioF()
        )
        painter.fillRect(tile, option.palette.color(QPalette.ColorRole.Midlight))
        painter.setPen(QPen(option.palette.color(QPalette.ColorRole.Mid), 1))
        if thumbnail is not None and not thumbnail.image.isNull():
            size = thumbnail.image.deviceIndependentSize().toSize()
            painter.drawImage(
                QPoint(tile.left() + (tile.width() - size.width()) // 2,
                       tile.top() + (tile.height() - size.height()) // 2),
                thumbnail.image,
            )
        elif thumbnail is not None:
            # An unreadable file keeps its row; a crossed tile tells it apart
            # from one that is still loading, and the tooltip gives the reason.
            painter.drawLine(tile.topLeft(), tile.bottomRight())
            painter.drawLine(tile.bottomLeft(), tile.topRight())
        painter.drawRect(tile.adjusted(0, 0, -1, -1))
        status = index.data(Qt.ItemDataRole.DecorationRole)
        if isinstance(status, QIcon) and not status.isNull():
            # The analysis status sits on the thumbnail, with a plate so that it
            # stays legible on any photograph.
            badge = QRect(0, 0, _BADGE_SIZE, _BADGE_SIZE)
            badge.moveBottomRight(tile.bottomRight() + QPoint(2, 2))
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(option.palette.color(QPalette.ColorRole.Base))
            painter.drawEllipse(badge.adjusted(-1, -1, 1, 1))
            status.paint(painter, badge, Qt.AlignmentFlag.AlignCenter)

    def editorEvent(self, event, model, option, index) -> bool:  # noqa: N802 - Qt override
        if (
            event.type() in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonRelease,
                             QEvent.Type.MouseButtonDblClick)
            and event.button() == Qt.MouseButton.LeftButton
            and self.remove_rect(option.rect).contains(event.position().toPoint())
        ):
            if event.type() == QEvent.Type.MouseButtonRelease:
                self.remove_requested.emit(index.data(Qt.ItemDataRole.UserRole))
            return True
        return super().editorEvent(event, model, option, index)

    def helpEvent(self, event, view, option, index) -> bool:  # noqa: N802 - Qt override
        if event.type() != QEvent.Type.ToolTip:
            return super().helpEvent(event, view, option, index)
        # A narrow row may abbreviate the species, so the tooltip always names it in full.
        lines = [index.data(Qt.ItemDataRole.ToolTipRole) or ""]
        species = index.data(IMAGE_SPECIES_ROLE)
        if self._show_species and species:
            lines.append(f"Species: {species}")
        error = self._thumbnails.error(index.data(Qt.ItemDataRole.UserRole) or "")
        if error:
            lines.append(f"No thumbnail: {error}")
        QToolTip.showText(event.globalPos(), "\n".join(lines), view)
        return True
