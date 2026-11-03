"""The overlay selector and opacity slider shown in the image display's header."""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QSizePolicy, QWidget

from seedvision.ui.canvas_controls import style_canvas_control_bar


GROUP_GAP = 8
COMPACT_SELECTOR_WIDTH = 120
NARROW_SELECTOR_WIDTH = 72
HEADER_STYLE = """
    QFrame#overlayHeaderControls {
        border: 0;
    }
    QFrame#overlayHeaderControls QToolButton {
        padding-right: 16px;
    }
    QSlider::groove:horizontal {
        height: 4px;
        background: #647383;
        border-radius: 2px;
    }
    QSlider::groove:horizontal:disabled {
        background: #4b5865;
    }
    QSlider::handle:horizontal {
        width: 12px;
        margin: -5px 0;
        background: #f2f6fa;
        border: 1px solid #8295a8;
        border-radius: 6px;
    }
    QSlider::handle:horizontal:disabled {
        background: #aeb9c4;
        border-color: #4b5865;
    }
"""


class OverlayHeaderControls(QFrame):
    """Overlay selector and opacity, on one row of a fixed-height header.

    The host can give the row less than its preferred width. It then gives up width
    in the order of ``_ARRANGEMENTS``: the slider shortens first, because the row
    shares any shortfall out to it, then the opacity heading, the selector's text,
    the headings, the percentage and finally the slider. Tooltips still say what
    each control does once its heading has gone.
    """

    width_preference_changed = Signal()

    # The optional controls shown, and the narrowest the selector may be before the
    # next arrangement is tried (None: it must show the whole title).
    _ARRANGEMENTS = (
        (("selector_heading", "opacity_heading", "slider", "value_label"), None),
        (("selector_heading", "short_opacity_heading", "slider", "value_label"), COMPACT_SELECTOR_WIDTH),
        (("short_opacity_heading", "slider", "value_label"), COMPACT_SELECTOR_WIDTH),
        (("slider", "value_label"), COMPACT_SELECTOR_WIDTH),
        (("slider",), NARROW_SELECTOR_WIDTH),
        ((), NARROW_SELECTOR_WIDTH),
    )

    def __init__(self, selector, slider, value_label, parent=None) -> None:
        super().__init__(parent)
        self.selector = selector
        self.slider = slider
        self.value_label = value_label
        self._title = ""
        self._opacity_visible = True
        self._offered_width = None
        self._fitting = False
        self.selector_heading = QLabel("Overlay:", self)
        self.opacity_controls = QWidget(self)
        self.opacity_heading = QLabel("Overlay opacity:", self.opacity_controls)
        self.short_opacity_heading = QLabel("Opacity:", self.opacity_controls)
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(3)
        row.addWidget(self.selector_heading)
        row.addWidget(selector)
        row.addWidget(self.opacity_controls)
        opacity = QHBoxLayout(self.opacity_controls)
        opacity.setContentsMargins(GROUP_GAP, 0, 0, 0)
        opacity.setSpacing(row.spacing())
        opacity.addWidget(self.opacity_heading)
        opacity.addWidget(self.short_opacity_heading)
        opacity.addWidget(slider, 1)
        opacity.addWidget(value_label)
        style_canvas_control_bar(self, "overlayHeaderControls")
        self.setStyleSheet(self.styleSheet() + HEADER_STYLE)
        self._fit()

    def set_title(self, title: str, hint: str) -> None:
        """Show the selected overlay; the selector elides it only when it must."""

        self._title = title
        self.selector.setAccessibleName(title)
        self.selector.setToolTip(f"{title}\n{hint}")
        self._preference_changed()

    def set_opacity_visible(self, visible: bool) -> None:
        """Offer opacity only where the selected overlay can be blended."""

        self._opacity_visible = visible
        self._preference_changed()

    def offer_width(self, width: int | None) -> None:
        """Prefer the width a host can spare; None is no limit beyond the content."""

        if width != self._offered_width:
            self._offered_width = width
            self.updateGeometry()

    def preferred_width(self) -> int:
        """The width at which every control shows in full."""

        shown, _ = self._ARRANGEMENTS[0]
        return self._rest_width(shown, self.slider.maximumWidth()) + self._selector_width()

    def comfortable_width(self) -> int:
        """The width that still holds a usable selector and the slider."""

        return self._rest_width(("slider",), self.slider.minimumWidth()) + min(
            self._selector_width(), COMPACT_SELECTOR_WIDTH
        )

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt override
        width = self.preferred_width()
        if self._offered_width is not None:
            width = min(width, self._offered_width)
        return QSize(width, self.selector.sizeHint().height())

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt override
        return QSize(
            min(self._selector_width(), NARROW_SELECTOR_WIDTH),
            self.selector.sizeHint().height(),
        )

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._fit()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().showEvent(event)
        self._fit()

    def _preference_changed(self) -> None:
        self.updateGeometry()
        self.width_preference_changed.emit()
        self._fit()

    def _selector_chrome(self) -> int:
        """What the selector needs beyond its text: padding, border and menu arrow."""

        metrics = self.selector.fontMetrics()
        return self.selector.sizeHint().width() - metrics.horizontalAdvance(self.selector.text())

    def _selector_width(self) -> int:
        """The width the whole title needs, up to the selector's own limit."""

        wanted = self.selector.fontMetrics().horizontalAdvance(self._title) + self._selector_chrome()
        return min(wanted, self.selector.maximumWidth())

    def _rest_width(self, shown: tuple[str, ...], slider_width: int) -> int:
        """The row's width without the selector, showing only the named controls."""

        spacing = self.layout().spacing()
        width = self.selector_heading.sizeHint().width() + spacing if "selector_heading" in shown else 0
        opacity = [name for name in shown if name != "selector_heading"] if self._opacity_visible else []
        if opacity:
            width += spacing + GROUP_GAP + spacing * (len(opacity) - 1)
        for name in opacity:
            control = getattr(self, name)
            width += slider_width if control is self.slider else max(
                control.sizeHint().width(), control.minimumWidth()
            )
        return width

    def _fit(self) -> None:
        """Arrange the row for its width, and again if arranging it changed that width."""

        if self._fitting:
            return
        self._fitting = True
        try:
            arranged = None
            # Showing a control activates the host's layout, which can resize this row.
            # A hidden row has no meaningful width yet; showing it fits it again.
            while arranged != (width := self.width() if self.isVisible() else self.sizeHint().width()):
                arranged = width
                self._arrange(width)
        finally:
            self._fitting = False

    def _arrange(self, available: int) -> None:
        wanted = self._selector_width()
        for shown, narrowest in self._ARRANGEMENTS:
            rest = self._rest_width(shown, self.slider.minimumWidth())
            if rest + (wanted if narrowest is None else min(wanted, narrowest)) <= available:
                break
        self.selector_heading.setVisible("selector_heading" in shown)
        for name in ("opacity_heading", "short_opacity_heading", "slider", "value_label"):
            getattr(self, name).setVisible(name in shown)
        self.opacity_controls.setVisible(self._opacity_visible and "slider" in shown)
        width = max(min(wanted, available - rest), NARROW_SELECTOR_WIDTH)
        text = self.selector.fontMetrics().elidedText(
            self._title, Qt.TextElideMode.ElideRight, width - self._selector_chrome()
        )
        if self.selector.text() != text:
            self.selector.setText(text)
