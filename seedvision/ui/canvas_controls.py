"""Shared styling for controls drawn over the dark image and graph canvases."""

from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QFrame, QLabel, QToolButton


CANVAS_CONTROL_BACKGROUND = "#252d35"
CANVAS_BUTTON_BACKGROUND = "#364452"
CANVAS_BUTTON_TEXT = "#f2f6fa"
CANVAS_DISABLED_BUTTON_BACKGROUND = "#303a44"
CANVAS_DISABLED_BUTTON_TEXT = "#aeb9c4"


def style_canvas_control_bar(frame: QFrame, object_name: str) -> None:
    """Give canvas controls explicit, readable enabled and disabled colours."""

    frame.setObjectName(object_name)
    frame.setStyleSheet(
        f"""
        QFrame#{object_name} {{
            background-color: {CANVAS_CONTROL_BACKGROUND};
            border-bottom: 1px solid #3d4853;
        }}
        QFrame#{object_name} QToolButton {{
            color: {CANVAS_BUTTON_TEXT};
            background-color: {CANVAS_BUTTON_BACKGROUND};
            border: 1px solid #647383;
            border-radius: 3px;
            padding: 2px 7px;
        }}
        QFrame#{object_name} QToolButton:hover {{
            background-color: #435465;
            border-color: #8295a8;
        }}
        QFrame#{object_name} QToolButton:pressed {{
            background-color: #202a33;
            border-color: #718395;
        }}
        QFrame#{object_name} QToolButton:checked {{
            color: #ffffff;
            background-color: #245f75;
            border-color: #57d8ff;
        }}
        QFrame#{object_name} QToolButton:checked:hover {{
            background-color: #2c7189;
            border-color: #83e4ff;
        }}
        QFrame#{object_name} QToolButton:disabled {{
            color: {CANVAS_DISABLED_BUTTON_TEXT};
            background-color: {CANVAS_DISABLED_BUTTON_BACKGROUND};
            border-color: #4b5865;
        }}
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
