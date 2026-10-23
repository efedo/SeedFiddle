"""Help menu: operator guide, keyboard shortcuts, runtime summary and About."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QAction, QDesktopServices, QKeySequence
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


def shortcut_rows(window) -> list[tuple[str, str]]:
    """Return (action, shortcut) pairs for every window action with a shortcut."""

    rows: dict[str, str] = {}
    for action in window.findChildren(QAction):
        sequence = action.shortcut()
        if sequence.isEmpty() or not action.text():
            continue
        name = action.text().replace("&", "").rstrip("…").strip()
        rows.setdefault(name, sequence.toString(QKeySequence.SequenceFormat.NativeText))
    return sorted(rows.items(), key=lambda row: row[0].casefold())


def show_shortcuts(window) -> QDialog:
    dialog = QDialog(window)
    dialog.setWindowTitle("Keyboard shortcuts")
    layout = QVBoxLayout(dialog)
    rows = shortcut_rows(window)
    table = QTableWidget(len(rows), 2, dialog)
    table.setObjectName("shortcutTable")
    table.setHorizontalHeaderLabels(("Action", "Shortcut"))
    table.verticalHeader().hide()
    table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    for row, (name, sequence) in enumerate(rows):
        table.setItem(row, 0, QTableWidgetItem(name))
        table.setItem(row, 1, QTableWidgetItem(sequence))
    table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
    table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
    layout.addWidget(table)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, dialog)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    dialog.resize(460, 520)
    dialog.show()
    return dialog


def operator_guide_path(root: Path) -> Path:
    return Path(root) / "docs" / "OPERATOR_GUIDE.md"


def open_operator_guide(window) -> bool:
    path = operator_guide_path(window._root)
    if not path.is_file():
        QMessageBox.information(window, "Operator guide", f"The guide was not found at {path}.")
        return False
    return QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


def show_about(window) -> None:
    from seedvision import __version__

    QMessageBox.about(
        window,
        "About Seed Fiddle",
        f"<b>Seed Fiddle</b> {__version__}<br>"
        "Native desktop seed-image analysis with PyTorch CUDA and PySide6.<br>"
        "Results are approximate proposals for correction, not validated counts.",
    )


def install_help_actions(window, menu) -> None:
    """Populate the Help menu; the runtime summary action already exists."""

    window.operator_guide_action = QAction("Operator guide", window)
    window.operator_guide_action.setShortcut(QKeySequence.StandardKey.HelpContents)
    window.operator_guide_action.triggered.connect(lambda: open_operator_guide(window))
    window.shortcuts_action = QAction("Keyboard shortcuts…", window)
    window.shortcuts_action.triggered.connect(lambda: show_shortcuts(window))
    window.about_action = QAction("About Seed Fiddle", window)
    window.about_action.triggered.connect(lambda: show_about(window))
    menu.addAction(window.operator_guide_action)
    menu.addAction(window.shortcuts_action)
    menu.addSeparator()
    menu.addAction(window.diagnostics_action)
    menu.addSeparator()
    menu.addAction(window.about_action)
