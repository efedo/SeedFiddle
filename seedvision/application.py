"""Delayed entry point for the PySide6 application."""

from __future__ import annotations

import sys
from pathlib import Path


def run(root: Path) -> int:
    """Create and run the desktop application after bootstrap has succeeded."""

    from PySide6.QtCore import QCoreApplication
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication

    from seedvision.ui.main_window import MainWindow

    QCoreApplication.setOrganizationName("Seed Vision")
    QCoreApplication.setApplicationName("Seed Vision")
    QCoreApplication.setApplicationVersion("0.1.0-dev")

    application = QApplication.instance() or QApplication(sys.argv)
    application.setStyle("Fusion")
    icon_path = Path(__file__).resolve().parent / "assets" / "seed_vision_icon.png"
    icon = QIcon(str(icon_path))
    application.setWindowIcon(icon)
    window = MainWindow(root)
    window.setWindowIcon(icon)
    window.showMaximized()
    return application.exec()
