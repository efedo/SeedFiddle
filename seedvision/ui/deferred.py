"""Run a callback once the event loop is next free."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QObject, QTimer


def run_soon(owner: QObject, callback: Callable[[], object]) -> None:
    """Call ``callback`` on the next event-loop pass unless ``owner`` is deleted.

    Use this rather than ``QTimer.singleShot(0, callback)``. That timer outlives
    its target, and PySide stores what its callback raises instead of reporting
    it. During ``processEvents()`` or a static dialog such as
    ``QMessageBox.warning()``, the stored error reaches the next Python override,
    and graphics items painting for the first time can crash the process. A timer
    owned by ``owner`` dies with it, and an error in its slot is reported normally.
    """

    timer = QTimer(owner)
    timer.setSingleShot(True)
    timer.timeout.connect(callback)
    timer.timeout.connect(timer.deleteLater)
    timer.start(0)
