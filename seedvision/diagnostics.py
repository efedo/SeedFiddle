"""Persistent diagnostics for the desktop application.

The GUI intentionally catches worker and display-installation exceptions so one
bad image cannot strand the serial analysis queue. Those boundaries must log the
active traceback before reducing an exception to a user-facing message.
"""

from __future__ import annotations

import faulthandler
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import platform
import sys
import tempfile
import threading
from types import TracebackType
from typing import TextIO


LOGGER_NAME = "seedfiddle"
LOG_FILE_NAME = "seed-fiddle.log"
CRASH_LOG_FILE_NAME = "seed-fiddle-crash.log"
MAX_LOG_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 5

LOGGER = logging.getLogger(LOGGER_NAME)
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False
if not LOGGER.handlers:
    LOGGER.addHandler(logging.NullHandler())

_log_path: Path | None = None
_crash_log_path: Path | None = None
_crash_stream: TextIO | None = None
_hooks_installed = False
_qt_handler_installed = False
_qt_message_handler = None
_previous_qt_message_handler = None
_previous_sys_excepthook = sys.excepthook
_previous_thread_excepthook = getattr(threading, "excepthook", None)


def default_log_directory() -> Path:
    """Return the per-user directory used for persistent diagnostic logs."""

    override = os.environ.get("SEEDFIDDLE_LOG_DIR")
    if override:
        return Path(override).expanduser()
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        if base:
            return Path(base) / "Seed Fiddle" / "logs"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Logs" / "Seed Fiddle"
    state_home = os.environ.get("XDG_STATE_HOME")
    base = Path(state_home).expanduser() if state_home else Path.home() / ".local" / "state"
    return base / "seed-fiddle" / "logs"


def _create_log_directory(requested: Path) -> Path:
    try:
        requested.mkdir(parents=True, exist_ok=True)
        return requested
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "Seed Fiddle" / "logs"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def _install_exception_hooks() -> None:
    global _hooks_installed
    if _hooks_installed:
        return

    def sys_exception_hook(
        exception_type: type[BaseException],
        exception: BaseException,
        traceback: TracebackType | None,
    ) -> None:
        if issubclass(exception_type, KeyboardInterrupt):
            _previous_sys_excepthook(exception_type, exception, traceback)
            return
        LOGGER.critical(
            "Unhandled main-thread exception",
            exc_info=(exception_type, exception, traceback),
        )
        _previous_sys_excepthook(exception_type, exception, traceback)

    def thread_exception_hook(arguments) -> None:
        LOGGER.critical(
            "Unhandled exception in thread %s",
            getattr(arguments.thread, "name", "unknown"),
            exc_info=(
                arguments.exc_type,
                arguments.exc_value,
                arguments.exc_traceback,
            ),
        )
        if _previous_thread_excepthook is not None:
            _previous_thread_excepthook(arguments)

    sys.excepthook = sys_exception_hook
    if hasattr(threading, "excepthook"):
        threading.excepthook = thread_exception_hook
    _hooks_installed = True


def _enable_fatal_crash_log(directory: Path) -> None:
    global _crash_log_path, _crash_stream
    if _crash_stream is not None:
        return
    crash_path = directory / CRASH_LOG_FILE_NAME
    try:
        stream = crash_path.open("a", encoding="utf-8", buffering=1)
        faulthandler.enable(file=stream, all_threads=True)
    except (OSError, RuntimeError):
        return
    _crash_log_path = crash_path
    _crash_stream = stream


def configure_error_logging(
    root: Path | None = None,
    *,
    log_directory: Path | None = None,
    install_hooks: bool = True,
) -> Path:
    """Configure a bounded persistent log and return its absolute path."""

    global _log_path
    directory = _create_log_directory(
        Path(log_directory) if log_directory is not None else default_log_directory()
    )
    destination = (directory / LOG_FILE_NAME).resolve()
    existing = next(
        (
            handler
            for handler in LOGGER.handlers
            if getattr(handler, "_seedfiddle_diagnostic_handler", False)
            and Path(handler.baseFilename).resolve() == destination
        ),
        None,
    )
    if existing is None:
        for handler in tuple(LOGGER.handlers):
            if getattr(handler, "_seedfiddle_diagnostic_handler", False):
                LOGGER.removeHandler(handler)
                handler.close()
        handler = RotatingFileHandler(
            destination,
            maxBytes=MAX_LOG_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        handler._seedfiddle_diagnostic_handler = True
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s.%(msecs)03d %(levelname)s "
                "[process=%(process)d thread=%(threadName)s] "
                "%(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        LOGGER.addHandler(handler)
    _log_path = destination
    if install_hooks:
        _install_exception_hooks()
        _enable_fatal_crash_log(directory)
    LOGGER.info(
        "Seed Fiddle diagnostic session started; root=%s; python=%s; platform=%s",
        Path(root).resolve() if root is not None else "unknown",
        sys.version.replace("\n", " "),
        platform.platform(),
    )
    return destination


def install_qt_message_logging() -> None:
    """Capture Qt warnings and fatal messages after PySide6 is available."""

    global _qt_handler_installed, _qt_message_handler, _previous_qt_message_handler
    if _qt_handler_installed:
        return
    try:
        from PySide6.QtCore import QtMsgType, qInstallMessageHandler
    except ImportError:
        return

    levels = {
        QtMsgType.QtDebugMsg: logging.DEBUG,
        QtMsgType.QtInfoMsg: logging.INFO,
        QtMsgType.QtWarningMsg: logging.WARNING,
        QtMsgType.QtCriticalMsg: logging.ERROR,
        QtMsgType.QtFatalMsg: logging.CRITICAL,
    }

    def qt_message_handler(message_type, context, message) -> None:
        details = ""
        if context is not None and getattr(context, "file", None):
            details = (
                f" ({context.file}:{getattr(context, 'line', 0)} "
                f"{getattr(context, 'function', '')})"
            )
        LOGGER.log(levels.get(message_type, logging.INFO), "Qt: %s%s", message, details)
        if _previous_qt_message_handler is not None:
            _previous_qt_message_handler(message_type, context, message)

    _qt_message_handler = qt_message_handler
    _previous_qt_message_handler = qInstallMessageHandler(qt_message_handler)
    _qt_handler_installed = True


def current_log_path() -> Path | None:
    return _log_path


def current_crash_log_path() -> Path | None:
    return _crash_log_path


def diagnostic_log_note() -> str:
    """Return a user-facing location note when logging is configured."""

    if _log_path is None:
        return ""
    return f"\n\nFull diagnostic traceback:\n{_log_path}"


def shutdown_error_logging() -> None:
    """Close project-owned handlers; intended for isolated tests."""

    global _log_path, _crash_log_path, _crash_stream
    for handler in tuple(LOGGER.handlers):
        if getattr(handler, "_seedfiddle_diagnostic_handler", False):
            LOGGER.removeHandler(handler)
            handler.close()
    if _crash_stream is not None:
        if faulthandler.is_enabled():
            faulthandler.disable()
        _crash_stream.close()
        _crash_stream = None
    _log_path = None
    _crash_log_path = None
