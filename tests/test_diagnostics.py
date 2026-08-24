from __future__ import annotations

from logging.handlers import RotatingFileHandler
from pathlib import Path
import tempfile
import unittest

from seedvision.diagnostics import (
    LOGGER,
    LOG_BACKUP_COUNT,
    MAX_LOG_BYTES,
    configure_error_logging,
    current_log_path,
    diagnostic_log_note,
    shutdown_error_logging,
)


class DiagnosticLoggingTests(unittest.TestCase):
    def tearDown(self) -> None:
        shutdown_error_logging()

    def test_rotating_log_retains_full_exception_traceback(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        try:
            path = configure_error_logging(
                Path("D:/example/SeedFiddle"),
                log_directory=Path(temporary.name),
                install_hooks=False,
            )
            try:
                raise ValueError("synthetic diagnostic failure")
            except ValueError:
                LOGGER.exception("Could not install completed analysis")
            for handler in LOGGER.handlers:
                handler.flush()

            text = path.read_text(encoding="utf-8")
            self.assertIn("Could not install completed analysis", text)
            self.assertIn("Traceback (most recent call last):", text)
            self.assertIn("ValueError: synthetic diagnostic failure", text)
            self.assertEqual(current_log_path(), path)
            self.assertIn(str(path), diagnostic_log_note())

            rotating = [
                handler
                for handler in LOGGER.handlers
                if isinstance(handler, RotatingFileHandler)
            ]
            self.assertEqual(len(rotating), 1)
            self.assertEqual(rotating[0].maxBytes, MAX_LOG_BYTES)
            self.assertEqual(rotating[0].backupCount, LOG_BACKUP_COUNT)
        finally:
            shutdown_error_logging()
            temporary.cleanup()


if __name__ == "__main__":
    unittest.main()
