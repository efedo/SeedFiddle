from __future__ import annotations

import sys
import subprocess
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class EntrypointTests(unittest.TestCase):
    def test_importing_entrypoint_does_not_import_gui_or_ml_packages(self) -> None:
        root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            (
                sys.executable,
                "-c",
                "import seed_vision,sys; "
                "print(','.join(name for name in ('PySide6','cv2','torch') "
                "if name in sys.modules))",
            ),
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(completed.stdout.strip(), "")

    def test_parser_defaults_to_safe_interactive_bootstrap(self) -> None:
        import seed_vision

        arguments = seed_vision.build_parser().parse_args([])
        self.assertEqual(arguments.environment, "auto")
        self.assertEqual(arguments.bootstrap, "ask")
        self.assertFalse(arguments.offline)

    def test_application_starts_main_window_maximized(self) -> None:
        from seedvision.application import run

        events: list[tuple[str, object]] = []

        class FakeCoreApplication:
            setOrganizationName = staticmethod(lambda value: None)
            setApplicationName = staticmethod(lambda value: None)
            setApplicationVersion = staticmethod(lambda value: None)

        class FakeApplication:
            @staticmethod
            def instance():
                return None

            def __init__(self, arguments) -> None:
                events.append(("application", arguments))

            def setStyle(self, style: str) -> None:
                events.append(("style", style))

            def setWindowIcon(self, icon) -> None:
                events.append(("application_icon", icon.path))

            def exec(self) -> int:
                return 0

        class FakeMainWindow:
            def __init__(self, root: Path) -> None:
                events.append(("window", root))

            def setWindowIcon(self, icon) -> None:
                events.append(("window_icon", icon.path))

            def showMaximized(self) -> None:
                events.append(("show", "maximized"))

        class FakeIcon:
            def __init__(self, path: str) -> None:
                self.path = path

        qtcore = types.ModuleType("PySide6.QtCore")
        qtcore.QCoreApplication = FakeCoreApplication
        qtgui = types.ModuleType("PySide6.QtGui")
        qtgui.QIcon = FakeIcon
        qtwidgets = types.ModuleType("PySide6.QtWidgets")
        qtwidgets.QApplication = FakeApplication
        main_window = types.ModuleType("seedvision.ui.main_window")
        main_window.MainWindow = FakeMainWindow
        with patch.dict(
            sys.modules,
            {
                "PySide6": types.ModuleType("PySide6"),
                "PySide6.QtCore": qtcore,
                "PySide6.QtGui": qtgui,
                "PySide6.QtWidgets": qtwidgets,
                "seedvision.ui.main_window": main_window,
            },
        ):
            self.assertEqual(run(Path("workspace")), 0)

        self.assertIn(("show", "maximized"), events)
        application_icons = [value for name, value in events if name == "application_icon"]
        window_icons = [value for name, value in events if name == "window_icon"]
        self.assertEqual(application_icons, window_icons)
        self.assertTrue(application_icons[0].endswith("seed_vision_icon.png"))


if __name__ == "__main__":
    unittest.main()
