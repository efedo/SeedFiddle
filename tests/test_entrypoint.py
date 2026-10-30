from __future__ import annotations

from tests.functional_units import functional_units

import io
from contextlib import redirect_stderr, redirect_stdout
import sys
import subprocess
import types
import unittest
from pathlib import Path
from unittest.mock import patch


@functional_units
class EntrypointTests(unittest.TestCase):
    def check_test_importing_entrypoint_does_not_import_gui_or_ml_packages(self) -> None:
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

    def check_test_parser_defaults_to_safe_interactive_bootstrap(self) -> None:
        import seed_vision

        arguments = seed_vision.build_parser().parse_args([])
        self.assertEqual(arguments.environment, "auto")
        self.assertEqual(arguments.bootstrap, "ask")
        self.assertFalse(arguments.offline)
        self.assertFalse(arguments.no_cellpose)

    def check_test_failed_cuda_readiness_blocks_launch_and_diagnostics_returns_nonzero(self) -> None:
        import seed_vision

        report = types.SimpleNamespace(
            ready=False,
            packages_to_install=(),
            python_compatible=True,
            architecture_bits=64,
            torch_probe=None,
        )
        with patch("seed_vision.inspect_runtime", return_value=report), \
             patch("seed_vision.format_report", return_value="CUDA device execution: FAIL"), \
             patch("seed_vision._launch_application") as launch:
            diagnostics_output = io.StringIO()
            with redirect_stdout(diagnostics_output):
                diagnostics_status = seed_vision.main(
                    ["--environment", "current", "--diagnostics"]
                )
            launch_output = io.StringIO()
            with redirect_stdout(launch_output), redirect_stderr(io.StringIO()):
                launch_status = seed_vision.main(
                    ["--environment", "current", "--no-bootstrap"]
                )
        self.assertEqual(diagnostics_status, 2)
        self.assertIn("CUDA device execution: FAIL", diagnostics_output.getvalue())
        self.assertEqual(launch_status, 2)
        launch.assert_not_called()

    def check_test_hardware_failure_does_not_trigger_package_reinstallation(self) -> None:
        import seed_vision

        report = types.SimpleNamespace(
            ready=False,
            packages_to_install=(),
            python_compatible=True,
            architecture_bits=64,
            torch_probe=None,
        )
        with patch("seed_vision.inspect_runtime", return_value=report), \
             patch("seed_vision.format_report", return_value="CUDA device unavailable"), \
             patch("seed_vision.install_dependencies") as install, \
             patch("seed_vision._run_with_interpreter") as rerun:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                status = seed_vision.main(["--environment", "current", "--bootstrap", "auto"])
        self.assertEqual(status, 2)
        install.assert_not_called()
        rerun.assert_not_called()

    def check_test_no_bootstrap_does_not_install_cuda_package_repair(self) -> None:
        import seed_vision

        torch = types.SimpleNamespace(
            distribution="torch",
            install_spec="torch==2.14.1+cu126",
        )
        report = types.SimpleNamespace(
            ready=False,
            packages_to_install=(torch,),
            python_compatible=True,
            architecture_bits=64,
            torch_probe=None,
        )
        with patch("seed_vision.inspect_runtime", return_value=report), \
             patch("seed_vision.format_report", return_value="CPU-only PyTorch"), \
             patch("seed_vision.install_dependencies") as install:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                status = seed_vision.main(
                    ["--environment", "current", "--no-bootstrap"]
                )
        self.assertEqual(status, 2)
        install.assert_not_called()

    def check_test_cuda_package_repair_rechecks_target_without_bootstrap_loop(self) -> None:
        import seed_vision

        torch = types.SimpleNamespace(
            distribution="torch",
            install_spec="torch==2.14.1+cu126",
        )
        report = types.SimpleNamespace(
            ready=False,
            packages_to_install=(torch,),
            python_compatible=True,
            architecture_bits=64,
            in_virtual_environment=False,
            cellpose_missing=False,
            torch_probe=None,
        )
        with patch("seed_vision.inspect_runtime", return_value=report), \
             patch("seed_vision.format_report", return_value="CPU-only PyTorch"), \
             patch("seed_vision.install_dependencies") as install, \
             patch("seed_vision.install_cellpose") as install_cellpose, \
             patch("seed_vision._run_with_interpreter", return_value=2) as rerun:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                status = seed_vision.main(
                    ["--environment", "current", "--bootstrap", "auto"]
                )
        self.assertEqual(status, 2)
        install.assert_called_once()
        self.assertEqual(
            install.call_args.args[1],
            (torch,),
        )
        self.assertEqual(rerun.call_args.kwargs["bootstrap"], "never")
        install_cellpose.assert_not_called()

    def check_test_ready_launch_adds_cellpose_unless_told_not_to(self) -> None:
        import seed_vision

        report = types.SimpleNamespace(ready=True, cellpose_missing=True)
        with patch("seed_vision.inspect_runtime", return_value=report), \
             patch("seed_vision.install_cellpose") as install, \
             patch("seed_vision._confirm_cellpose", return_value=False) as confirm, \
             patch("seed_vision._launch_application", return_value=0) as launch:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                statuses = [
                    seed_vision.main(["--environment", "current", "--bootstrap", "auto"]),
                    seed_vision.main(["--environment", "current", "--bootstrap", "auto",
                                      "--no-cellpose"]),
                    seed_vision.main(["--environment", "current", "--no-bootstrap"]),
                    seed_vision.main(["--environment", "current"]),
                ]
        self.assertEqual(statuses, [0, 0, 0, 0])
        self.assertEqual(launch.call_count, 4)
        install.assert_called_once()
        self.assertEqual(install.call_args.args[0], sys.executable)
        # The default ask mode offered it once and the answer was no.
        confirm.assert_called_once()

    def check_test_failed_cellpose_install_still_launches(self) -> None:
        import seed_vision
        from seedvision.bootstrap import BootstrapError

        report = types.SimpleNamespace(ready=True, cellpose_missing=True)
        errors = io.StringIO()
        with patch("seed_vision.inspect_runtime", return_value=report), \
             patch("seed_vision.install_cellpose",
                   side_effect=BootstrapError("no torchvision wheel")), \
             patch("seed_vision._launch_application", return_value=0) as launch:
            with redirect_stdout(io.StringIO()), redirect_stderr(errors):
                status = seed_vision.main(["--environment", "current", "--bootstrap", "auto"])
        self.assertEqual(status, 0)
        launch.assert_called_once_with()
        self.assertIn("Cellpose-SAM was not installed: no torchvision wheel", errors.getvalue())

    def check_test_new_environment_gets_cellpose_after_required_packages(self) -> None:
        import seed_vision

        report = types.SimpleNamespace(
            ready=False,
            packages_to_install=(types.SimpleNamespace(install_spec="PySide6==6.11.1"),),
            python_compatible=True,
            architecture_bits=64,
            in_virtual_environment=False,
            cellpose_missing=False,
            torch_probe=None,
        )
        missing_venv = Path("missing-venv") / "python.exe"
        order = []
        with patch("seed_vision.inspect_runtime", return_value=report), \
             patch("seed_vision.format_report", return_value="PySide6 missing"), \
             patch("seed_vision.project_venv_python", return_value=missing_venv), \
             patch("seed_vision.create_virtual_environment", return_value="venv.exe"), \
             patch("seed_vision.install_dependencies",
                   side_effect=lambda *a, **k: order.append("required")), \
             patch("seed_vision.install_cellpose",
                   side_effect=lambda *a, **k: order.append("cellpose")) as install_cellpose, \
             patch("seed_vision._run_with_interpreter", return_value=0) as rerun:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                status = seed_vision.main(["--environment", "venv", "--bootstrap", "auto"])
                skipped_status = seed_vision.main(
                    ["--environment", "venv", "--bootstrap", "auto", "--no-cellpose"]
                )
        self.assertEqual((status, skipped_status), (0, 0))
        self.assertEqual(order, ["required", "cellpose", "required"])
        self.assertEqual(install_cellpose.call_args.args[0], "venv.exe")
        self.assertFalse(rerun.call_args_list[0].kwargs["no_cellpose"])
        self.assertTrue(rerun.call_args_list[1].kwargs["no_cellpose"])

    def check_test_cpu_learning_maintenance_does_not_require_a_gpu(self) -> None:
        import seed_vision

        with patch("seed_vision.inspect_runtime") as inspect, \
             patch("seed_vision._run_learning_action", return_value=0) as action:
            for arguments in (
                ["--learning-audit", "manifest.json"],
                ["--create-synthetic-learning-data", "synthetic-data"],
            ):
                self.assertEqual(seed_vision.main(arguments), 0)
        inspect.assert_not_called()
        self.assertEqual(action.call_count, 2)

    def check_test_batch_analysis_runs_shared_runtime_preflight_before_importing_runner(self) -> None:
        from scripts import analyze_pilot
        from seedvision.bootstrap import BootstrapError

        with patch(
            "scripts.analyze_pilot.require_runtime_ready",
            side_effect=BootstrapError("CUDA is not ready"),
        ) as preflight:
            output = io.StringIO()
            errors = io.StringIO()
            with redirect_stdout(output), redirect_stderr(errors):
                status = analyze_pilot.main(["--project", "missing.seedfiddle-project.json"])
        self.assertEqual(status, 2)
        preflight.assert_called_once_with()
        self.assertIn("CUDA is not ready", errors.getvalue())

    def check_test_application_starts_main_window_maximized(self) -> None:
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

    def test_workflow_lifecycle(self):
        """3 related EntrypointTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_importing_entrypoint_does_not_import_gui_or_ml_packages)
        self.scenario(self.check_test_parser_defaults_to_safe_interactive_bootstrap)
        self.scenario(self.check_test_application_starts_main_window_maximized)

    def test_cellpose_installation(self):
        """3 Cellpose-SAM launch scenarios, each with isolated bootstrap mocks."""
        self.scenario(self.check_test_ready_launch_adds_cellpose_unless_told_not_to)
        self.scenario(self.check_test_failed_cellpose_install_still_launches)
        self.scenario(self.check_test_new_environment_gets_cellpose_after_required_packages)

    def test_cuda_readiness_enforcement(self):
        """6 readiness scenarios, each with isolated bootstrap mocks."""
        self.scenario(self.check_test_failed_cuda_readiness_blocks_launch_and_diagnostics_returns_nonzero)
        self.scenario(self.check_test_hardware_failure_does_not_trigger_package_reinstallation)
        self.scenario(self.check_test_no_bootstrap_does_not_install_cuda_package_repair)
        self.scenario(self.check_test_cuda_package_repair_rechecks_target_without_bootstrap_loop)
        self.scenario(self.check_test_cpu_learning_maintenance_does_not_require_a_gpu)
        self.scenario(self.check_test_batch_analysis_runs_shared_runtime_preflight_before_importing_runner)


if __name__ == "__main__":
    unittest.main()
