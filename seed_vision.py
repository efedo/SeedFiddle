"""Single launch point for the Seed Fiddle desktop application."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from seedvision.bootstrap import (  # noqa: E402 - intentionally after ROOT setup
    BootstrapError,
    create_virtual_environment,
    format_report,
    inspect_runtime,
    install_dependencies,
    load_configuration,
    probe_torch,
    project_venv_python,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch the local Seed Fiddle desktop application."
    )
    parser.add_argument(
        "--environment",
        choices=("auto", "current", "venv"),
        default="auto",
        help="dependency target when bootstrapping is required",
    )
    parser.add_argument(
        "--bootstrap",
        choices=("ask", "auto", "never"),
        default="ask",
        help="whether missing dependencies may be installed",
    )
    parser.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="alias for --bootstrap never",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="install only from compatible wheels in the local wheels directory",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="print runtime and package status without installing or launching",
    )
    return parser


def _ask_environment() -> str | None:
    if not sys.stdin.isatty():
        return None
    print("\nRequired packages are missing or incompatible.")
    print("  1. Create or use .venv beside seed_vision.py (recommended)")
    print("  2. Install into the current Python interpreter")
    print("  q. Exit without making changes")
    while True:
        answer = input("Select an environment [1/2/q]: ").strip().lower()
        if answer in {"1", "venv"}:
            return "venv"
        if answer in {"2", "current"}:
            return "current"
        if answer in {"q", "quit", "exit"}:
            return None
        print("Please enter 1, 2, or q.")


def _confirm_install(environment: str, specs: Sequence[str]) -> bool:
    if not sys.stdin.isatty():
        return False
    target = ".venv" if environment == "venv" else sys.executable
    print(f"\nTarget: {target}")
    print("Packages:")
    for spec in specs:
        print(f"  - {spec}")
    answer = input("Proceed with package installation? [y/N]: ").strip().lower()
    return answer in {"y", "yes"}


def _run_with_interpreter(
    python_executable: Path | str,
    *,
    offline: bool,
    bootstrap: str,
) -> int:
    command = [
        str(python_executable),
        str(ROOT / "seed_vision.py"),
        "--environment",
        "current",
        "--bootstrap",
        bootstrap,
    ]
    if offline:
        command.append("--offline")
    try:
        return subprocess.call(command)
    except OSError as error:
        raise BootstrapError(
            f"Could not start Seed Fiddle with {python_executable}: {error}"
        ) from error


def _launch_application() -> int:
    # Third-party imports begin only inside this delayed entry point.
    try:
        from seedvision.application import run

        return run(ROOT)
    except (ImportError, OSError) as error:
        raise BootstrapError(
            "A required package was discovered but could not be imported. "
            f"Run --diagnostics and check binary compatibility. Details: {error}"
        ) from error


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bootstrap_mode = "never" if args.no_bootstrap else args.bootstrap

    try:
        configuration = load_configuration(ROOT)
        report = inspect_runtime(configuration)

        if args.diagnostics:
            print(format_report(report, configuration))
            torch_check = next(
                (
                    check
                    for check in report.dependency_checks
                    if check.dependency.import_name == "torch"
                ),
                None,
            )
            if torch_check and torch_check.compatible:
                print(f"Acceleration: {probe_torch(report.python_executable)}")
                print(
                    "Image pipeline: PyTorch CUDA for full-raster calibration, "
                    "colour/noise probabilities, image-quality products, and edge "
                    "diagnostics; GPU-resized evidence plus bounded OpenCV/NumPy CPU "
                    "topology for active procedural instances; former experimental "
                    "proposal/mask stages retained in the node toolbox; CPU also used "
                    "for decoding, Qt display, and compact metadata."
                )
            return 0 if report.ready else 2

        if report.ready:
            return _launch_application()

        # If a local environment already exists, auto mode gives it first chance
        # before proposing changes to the current interpreter.
        venv_python = project_venv_python(ROOT)
        if (
            args.environment in {"auto", "venv"}
            and venv_python.is_file()
            and Path(sys.executable).resolve() != venv_python.resolve()
        ):
            return _run_with_interpreter(
                venv_python,
                offline=args.offline,
                bootstrap=bootstrap_mode,
            )

        print(format_report(report, configuration))
        if not report.python_compatible or report.architecture_bits != 64:
            raise BootstrapError(
                "Dependency installation cannot repair this Python runtime. Use a "
                f"64-bit Python >= {configuration.python.minimum} and "
                f"< {configuration.python.maximum_exclusive}."
            )
        if bootstrap_mode == "never":
            print("\nBootstrap is disabled; no changes were made.", file=sys.stderr)
            return 2

        environment = args.environment
        if environment == "auto":
            environment = "venv" if bootstrap_mode == "auto" else _ask_environment()
        if environment is None:
            print("No changes were made.")
            return 2

        running_in_project_venv = (
            report.in_virtual_environment
            and Path(sys.executable).resolve() == project_venv_python(ROOT).resolve()
        )
        dependencies = (
            configuration.required_packages
            if environment == "venv" and not running_in_project_venv
            else report.packages_to_install
        )
        specs = tuple(dependency.install_spec for dependency in dependencies)
        if bootstrap_mode == "ask" and not _confirm_install(environment, specs):
            print("No changes were made.")
            return 2

        target_python: Path | str
        if environment == "venv":
            target_python = create_virtual_environment(ROOT)
        else:
            target_python = sys.executable

        install_dependencies(
            target_python,
            dependencies,
            root=ROOT,
            offline=args.offline,
        )
        return _run_with_interpreter(
            target_python,
            offline=args.offline,
            bootstrap="never",
        )
    except BootstrapError as error:
        print(f"Seed Fiddle bootstrap error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
