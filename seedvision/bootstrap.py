"""Standard-library-only runtime inspection and dependency bootstrapping.

This module must remain importable before PySide6, NumPy, OpenCV, or PyTorch are
installed. Keep all third-party imports in the application modules.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import platform
import re
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence


class BootstrapError(RuntimeError):
    """Raised when the runtime cannot be inspected or provisioned safely."""


@dataclass(frozen=True)
class PythonRequirement:
    minimum: str
    maximum_exclusive: str
    recommended: str


@dataclass(frozen=True)
class Dependency:
    distribution: str
    import_name: str
    install_spec: str
    minimum: str
    maximum_exclusive: str
    purpose: str
    index_url: str | None = None
    required: bool = True


@dataclass(frozen=True)
class BootstrapConfiguration:
    python: PythonRequirement
    packages: tuple[Dependency, ...]

    @property
    def required_packages(self) -> tuple[Dependency, ...]:
        return tuple(package for package in self.packages if package.required)


@dataclass(frozen=True)
class DependencyCheck:
    dependency: Dependency
    installed: bool
    compatible: bool
    version: str | None
    detail: str


@dataclass(frozen=True)
class RuntimeReport:
    python_version: str
    python_executable: str
    architecture_bits: int
    in_virtual_environment: bool
    python_compatible: bool
    dependency_checks: tuple[DependencyCheck, ...]

    @property
    def ready(self) -> bool:
        return (
            self.architecture_bits == 64
            and self.python_compatible
            and all(
                check.compatible
                for check in self.dependency_checks
                if check.dependency.required
            )
        )

    @property
    def packages_to_install(self) -> tuple[Dependency, ...]:
        return tuple(
            check.dependency
            for check in self.dependency_checks
            if check.dependency.required and not check.compatible
        )


_VERSION_PATTERN = re.compile(r"^\s*(\d+)(?:\.(\d+))?(?:\.(\d+))?")


def version_key(value: str) -> tuple[int, int, int]:
    """Return the numeric release prefix needed for bounded compatibility checks."""

    match = _VERSION_PATTERN.match(value)
    if not match:
        raise ValueError(f"Unsupported version string: {value!r}")
    return tuple(int(part or 0) for part in match.groups())


def version_is_compatible(value: str, minimum: str, maximum_exclusive: str) -> bool:
    candidate = version_key(value)
    return version_key(minimum) <= candidate < version_key(maximum_exclusive)


def load_configuration(root: Path) -> BootstrapConfiguration:
    manifest_path = root / "config" / "runtime_dependencies.json"
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BootstrapError(
            f"Cannot read dependency manifest {manifest_path}: {error}"
        ) from error

    try:
        python = PythonRequirement(**raw["python"])
        packages = tuple(Dependency(**item) for item in raw["packages"])
    except (KeyError, TypeError) as error:
        raise BootstrapError(
            f"Dependency manifest {manifest_path} has an invalid schema: {error}"
        ) from error

    if not packages:
        raise BootstrapError("The dependency manifest does not list any packages.")
    return BootstrapConfiguration(python=python, packages=packages)


def inspect_dependency(
    dependency: Dependency,
    *,
    module_finder: Callable[[str], object | None] = importlib.util.find_spec,
    version_getter: Callable[[str], str] = importlib.metadata.version,
) -> DependencyCheck:
    try:
        module_spec = module_finder(dependency.import_name)
    except (ImportError, AttributeError, ValueError) as error:
        return DependencyCheck(
            dependency,
            installed=False,
            compatible=False,
            version=None,
            detail=f"module discovery failed: {error}",
        )

    if module_spec is None:
        return DependencyCheck(
            dependency,
            installed=False,
            compatible=False,
            version=None,
            detail="not installed",
        )

    try:
        installed_version = version_getter(dependency.distribution)
    except importlib.metadata.PackageNotFoundError:
        # A centrally supplied or conda-built module may not expose standard pip
        # distribution metadata. Accept it provisionally and verify the real import
        # when the application starts.
        return DependencyCheck(
            dependency,
            installed=True,
            compatible=True,
            version=None,
            detail="import available; distribution version unavailable",
        )

    try:
        compatible = version_is_compatible(
            installed_version,
            dependency.minimum,
            dependency.maximum_exclusive,
        )
    except ValueError as error:
        return DependencyCheck(
            dependency,
            installed=True,
            compatible=False,
            version=installed_version,
            detail=str(error),
        )

    detail = "compatible" if compatible else (
        f"requires >= {dependency.minimum} and < {dependency.maximum_exclusive}"
    )
    return DependencyCheck(
        dependency,
        installed=True,
        compatible=compatible,
        version=installed_version,
        detail=detail,
    )


def inspect_runtime(configuration: BootstrapConfiguration) -> RuntimeReport:
    python_version = platform.python_version()
    try:
        python_compatible = version_is_compatible(
            python_version,
            configuration.python.minimum,
            configuration.python.maximum_exclusive,
        )
    except ValueError:
        python_compatible = False

    return RuntimeReport(
        python_version=python_version,
        python_executable=sys.executable,
        architecture_bits=struct.calcsize("P") * 8,
        in_virtual_environment=sys.prefix != sys.base_prefix,
        python_compatible=python_compatible,
        dependency_checks=tuple(
            inspect_dependency(dependency) for dependency in configuration.packages
        ),
    )


def format_report(report: RuntimeReport, configuration: BootstrapConfiguration) -> str:
    environment = "virtual environment" if report.in_virtual_environment else "current interpreter"
    python_status = "compatible" if report.python_compatible else (
        f"requires >= {configuration.python.minimum} and "
        f"< {configuration.python.maximum_exclusive}"
    )
    lines = [
        "Seed Fiddle runtime diagnostics",
        f"Python: {report.python_version} ({report.architecture_bits}-bit; {python_status})",
        f"Interpreter: {report.python_executable}",
        f"Environment: {environment}",
        "Packages:",
    ]
    for check in report.dependency_checks:
        if not check.dependency.required and not check.compatible:
            marker = "OPTIONAL"
        else:
            marker = "OK" if check.compatible else (
                "UPDATE" if check.installed else "MISSING"
            )
        version = check.version or "unknown"
        lines.append(
            f"  [{marker}] {check.dependency.distribution} {version} - {check.detail}"
        )
    lines.append(f"Ready: {'yes' if report.ready else 'no'}")
    return "\n".join(lines)


def project_venv_python(root: Path) -> Path:
    if sys.platform == "win32":
        return root / ".venv" / "Scripts" / "python.exe"
    return root / ".venv" / "bin" / "python"


def wheel_files(wheelhouse: Path) -> tuple[Path, ...]:
    if not wheelhouse.is_dir():
        return ()
    return tuple(sorted(wheelhouse.glob("*.whl")))


def build_pip_command(
    python_executable: Path | str,
    install_specs: Iterable[str],
    *,
    wheelhouse: Path,
    offline: bool,
    index_url: str | None = None,
) -> list[str]:
    specs = tuple(install_specs)
    if not specs:
        raise BootstrapError("No dependency specifications were provided to pip.")
    available_wheels = wheel_files(wheelhouse)
    if offline and not available_wheels:
        raise BootstrapError(
            f"Offline mode requires wheel files in {wheelhouse}."
        )

    command = [
        str(python_executable),
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
    ]
    if available_wheels:
        if offline:
            command.append("--no-index")
        command.extend(("--find-links", str(wheelhouse)))
    if index_url and not offline:
        command.extend(("--index-url", index_url))
    command.extend(specs)
    return command


def run_checked(command: Sequence[str], *, description: str) -> None:
    try:
        result = subprocess.run(list(command), check=False)
    except OSError as error:
        raise BootstrapError(f"Could not {description}: {error}") from error
    if result.returncode != 0:
        raise BootstrapError(
            f"Could not {description}; command exited with code {result.returncode}."
        )


def create_virtual_environment(root: Path) -> Path:
    environment_python = project_venv_python(root)
    if environment_python.is_file():
        return environment_python
    run_checked(
        (sys.executable, "-m", "venv", str(root / ".venv")),
        description="create the project virtual environment",
    )
    if not environment_python.is_file():
        raise BootstrapError(
            f"Virtual environment was created without an interpreter at {environment_python}."
        )
    return environment_python


def install_dependencies(
    python_executable: Path | str,
    dependencies: Iterable[Dependency],
    *,
    root: Path,
    offline: bool,
) -> None:
    requested = tuple(dependencies)
    if not requested:
        raise BootstrapError("No runtime dependencies require installation.")

    if offline:
        command = build_pip_command(
            python_executable,
            (dependency.install_spec for dependency in requested),
            wheelhouse=root / "wheels",
            offline=True,
        )
        run_checked(command, description="install Seed Fiddle runtime dependencies")
        return

    groups: dict[str | None, list[Dependency]] = {}
    for dependency in requested:
        groups.setdefault(dependency.index_url, []).append(dependency)
    for index_url, group in groups.items():
        command = build_pip_command(
            python_executable,
            (dependency.install_spec for dependency in group),
            wheelhouse=root / "wheels",
            offline=False,
            index_url=index_url,
        )
        run_checked(command, description="install Seed Fiddle runtime dependencies")


def probe_torch(python_executable: Path | str) -> str:
    """Query PyTorch in a child process so a failed DLL load cannot break bootstrap."""

    script = (
        "import torch; "
        "available=torch.cuda.is_available(); "
        "name=torch.cuda.get_device_name(0) if available else 'none'; "
        "print(f'torch={torch.__version__}; cuda={available}; device={name}')"
    )
    try:
        result = subprocess.run(
            (str(python_executable), "-c", script),
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"PyTorch probe failed: {error}"
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()
        return f"PyTorch probe failed: {detail[-1] if detail else 'unknown error'}"
    return result.stdout.strip()
