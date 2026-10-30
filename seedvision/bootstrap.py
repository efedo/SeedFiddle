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
    requires_cuda: bool = False


@dataclass(frozen=True)
class BootstrapConfiguration:
    python: PythonRequirement
    packages: tuple[Dependency, ...]
    cellpose_dependencies: tuple[str, ...] = ()

    @property
    def required_packages(self) -> tuple[Dependency, ...]:
        return tuple(package for package in self.packages if package.required)

    @property
    def cellpose(self) -> Dependency | None:
        return next(
            (package for package in self.packages if package.distribution == "cellpose"),
            None,
        )


@dataclass(frozen=True)
class DependencyCheck:
    dependency: Dependency
    installed: bool
    compatible: bool
    version: str | None
    detail: str


@dataclass(frozen=True)
class TorchProbeResult:
    python_executable: str
    state: str
    torch_version: str | None = None
    module_location: str | None = None
    cuda_built: bool | None = None
    cuda_version: str | None = None
    hip_version: str | None = None
    cuda_available: bool | None = None
    gpu_name: str | None = None
    calculation_succeeded: bool = False
    failure_detail: str | None = None

    @property
    def nvidia_cuda_build(self) -> bool:
        return (
            self.state == "completed"
            and self.cuda_built is True
            and bool(self.cuda_version)
            and not self.hip_version
        )


@dataclass(frozen=True)
class RuntimeReport:
    python_version: str
    python_executable: str
    architecture_bits: int
    in_virtual_environment: bool
    python_compatible: bool
    dependency_checks: tuple[DependencyCheck, ...]
    torch_probe: TorchProbeResult | None = None
    opencv_conflict: str | None = None

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
            and self.cuda_ready
        )

    @property
    def cuda_ready(self) -> bool:
        required = any(
            check.dependency.requires_cuda and check.dependency.required
            for check in self.dependency_checks
        )
        return not required or bool(
            self.torch_probe
            and self.torch_probe.nvidia_cuda_build
            and self.torch_probe.calculation_succeeded
        )

    @property
    def has_cuda_package_problem(self) -> bool:
        return any(
            check.dependency.requires_cuda
            and check.dependency.required
            and not check.compatible
            for check in self.dependency_checks
        )

    @property
    def packages_to_install(self) -> tuple[Dependency, ...]:
        return tuple(
            check.dependency
            for check in self.dependency_checks
            if check.dependency.required and not check.compatible
        )

    @property
    def cellpose_missing(self) -> bool:
        return any(
            check.dependency.distribution == "cellpose" and not check.compatible
            for check in self.dependency_checks
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
        cellpose_dependencies = tuple(raw.get("cellpose_dependencies", ()))
    except (KeyError, TypeError) as error:
        raise BootstrapError(
            f"Dependency manifest {manifest_path} has an invalid schema: {error}"
        ) from error

    if not packages:
        raise BootstrapError("The dependency manifest does not list any packages.")
    if any(type(package.requires_cuda) is not bool for package in packages):
        raise BootstrapError("Dependency manifest requires_cuda values must be booleans.")
    if any(type(spec) is not str for spec in cellpose_dependencies):
        raise BootstrapError("Dependency manifest cellpose_dependencies must be strings.")
    return BootstrapConfiguration(
        python=python,
        packages=packages,
        cellpose_dependencies=cellpose_dependencies,
    )


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


def opencv_conflict(
    version_getter: Callable[[str], str] = importlib.metadata.version,
) -> str | None:
    """Describe a headless OpenCV installed beside the GUI build, if any.

    Both distributions install the same cv2 module, so whichever was installed last
    silently replaces the other's files.
    """

    try:
        gui = version_getter("opencv-python")
        headless = version_getter("opencv-python-headless")
    except importlib.metadata.PackageNotFoundError:
        return None
    return (
        f"opencv-python-headless {headless} is installed beside opencv-python {gui} "
        "and overwrites its cv2 module. Run: python -m pip uninstall -y "
        "opencv-python-headless, then python -m pip install --force-reinstall "
        f"--no-deps opencv-python=={gui}"
    )


def inspect_runtime(
    configuration: BootstrapConfiguration,
    *,
    python_executable: Path | str = sys.executable,
    torch_probe: TorchProbeResult | None = None,
) -> RuntimeReport:
    python_version = platform.python_version()
    try:
        python_compatible = version_is_compatible(
            python_version,
            configuration.python.minimum,
            configuration.python.maximum_exclusive,
        )
    except ValueError:
        python_compatible = False

    checks = [
        inspect_dependency(dependency) for dependency in configuration.packages
    ]
    cuda_dependency = next(
        (
            dependency
            for dependency in configuration.packages
            if dependency.requires_cuda and dependency.required
        ),
        None,
    )
    if cuda_dependency is not None:
        check_index = next(
            index
            for index, check in enumerate(checks)
            if check.dependency == cuda_dependency
        )
        package_check = checks[check_index]
        if package_check.installed and package_check.compatible:
            torch_probe = torch_probe or probe_torch(python_executable)
            if torch_probe.nvidia_cuda_build is False:
                details = (
                    "PyTorch build does not provide NVIDIA CUDA; install a CUDA-enabled NVIDIA build"
                    if torch_probe.state == "completed" and not torch_probe.hip_version
                    else "ROCm PyTorch build is not supported; install an NVIDIA CUDA-enabled build"
                    if torch_probe.state == "completed" and torch_probe.hip_version
                    else f"CUDA package probe failed ({torch_probe.state}): "
                    f"{torch_probe.failure_detail or 'no usable NVIDIA CUDA build was reported'}"
                )
                if torch_probe.state == "completed":
                    checks[check_index] = DependencyCheck(
                        package_check.dependency,
                        package_check.installed,
                        False,
                        package_check.version,
                        details,
                    )

    return RuntimeReport(
        python_version=python_version,
        python_executable=str(python_executable),
        architecture_bits=struct.calcsize("P") * 8,
        in_virtual_environment=sys.prefix != sys.base_prefix,
        python_compatible=python_compatible,
        dependency_checks=tuple(checks),
        torch_probe=torch_probe,
        opencv_conflict=opencv_conflict(),
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
        detail = check.detail
        if marker == "OPTIONAL":
            detail += f"; {check.dependency.purpose}"
        lines.append(
            f"  [{marker}] {check.dependency.distribution} {version} - {detail}"
        )
    if report.opencv_conflict:
        lines.append(f"Conflict: {report.opencv_conflict}")
    if any(check.dependency.requires_cuda for check in report.dependency_checks):
        probe = report.torch_probe
        if probe is None:
            lines.append("CUDA device execution: NOT CHECKED (PyTorch is missing or out of range)")
            torch_check = next(
                (
                    check
                    for check in report.dependency_checks
                    if check.dependency.requires_cuda
                ),
                None,
            )
            if torch_check is not None and not torch_check.compatible:
                lines.append(
                    "Action: install an in-range NVIDIA CUDA PyTorch build from the "
                    "configured CUDA wheel source; see the operator guide for CUDA 12.6/13.2 options."
                )
        else:
            build = (
                "NVIDIA CUDA build"
                if probe.nvidia_cuda_build
                else "CPU-only or unsupported PyTorch build"
                if probe.state == "completed"
                else f"probe failed ({probe.state})"
            )
            lines.append(
                f"PyTorch build: {build}; version={probe.torch_version or 'unknown'}; "
                f"module={probe.module_location or 'unknown'}; "
                f"cuda_built={probe.cuda_built}; cuda_version={probe.cuda_version}; "
                f"hip_version={probe.hip_version}"
            )
            lines.append(
                f"CUDA device execution: {'PASS' if probe.calculation_succeeded else 'FAIL'}; "
                f"available={probe.cuda_available}; GPU={probe.gpu_name or 'unavailable'}"
            )
            if probe.failure_detail:
                lines.append(f"CUDA failure detail: {probe.failure_detail}")
            if probe.state == "completed" and not probe.nvidia_cuda_build:
                lines.append(
                    "Action: install a supported NVIDIA CUDA PyTorch build; see the "
                    "Windows Python 3.15 CUDA 12.6/13.2 options in the operator guide."
                )
            elif probe.state != "completed":
                lines.append(
                    "Action: resolve the PyTorch probe failure for this interpreter "
                    "before launching; inspect the reported import, DLL, crash, or timeout detail."
                )
            elif not probe.calculation_succeeded:
                lines.append(
                    "Action: check the NVIDIA GPU, driver compatibility, and CUDA "
                    "runtime; CUDA 12.6 does not cover Blackwell (use the CUDA 13.2 "
                    "option there). Package reinstallation will not fix an unavailable device."
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
    no_deps: bool = False,
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
    if no_deps:
        command.append("--no-deps")
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


def installed_version(python_executable: Path | str, distribution: str) -> str | None:
    """Ask the selected interpreter for an installed distribution's version."""

    script = (
        "import importlib.metadata, sys\n"
        "try:\n"
        "    print(importlib.metadata.version(sys.argv[1]))\n"
        "except importlib.metadata.PackageNotFoundError:\n"
        "    pass\n"
    )
    try:
        result = subprocess.run(
            (str(python_executable), "-c", script, distribution),
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise BootstrapError(
            f"Could not read the {distribution} version from {python_executable}: {error}"
        ) from error
    if result.returncode != 0:
        raise BootstrapError(
            f"Could not read the {distribution} version from {python_executable}: "
            f"{result.stderr.strip() or f'exit code {result.returncode}'}"
        )
    return result.stdout.strip() or None


def pytorch_index_url(torch_version: str) -> str | None:
    """The PyTorch wheel index named by a build's local label, such as cu126."""

    _, _, label = torch_version.partition("+")
    return f"https://download.pytorch.org/whl/{label}" if label else None


def cellpose_install_commands(
    python_executable: Path | str,
    configuration: BootstrapConfiguration,
    torch_version: str,
    *,
    root: Path,
    offline: bool,
) -> list[list[str]]:
    """Pip commands that add Cellpose-SAM without replacing PyTorch or OpenCV.

    torchvision comes from the installed PyTorch build's index, and pinning that
    build in the same request makes pip choose the torchvision release made for it.
    Cellpose goes last and without dependencies: its list names
    opencv-python-headless, which would replace the cv2 module of opencv-python.
    Its other dependencies are the manifest's pins.
    """

    cellpose = configuration.cellpose
    if cellpose is None:
        raise BootstrapError("The dependency manifest does not list cellpose.")
    wheelhouse = root / "wheels"
    commands = [
        build_pip_command(
            python_executable,
            ("torchvision", f"torch=={torch_version}"),
            wheelhouse=wheelhouse,
            offline=offline,
            index_url=pytorch_index_url(torch_version),
        )
    ]
    if configuration.cellpose_dependencies:
        commands.append(
            build_pip_command(
                python_executable,
                configuration.cellpose_dependencies,
                wheelhouse=wheelhouse,
                offline=offline,
            )
        )
    commands.append(
        build_pip_command(
            python_executable,
            (cellpose.install_spec,),
            wheelhouse=wheelhouse,
            offline=offline,
            no_deps=True,
        )
    )
    return commands


def install_cellpose(
    python_executable: Path | str,
    configuration: BootstrapConfiguration,
    *,
    root: Path,
    offline: bool,
) -> None:
    torch_version = installed_version(python_executable, "torch")
    if torch_version is None:
        raise BootstrapError("PyTorch is not installed, so Cellpose-SAM cannot be added.")
    for command in cellpose_install_commands(
        python_executable,
        configuration,
        torch_version,
        root=root,
        offline=offline,
    ):
        run_checked(command, description="install Cellpose-SAM")


def probe_torch(
    python_executable: Path | str,
    *,
    timeout: float = 30,
) -> TorchProbeResult:
    """Inspect and exercise CUDA in the selected interpreter, isolated in a child."""

    script = """
import json
import torch

result = {
    "torch_version": str(torch.__version__),
    "module_location": str(torch.__file__),
    "cuda_built": bool(torch.backends.cuda.is_built()),
    "cuda_version": torch.version.cuda,
    "hip_version": torch.version.hip,
    "cuda_available": False,
    "gpu_name": None,
    "calculation_succeeded": False,
    "failure_detail": None,
}
try:
    result["cuda_available"] = bool(torch.cuda.is_available())
    if not result["cuda_available"]:
        result["failure_detail"] = "torch.cuda.is_available() returned false"
    else:
        result["gpu_name"] = str(torch.cuda.get_device_name(0))
        values = torch.tensor([1.0, 2.0, 3.0], device="cuda")
        total = ((values + 2.0) * 3.0).sum().item()
        torch.cuda.synchronize()
        if total != 36.0:
            raise RuntimeError(f"CUDA calculation returned {total}, expected 36.0")
        result["calculation_succeeded"] = True
except Exception as error:
    result["failure_detail"] = f"{type(error).__name__}: {error}"
print(json.dumps(result, allow_nan=False))
"""
    try:
        result = subprocess.run(
            (str(python_executable), "-c", script),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        detail = f"CUDA probe exceeded its {timeout:g}s timeout"
        if error.stderr:
            detail += f": {str(error.stderr)[-1000:]}"
        return TorchProbeResult(str(python_executable), "timeout", failure_detail=detail)
    except OSError as error:
        return TorchProbeResult(
            str(python_executable), "launch_failed", failure_detail=str(error)
        )
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()
        lowered_stderr = result.stderr.lower()
        state = (
            "import_failed"
            if any(
                marker in lowered_stderr
                for marker in (
                    "modulenotfounderror",
                    "importerror",
                    "dll load failed",
                    "specified module could not be found",
                )
            )
            else "abnormal_exit"
        )
        return TorchProbeResult(
            str(python_executable),
            state,
            failure_detail=detail[-1] if detail else f"exit code {result.returncode}",
        )
    try:
        payload = json.loads(result.stdout)
        required = {
            "torch_version",
            "module_location",
            "cuda_built",
            "cuda_version",
            "hip_version",
            "cuda_available",
            "gpu_name",
            "calculation_succeeded",
            "failure_detail",
        }
        if not isinstance(payload, dict) or set(payload) != required:
            raise ValueError("unexpected result fields")
        if any(type(payload[key]) is not bool for key in ("cuda_built", "cuda_available", "calculation_succeeded")):
            raise ValueError("boolean probe fields have invalid types")
        if not isinstance(payload["torch_version"], str) or not isinstance(payload["module_location"], str):
            raise ValueError("PyTorch version or module location is invalid")
        if not payload["torch_version"] or not payload["module_location"]:
            raise ValueError("PyTorch version or module location is empty")
        for key in ("cuda_version", "hip_version", "gpu_name", "failure_detail"):
            if payload[key] is not None and not isinstance(payload[key], str):
                raise ValueError(f"{key} has an invalid type")
        if payload["calculation_succeeded"] and not payload["cuda_available"]:
            raise ValueError("CUDA calculation succeeded while no CUDA device was reported")
        if payload["cuda_available"] and not payload["gpu_name"]:
            raise ValueError("CUDA device is available but its reported name is missing")
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        return TorchProbeResult(
            str(python_executable),
            "invalid_response",
            failure_detail=f"invalid CUDA probe output: {error}; output={result.stdout[-1000:]!r}",
        )
    return TorchProbeResult(
        str(python_executable),
        "completed",
        **payload,
    )


def require_runtime_ready(python_executable: Path | str = sys.executable) -> RuntimeReport:
    """Raise an actionable error unless the production runtime passes CUDA execution."""

    root = Path(__file__).resolve().parents[1]
    configuration = load_configuration(root)
    report = inspect_runtime(configuration, python_executable=python_executable)
    if not report.ready:
        raise BootstrapError(format_report(report, configuration))
    return report
