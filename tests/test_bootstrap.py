from __future__ import annotations

from tests.functional_units import functional_units

import importlib.metadata
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from seedvision.bootstrap import (
    BootstrapError,
    Dependency,
    RuntimeReport,
    build_pip_command,
    install_dependencies,
    inspect_dependency,
    load_configuration,
    version_is_compatible,
    version_key,
)


def dependency() -> Dependency:
    return Dependency(
        distribution="Example",
        import_name="example",
        install_spec="Example==2.4.0",
        minimum="2.0.0",
        maximum_exclusive="3.0.0",
        purpose="test dependency",
    )


@functional_units
class VersionTests(unittest.TestCase):
    def check_test_version_key_accepts_common_suffixes(self) -> None:
        self.assertEqual(version_key("2.13.0+cu130"), (2, 13, 0))
        self.assertEqual(version_key("6.11"), (6, 11, 0))

    def check_test_compatibility_uses_inclusive_minimum_and_exclusive_maximum(self) -> None:
        self.assertTrue(version_is_compatible("2.0.0", "2.0.0", "3.0.0"))
        self.assertTrue(version_is_compatible("2.99.1", "2.0.0", "3.0.0"))
        self.assertFalse(version_is_compatible("1.99.9", "2.0.0", "3.0.0"))
        self.assertFalse(version_is_compatible("3.0.0", "2.0.0", "3.0.0"))

    def check_test_invalid_version_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            version_key("unknown")

    def test_workflow_lifecycle(self):
        """2 related VersionTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_version_key_accepts_common_suffixes)
        self.scenario(self.check_test_compatibility_uses_inclusive_minimum_and_exclusive_maximum)

    def test_validation_and_recovery(self):
        """1 related VersionTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_invalid_version_is_rejected)


@functional_units
class DependencyInspectionTests(unittest.TestCase):
    def check_test_missing_import_requires_installation(self) -> None:
        check = inspect_dependency(
            dependency(),
            module_finder=lambda _: None,
            version_getter=lambda _: "2.4.0",
        )
        self.assertFalse(check.installed)
        self.assertFalse(check.compatible)

    def check_test_compatible_supplied_package_is_kept(self) -> None:
        check = inspect_dependency(
            dependency(),
            module_finder=lambda _: object(),
            version_getter=lambda _: "2.7.1",
        )
        self.assertTrue(check.installed)
        self.assertTrue(check.compatible)

    def check_test_old_package_requires_installation(self) -> None:
        check = inspect_dependency(
            dependency(),
            module_finder=lambda _: object(),
            version_getter=lambda _: "1.9.9",
        )
        self.assertTrue(check.installed)
        self.assertFalse(check.compatible)

    def check_test_centrally_supplied_import_without_pip_metadata_is_accepted(self) -> None:
        def no_metadata(_: str) -> str:
            raise importlib.metadata.PackageNotFoundError("Example")

        check = inspect_dependency(
            dependency(),
            module_finder=lambda _: object(),
            version_getter=no_metadata,
        )
        self.assertTrue(check.installed)
        self.assertTrue(check.compatible)
        self.assertIsNone(check.version)

    def test_validation_and_recovery(self):
        """1 related DependencyInspectionTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_missing_import_requires_installation)

    def test_workflow_lifecycle(self):
        """3 related DependencyInspectionTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_compatible_supplied_package_is_kept)
        self.scenario(self.check_test_old_package_requires_installation)
        self.scenario(self.check_test_centrally_supplied_import_without_pip_metadata_is_accepted)


@functional_units
class PipCommandTests(unittest.TestCase):
    def check_test_online_command_can_prefer_local_wheels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            wheelhouse = Path(temporary_directory)
            (wheelhouse / "example-2.4.0-py3-none-any.whl").touch()
            command = build_pip_command(
                "python.exe",
                ("Example==2.4.0",),
                wheelhouse=wheelhouse,
                offline=False,
            )
        self.assertIn("--find-links", command)
        self.assertNotIn("--no-index", command)
        self.assertEqual(command[-1], "Example==2.4.0")

    def check_test_offline_command_disables_package_indexes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            wheelhouse = Path(temporary_directory)
            (wheelhouse / "example-2.4.0-py3-none-any.whl").touch()
            command = build_pip_command(
                "python.exe",
                ("Example==2.4.0",),
                wheelhouse=wheelhouse,
                offline=True,
            )
        self.assertIn("--no-index", command)
        self.assertIn("--find-links", command)

    def check_test_online_command_can_select_the_cuda_wheel_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            command = build_pip_command(
                "python.exe",
                ("torch==2.13.0",),
                wheelhouse=Path(temporary_directory),
                offline=False,
                index_url="https://download.pytorch.org/whl/cu126",
            )
        index_position = command.index("--index-url")
        self.assertEqual(
            command[index_position + 1],
            "https://download.pytorch.org/whl/cu126",
        )

    def check_test_offline_command_rejects_an_empty_wheelhouse(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaises(BootstrapError):
                build_pip_command(
                    "python.exe",
                    ("Example==2.4.0",),
                    wheelhouse=Path(temporary_directory),
                    offline=True,
                )

    def check_test_online_install_separates_default_and_cuda_indexes(self) -> None:
        standard = dependency()
        torch = Dependency(
            distribution="torch",
            import_name="torch",
            install_spec="torch==2.13.0",
            minimum="2.5.0",
            maximum_exclusive="3.0.0",
            purpose="inference",
            index_url="https://download.pytorch.org/whl/cu126",
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            with patch("seedvision.bootstrap.run_checked") as run_checked:
                install_dependencies(
                    "python.exe",
                    (standard, torch),
                    root=Path(temporary_directory),
                    offline=False,
                )
        self.assertEqual(run_checked.call_count, 2)
        first_command = run_checked.call_args_list[0].args[0]
        second_command = run_checked.call_args_list[1].args[0]
        self.assertNotIn("--index-url", first_command)
        self.assertIn("Example==2.4.0", first_command)
        self.assertIn("--index-url", second_command)
        self.assertIn("torch==2.13.0", second_command)

    def test_workflow_lifecycle(self):
        """4 related PipCommandTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_online_command_can_prefer_local_wheels)
        self.scenario(self.check_test_offline_command_disables_package_indexes)
        self.scenario(self.check_test_online_command_can_select_the_cuda_wheel_index)
        self.scenario(self.check_test_online_install_separates_default_and_cuda_indexes)

    def test_validation_and_recovery(self):
        """1 related PipCommandTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_offline_command_rejects_an_empty_wheelhouse)


class ManifestTests(unittest.TestCase):
    def test_repository_manifest_loads(self) -> None:
        root = Path(__file__).resolve().parents[1]
        configuration = load_configuration(root)
        distributions = {item.distribution for item in configuration.packages}
        self.assertEqual(
            distributions,
            {"PySide6", "numpy", "opencv-python", "matplotlib", "torch", "cellpose"},
        )
        cellpose = next(item for item in configuration.packages if item.distribution == "cellpose")
        self.assertFalse(cellpose.required)
        self.assertEqual(configuration.python.recommended, "3.12")
        self.assertEqual(configuration.python.minimum, "3.12.0")
        self.assertEqual(configuration.python.maximum_exclusive, "3.16.0")
        torch = next(item for item in configuration.packages if item.distribution == "torch")
        self.assertEqual(torch.index_url, "https://download.pytorch.org/whl/cu126")
        self.assertTrue(torch.required)

        requirement_lines = {
            line.strip()
            for line in (root / "requirements-runtime.txt").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        self.assertEqual(
            requirement_lines,
            {item.install_spec for item in configuration.required_packages},
        )

        self.assertTrue(version_is_compatible(
            "3.15.2",
            configuration.python.minimum,
            configuration.python.maximum_exclusive,
        ))
        self.assertFalse(version_is_compatible(
            "3.16.0",
            configuration.python.minimum,
            configuration.python.maximum_exclusive,
        ))
        self.assertFalse(version_is_compatible(
            "3.11.9",
            configuration.python.minimum,
            configuration.python.maximum_exclusive,
        ))

        installed_versions = {
            "PySide6": "6.12.0",
            "numpy": "2.5.4",
            "opencv-python": "5.0.0.93",
            "matplotlib": "3.11.2",
            "torch": "2.15.0",
            "cellpose": "4.2.1.1",
        }
        checks = tuple(
            inspect_dependency(
                item,
                module_finder=lambda _: object(),
                version_getter=installed_versions.__getitem__,
            )
            for item in configuration.packages
        )
        report = RuntimeReport(
            python_version="3.15.2",
            python_executable=sys.executable,
            architecture_bits=64,
            in_virtual_environment=True,
            python_compatible=True,
            dependency_checks=checks,
        )
        self.assertTrue(all(check.compatible for check in checks))
        self.assertEqual(report.packages_to_install, ())

        with tempfile.TemporaryDirectory() as temporary_directory:
            fixture_root = Path(temporary_directory) / "checkout"
            config_directory = fixture_root / "config"
            scripts_directory = fixture_root / "scripts"
            config_directory.mkdir(parents=True)
            scripts_directory.mkdir()
            lock_path = config_directory / "plantcv-reference-requirements.txt"
            config_path = config_directory / "peer_reference_sources.json"
            utility_path = scripts_directory / "repair_reference_lock.py"
            lock_source = (root / "config" / "plantcv-reference-requirements.txt").read_bytes()
            lock_lf = lock_source.replace(b"\r\n", b"\n")
            lock_crlf = lock_lf.replace(b"\n", b"\r\n")
            expected_sha = hashlib.sha256(lock_crlf).hexdigest()
            self.assertEqual(
                json.loads((root / "config" / "peer_reference_sources.json").read_text())[
                    "environment"
                ]["lockfile_sha256"],
                expected_sha,
            )
            config_path.write_bytes((root / "config" / "peer_reference_sources.json").read_bytes())
            utility_path.write_bytes((root / "scripts" / "repair_reference_lock.py").read_bytes())

            def run_utility(*arguments: str) -> subprocess.CompletedProcess[str]:
                return subprocess.run(
                    [sys.executable, str(utility_path), *arguments],
                    cwd=fixture_root.parent,
                    capture_output=True,
                    text=True,
                    check=False,
                )

            lock_path.write_bytes(lock_lf)
            before_check = lock_path.read_bytes()
            checked = run_utility()
            self.assertEqual(checked.returncode, 1, checked.stdout + checked.stderr)
            self.assertIn(hashlib.sha256(before_check).hexdigest(), checked.stdout)
            self.assertIn(expected_sha, checked.stdout)
            self.assertIn("CRLF line-ending conversion would resolve it", checked.stdout)
            self.assertEqual(lock_path.read_bytes(), before_check)

            repaired = run_utility("--repair")
            self.assertEqual(repaired.returncode, 0, repaired.stdout + repaired.stderr)
            self.assertEqual(lock_path.read_bytes(), lock_crlf)
            self.assertEqual(hashlib.sha256(lock_path.read_bytes()).hexdigest(), expected_sha)

            before_correct = lock_path.stat()
            unchanged = run_utility("--repair")
            self.assertEqual(unchanged.returncode, 0, unchanged.stdout + unchanged.stderr)
            self.assertIn("no change needed", unchanged.stdout)
            self.assertEqual(lock_path.stat().st_mtime_ns, before_correct.st_mtime_ns)

            altered = lock_crlf.replace(b"altair==5.5.0", b"altair==5.4.0", 1)
            lock_path.write_bytes(altered)
            rejected = run_utility("--repair")
            self.assertEqual(rejected.returncode, 1, rejected.stdout + rejected.stderr)
            self.assertIn("Refusing repair", rejected.stderr)
            self.assertEqual(lock_path.read_bytes(), altered)

            lock_path.unlink()
            missing = run_utility()
            self.assertEqual(missing.returncode, 1, missing.stdout + missing.stderr)
            self.assertIn("Reference lock error", missing.stderr)

    def test_trait_manifest_contains_agreed_species_and_damage_fields(self) -> None:
        root = Path(__file__).resolve().parents[1]
        traits = json.loads((root / "config" / "traits.json").read_text(encoding="utf-8"))
        species = {item["id"] for item in traits["species"]}
        self.assertEqual(
            species,
            {
                "soybean",
                "lupinus_mutabilis",
                "lupinus_polyphyllus",
                "lupinus_mexicanus",
            },
        )
        self.assertEqual(
            set(traits["damage_flags"]),
            {
                "cracking",
                "missing_or_peeling_coat",
                "chipping_or_breakage",
                "discoloration",
            },
        )
        mutabilis = next(
            item for item in traits["species"]
            if item["id"] == "lupinus_mutabilis"
        )
        self.assertEqual(
            mutabilis["reference_seed_coat_patterns"],
            ["white", "banded_light", "banded_dark", "other"],
        )
        self.assertEqual(
            traits["reference_seed_conditions"],
            ["immature", "split", "wrinkled", "stained"],
        )
        self.assertTrue(traits["visible_face_only"])


if __name__ == "__main__":
    unittest.main()
