from __future__ import annotations

import importlib.metadata
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from seedvision.bootstrap import (
    BootstrapError,
    Dependency,
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


class VersionTests(unittest.TestCase):
    def test_version_key_accepts_common_suffixes(self) -> None:
        self.assertEqual(version_key("2.13.0+cu130"), (2, 13, 0))
        self.assertEqual(version_key("6.11"), (6, 11, 0))

    def test_compatibility_uses_inclusive_minimum_and_exclusive_maximum(self) -> None:
        self.assertTrue(version_is_compatible("2.0.0", "2.0.0", "3.0.0"))
        self.assertTrue(version_is_compatible("2.99.1", "2.0.0", "3.0.0"))
        self.assertFalse(version_is_compatible("1.99.9", "2.0.0", "3.0.0"))
        self.assertFalse(version_is_compatible("3.0.0", "2.0.0", "3.0.0"))

    def test_invalid_version_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            version_key("unknown")


class DependencyInspectionTests(unittest.TestCase):
    def test_missing_import_requires_installation(self) -> None:
        check = inspect_dependency(
            dependency(),
            module_finder=lambda _: None,
            version_getter=lambda _: "2.4.0",
        )
        self.assertFalse(check.installed)
        self.assertFalse(check.compatible)

    def test_compatible_supplied_package_is_kept(self) -> None:
        check = inspect_dependency(
            dependency(),
            module_finder=lambda _: object(),
            version_getter=lambda _: "2.7.1",
        )
        self.assertTrue(check.installed)
        self.assertTrue(check.compatible)

    def test_old_package_requires_installation(self) -> None:
        check = inspect_dependency(
            dependency(),
            module_finder=lambda _: object(),
            version_getter=lambda _: "1.9.9",
        )
        self.assertTrue(check.installed)
        self.assertFalse(check.compatible)

    def test_centrally_supplied_import_without_pip_metadata_is_accepted(self) -> None:
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


class PipCommandTests(unittest.TestCase):
    def test_online_command_can_prefer_local_wheels(self) -> None:
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

    def test_offline_command_disables_package_indexes(self) -> None:
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

    def test_online_command_can_select_the_cuda_wheel_index(self) -> None:
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

    def test_offline_command_rejects_an_empty_wheelhouse(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaises(BootstrapError):
                build_pip_command(
                    "python.exe",
                    ("Example==2.4.0",),
                    wheelhouse=Path(temporary_directory),
                    offline=True,
                )

    def test_online_install_separates_default_and_cuda_indexes(self) -> None:
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


class ManifestTests(unittest.TestCase):
    def test_repository_manifest_loads(self) -> None:
        root = Path(__file__).resolve().parents[1]
        configuration = load_configuration(root)
        distributions = {item.distribution for item in configuration.packages}
        self.assertEqual(
            distributions,
            {"PySide6", "numpy", "opencv-python", "torch"},
        )
        self.assertEqual(configuration.python.recommended, "3.12")
        self.assertEqual(configuration.python.minimum, "3.12.0")
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
