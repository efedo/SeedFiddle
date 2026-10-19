from __future__ import annotations

from tests.functional_units import functional_units

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from seedvision.benchmarking import reference_runner
from seedvision.benchmarking.reference_setup import (
    MAX_PACKAGES,
    MAX_SOURCE_FILES,
    ReferenceSetupError,
    _SourceRecord,
    _fetch_source_atomically,
    _iter_bounded_chunks,
    _load_source_registry,
    _parse_locked_packages,
    _safe_child,
    _source_audit_projection,
    _validate_source_records,
    inspect_reference_setup,
    materialize_reference_setup,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "peer_reference_sources.json"
LOCK = ROOT / "config" / "plantcv-reference-requirements.txt"
ARTIFACTS = ROOT / "artifacts" / "reference-sources"
PINNED_COMMIT = "77e3322fd0cd5a9bef18bb99402f20c8f2db87a9"


@functional_units
class ReferenceSetupTests(unittest.TestCase):
    def check_test_reconstructed_manifests_match_runner_hash_pins_and_all_locked_packages(self) -> None:
        config, _ = _load_source_registry(CONFIG)
        records = _validate_source_records(config)
        self.assertEqual(len(records), MAX_SOURCE_FILES)
        projection = _source_audit_projection(config, records)
        source_raw = (json.dumps(projection, ensure_ascii=False, indent=2, allow_nan=False) + "\n").replace("\n", "\r\n").encode("utf-8")
        self.assertEqual(hashlib.sha256(source_raw).hexdigest(), reference_runner.PINNED_SOURCE_AUDIT_SHA256)
        self.assertEqual(projection["reference"], "plantcv")
        self.assertEqual(projection["commit"], PINNED_COMMIT)
        self.assertEqual(set(projection), {"reference", "version", "commit", "retrieved_date", "files"})

        lock_raw = LOCK.read_bytes()
        locked = _parse_locked_packages(lock_raw)
        self.assertEqual(len(locked), MAX_PACKAGES)
        runner_locked = reference_runner._parse_requirements_lock(LOCK)
        self.assertEqual(
            {name: (version, digest) for name, (_, version, digest) in locked.items()},
            {name: (version, digest) for name, (version, digest) in runner_locked.items()},
        )
        environment = config["environment"]
        inventory = {item["name"]: item["version"] for item in environment["packages"]}
        normalized_inventory = {reference_runner._canonical_name(name): version for name, version in inventory.items()}
        self.assertEqual(normalized_inventory, {name: version for name, (version, _) in runner_locked.items()})
        environment_raw = (json.dumps(inventory, ensure_ascii=False, indent=2, allow_nan=False) + "\n").replace("\n", "\r\n").encode("utf-8")
        self.assertEqual(len(inventory), 67)
        self.assertEqual(hashlib.sha256(environment_raw).hexdigest(), "92532d707ee1895e8e3a9c26f218fed9c51d4510b8cbc6a80a545cdbb9b05162")

    def check_test_materialization_is_idempotent_and_missing_sources_stay_explicit_without_fetch(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            target = Path(name) / "reference-sources"
            first = materialize_reference_setup(target, repository_root=ROOT)
            source_metadata = first.source_audit_path.read_bytes()
            environment_metadata = first.environment_manifest_path.read_bytes()
            self.assertTrue(first.metadata_available)
            self.assertEqual(first.source_file_count, 15)
            self.assertEqual(first.package_count, 67)
            self.assertEqual(first.unavailable_sources, tuple(record["path"] for record in json.loads(CONFIG.read_text(encoding="utf-8"))["files"]))
            self.assertIn(first.wheel_status, {"verified", "missing"})
            self.assertEqual(first.retained_wheel_available, first.wheel_status == "verified")
            self.assertFalse(first.provisioning_prerequisites_available)

            second = materialize_reference_setup(target, repository_root=ROOT)
            self.assertEqual(source_metadata, second.source_audit_path.read_bytes())
            self.assertEqual(environment_metadata, second.environment_manifest_path.read_bytes())
            self.assertEqual(first.unavailable_sources, second.unavailable_sources)
            self.assertTrue(second.metadata_available)

    def check_test_status_reports_missing_retained_flyr_wheel_for_fresh_project_root(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            fresh_root = Path(name)
            status = materialize_reference_setup(
                fresh_root / "artifacts" / "reference-sources",
                repository_root=fresh_root,
                config_path=CONFIG,
                requirements_lock_path=LOCK,
            )
            self.assertEqual(status.wheel_status, "missing")
            self.assertFalse(status.retained_wheel_available)
            self.assertFalse(status.provisioning_prerequisites_available)
            self.assertEqual(len(status.unavailable_sources), 15)

    def check_test_changed_lock_or_existing_source_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            temp = Path(name)
            altered_lock = temp / "requirements.txt"
            raw = LOCK.read_bytes()
            altered_lock.write_bytes(raw.replace(b"altair==5.5.0", b"altair==5.5.1", 1))
            target = temp / "target"
            with self.assertRaisesRegex(ReferenceSetupError, "lock has changed"):
                materialize_reference_setup(
                    target, repository_root=ROOT, requirements_lock_path=altered_lock,
                )
            self.assertFalse(target.exists())

            target = temp / "changed-source-target"
            wrong_source = target / "plantcv-v4.11.3" / "LICENSE"
            wrong_source.parent.mkdir(parents=True)
            wrong_source.write_bytes(b"changed source bytes")
            with self.assertRaisesRegex(ReferenceSetupError, "does not match its pinned bytes"):
                materialize_reference_setup(target, repository_root=ROOT)
            self.assertEqual(wrong_source.read_bytes(), b"changed source bytes")
            self.assertFalse((target / "plantcv-source-audit.json").exists())

    def check_test_source_inventory_rejects_traversal_duplicates_and_unpinned_urls(self) -> None:
        config, _ = _load_source_registry(CONFIG)
        for mutation, message in (
            (lambda files: files[0].update(path="../outside"), "unsafe path"),
            (lambda files: files[1].update(path=files[0]["path"]), "Duplicate pinned source path"),
            (lambda files: files[0].update(url="https://example.invalid/source.py"), "not pinned"),
        ):
            altered = json.loads(json.dumps(config))
            mutation(altered["files"])
            with self.subTest(message=message), self.assertRaisesRegex(ReferenceSetupError, message):
                _validate_source_records(altered)

        with tempfile.TemporaryDirectory() as name:
            root = Path(name) / "root"
            root.mkdir()
            with self.assertRaisesRegex(ReferenceSetupError, "unsafe path"):
                _safe_child(root, "../outside", label="test target")

    def check_test_fetch_callback_is_explicit_pinned_bounded_and_hash_verified(self) -> None:
        content = b"abcdef"
        record = _SourceRecord(
            path="small/fake.py",
            url=f"https://raw.githubusercontent.com/danforthcenter/plantcv/{PINNED_COMMIT}/small/fake.py",
            sha256=hashlib.sha256(content).hexdigest(),
            byte_count=len(content),
        )
        with tempfile.TemporaryDirectory() as name:
            target = Path(name) / "source" / "small" / "fake.py"
            seen: list[tuple[str, int]] = []

            def fetch(url: str, maximum: int):
                seen.append((url, maximum))
                return iter((b"abc", b"def"))

            self.assertTrue(_fetch_source_atomically(target, record, fetch))
            self.assertEqual(target.read_bytes(), content)
            self.assertEqual(seen, [(record.url, 6)])
            self.assertFalse(_fetch_source_atomically(target, record, lambda *_: (_ for _ in ()).throw(AssertionError("existing source should not fetch"))))

        bad_record = _SourceRecord(record.path, record.url, record.sha256, record.byte_count)
        with tempfile.TemporaryDirectory() as name:
            target = Path(name) / "bad.py"
            with self.assertRaisesRegex(ReferenceSetupError, "exceeds its declared byte limit"):
                _fetch_source_atomically(target, bad_record, lambda *_: (part for part in (b"abcdef", b"g")))
            self.assertFalse(target.exists())
            self.assertEqual(list(target.parent.iterdir()), [])

    def check_test_cancelled_fetch_cleans_partial_temporary_output(self) -> None:
        content = b"abcdef"
        record = _SourceRecord(
            path="small/fake.py",
            url=f"https://raw.githubusercontent.com/danforthcenter/plantcv/{PINNED_COMMIT}/small/fake.py",
            sha256=hashlib.sha256(content).hexdigest(),
            byte_count=len(content),
        )

        def interrupted(*_):
            yield b"abc"
            raise KeyboardInterrupt

        with tempfile.TemporaryDirectory() as name:
            target = Path(name) / "source" / "small" / "fake.py"
            with self.assertRaises(KeyboardInterrupt):
                _fetch_source_atomically(target, record, interrupted)
            self.assertFalse(target.exists())
            self.assertEqual(list(target.parent.iterdir()), [])

    def check_test_fetch_chunk_bounds_precede_copy_and_reject_nonprogress(self) -> None:
        large = bytearray(16 * 1024 * 1024)
        chunk = memoryview(large)
        with mock.patch("seedvision.benchmarking.reference_setup._copy_fetch_chunk") as copy_chunk:
            with self.assertRaisesRegex(ReferenceSetupError, "exceeds its declared byte limit"):
                list(_iter_bounded_chunks((chunk,), maximum=3, label="probe"))
            copy_chunk.assert_not_called()

        consumed = 0

        def endless_empty_chunks():
            nonlocal consumed
            while True:
                consumed += 1
                yield b""

        with self.assertRaisesRegex(ReferenceSetupError, "made no byte progress"):
            list(_iter_bounded_chunks(endless_empty_chunks(), maximum=3, label="probe"))
        self.assertEqual(consumed, 33)

        with mock.patch("seedvision.benchmarking.reference_setup.MAX_FETCH_CHUNKS", 3):
            with self.assertRaisesRegex(ReferenceSetupError, "chunk limit"):
                list(_iter_bounded_chunks((b"a" for _ in range(10)), maximum=10, label="probe"))

    def check_test_generated_metadata_passes_runner_source_validator_when_committed_bytes_exist(self) -> None:
        pinned_source_root = ARTIFACTS / "plantcv-v4.11.3"
        config, _ = _load_source_registry(CONFIG)
        if not all((pinned_source_root / item["path"]).is_file() for item in config["files"]):
            self.skipTest("ignored pinned source bytes are not present in this checkout")
        with tempfile.TemporaryDirectory() as name:
            fresh_root = Path(name) / "fresh-project"
            target = fresh_root / "artifacts" / "reference-sources"
            sources_by_url = {
                item["url"]: pinned_source_root / item["path"]
                for item in config["files"]
            }

            def local_pinned_fetch(url: str, byte_limit: int):
                self.assertIn(url, sources_by_url)
                raw = sources_by_url[url].read_bytes()
                self.assertLessEqual(len(raw), byte_limit)
                return (raw[index:index + 4096] for index in range(0, len(raw), 4096))

            status = materialize_reference_setup(
                target,
                repository_root=fresh_root,
                config_path=CONFIG,
                requirements_lock_path=LOCK,
                fetch_sources=True,
                fetcher=local_pinned_fetch,
            )
            self.assertTrue(status.sources_available)
            self.assertTrue(status.metadata_available)
            self.assertEqual(status.wheel_status, "missing")
            self.assertFalse(status.provisioning_prerequisites_available)
            _, source_digest = reference_runner._verify_source_manifest(status.source_audit_path, status.source_root)
            self.assertEqual(source_digest, reference_runner._verify_source_manifest(ARTIFACTS / "plantcv-source-audit.json", pinned_source_root)[1])
            expected = reference_runner._build_expected_identity(
                source_manifest_path=status.source_audit_path,
                source_root=status.source_root,
                environment_manifest_path=status.environment_manifest_path,
                requirements_lock_path=LOCK,
            )
            self.assertEqual(expected["source_audit_sha256"], reference_runner.PINNED_SOURCE_AUDIT_SHA256)
            self.assertEqual(len(expected["expected_distributions"]), 67)

    def test_generated_metadata_passes_runner_source_validator_when_committed_bytes_exist(self):
        self.scenario(self.check_test_generated_metadata_passes_runner_source_validator_when_committed_bytes_exist, labelled=False)

    def test_workflow_lifecycle(self):
        """3 related ReferenceSetupTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_reconstructed_manifests_match_runner_hash_pins_and_all_locked_packages)
        self.scenario(self.check_test_changed_lock_or_existing_source_is_never_overwritten)
        self.scenario(self.check_test_fetch_callback_is_explicit_pinned_bounded_and_hash_verified)

    def test_validation_and_recovery(self):
        """5 related ReferenceSetupTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_materialization_is_idempotent_and_missing_sources_stay_explicit_without_fetch)
        self.scenario(self.check_test_status_reports_missing_retained_flyr_wheel_for_fresh_project_root)
        self.scenario(self.check_test_source_inventory_rejects_traversal_duplicates_and_unpinned_urls)
        self.scenario(self.check_test_cancelled_fetch_cleans_partial_temporary_output)
        self.scenario(self.check_test_fetch_chunk_bounds_precede_copy_and_reject_nonprogress)


if __name__ == "__main__":
    unittest.main()
