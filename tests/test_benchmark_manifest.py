from __future__ import annotations

from tests.functional_units import functional_units

from copy import deepcopy
from hashlib import sha256
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from seedvision.benchmarking.checkpoints import CoordinateFrame
from seedvision.benchmarking import manifest as manifest_module
from seedvision.benchmarking.manifest import (
    AccessPurpose,
    AccessRecord,
    BENCHMARK_MANIFEST_FORMAT,
    BENCHMARK_MANIFEST_VERSION,
    BenchmarkManifest,
    BenchmarkManifestError,
    BenchmarkSample,
    SplitRole,
    TruthIdentity,
    TruthState,
    audit_benchmark_manifest,
)


def _truth(sample_id: str) -> TruthIdentity:
    digest = sha256(sample_id.encode("utf-8")).hexdigest()
    return TruthIdentity(
        state=TruthState.AVAILABLE,
        annotation_sha256=digest,
        reviewed_regions_sha256=sha256((sample_id + ":regions").encode()).hexdigest(),
        eligibility_sha256=sha256((sample_id + ":eligible").encode()).hexdigest(),
        revision="review-r1",
        reviewer_ids=("reviewer_a", "reviewer_b"),
        adjudication_sha256=sha256((sample_id + ":adjudication").encode()).hexdigest(),
    )


def _sample(
    sample_id: str,
    role: SplitRole = SplitRole.DEVELOPMENT,
    *,
    source_sha256: str | None = None,
    truth: TruthIdentity | None = None,
    biological_lot_id: str | None = None,
    specimen_group_id: str | None = None,
    physical_specimen_id: str | None = None,
    capture_session_id: str | None = None,
    capture_group_id: str | None = None,
) -> BenchmarkSample:
    return BenchmarkSample(
        sample_id=sample_id,
        source_sha256=source_sha256 or sha256((sample_id + ":source").encode()).hexdigest(),
        coordinate_frame=CoordinateFrame(
            frame_id=f"frame_{sample_id}",
            shape=(24, 32),
            origin="top_left",
            axes="x_right_y_down",
            units="pixel",
        ),
        species="soybean",
        split_role=role,
        truth=truth or _truth(sample_id),
        biological_lot_id=biological_lot_id,
        specimen_group_id=specimen_group_id,
        physical_specimen_id=physical_specimen_id,
        capture_session_id=capture_session_id,
        capture_group_id=capture_group_id,
    )


def _ready_manifest(
    *, samples: tuple[BenchmarkSample, ...] | None = None,
    access_records: tuple[AccessRecord, ...] = (),
    access_log_complete: bool = True,
    access_log_unavailable_reason: str | None = None,
) -> BenchmarkManifest:
    if samples is None:
        samples = (
            _sample(
                "dev_1",
                biological_lot_id="lot_dev",
                specimen_group_id="soy_dev",
                capture_session_id="session_dev",
            ),
            _sample(
                "test_1",
                SplitRole.LOCKED_TEST,
                biological_lot_id="lot_test",
                specimen_group_id="soy_test",
                capture_session_id="session_test",
            ),
        )
    return BenchmarkManifest(
        study_id="synthetic_manifest_test",
        samples=samples,
        access_log_complete=access_log_complete,
        access_records=access_records,
        access_log_unavailable_reason=access_log_unavailable_reason,
    )


@functional_units
class BenchmarkManifestTests(unittest.TestCase):
    def check_test_round_trip_is_immutable_strict_and_does_not_mutate_payload(self) -> None:
        manifest = _ready_manifest()
        payload = manifest.to_dict()
        original = deepcopy(payload)

        decoded_once = BenchmarkManifest.from_dict(payload)
        decoded_twice = BenchmarkManifest.from_dict(payload)
        decoded_json = BenchmarkManifest.from_json(manifest.to_json())

        self.assertEqual(payload, original)
        self.assertEqual(decoded_once, manifest)
        self.assertEqual(decoded_twice, manifest)
        self.assertEqual(decoded_json, manifest)
        self.assertEqual(
            decoded_once.samples[0].identity_sha256,
            manifest.samples[0].identity_sha256,
        )
        self.assertEqual(decoded_once.samples[0].coordinate_frame.shape, (24, 32))
        self.assertEqual(decoded_once.samples[0].truth.reviewer_ids, ("reviewer_a", "reviewer_b"))
        with self.assertRaises((AttributeError, TypeError)):
            decoded_once.samples[0].sample_id = "changed"

    def check_test_atomic_file_round_trip_and_audit_are_explicitly_not_science_certification(self) -> None:
        manifest = _ready_manifest()
        with tempfile.TemporaryDirectory() as temporary:
            destination = manifest.save(Path(temporary) / "study.json")
            restored = BenchmarkManifest.load(destination)
        audit = audit_benchmark_manifest(restored)

        self.assertEqual(restored, manifest)
        self.assertTrue(audit.evaluation_eligible)
        self.assertEqual(audit.audit_version, 1)
        self.assertEqual(audit.manifest_sha256, manifest.manifest_sha256)
        self.assertEqual(len(manifest.manifest_sha256), 64)
        self.assertEqual(audit.sample_count, 2)
        self.assertEqual(audit.access_record_count, 0)
        self.assertEqual(dict((role.value, count) for role, count in audit.split_counts)["locked_test"], 1)
        self.assertEqual(audit.scientific_validation_status, "not_certified")
        self.assertEqual(audit.to_dict()["scientific_validation_status"], "not_certified")

    def check_test_same_group_may_contain_multiple_images_within_one_split(self) -> None:
        samples = (
            _sample(
                "dev_a",
                biological_lot_id="lot_shared",
                specimen_group_id="specimens_shared",
                capture_session_id="session_shared",
            ),
            _sample(
                "dev_b",
                biological_lot_id="lot_shared",
                specimen_group_id="specimens_shared",
                capture_session_id="session_shared",
            ),
            _sample(
                "test_a",
                SplitRole.LOCKED_TEST,
                biological_lot_id="lot_held_out",
                specimen_group_id="specimens_held_out",
                capture_session_id="session_held_out",
            ),
        )
        audit = audit_benchmark_manifest(_ready_manifest(samples=samples))

        self.assertTrue(audit.evaluation_eligible)
        self.assertEqual(audit.related_group_leakage_count, 0)
        self.assertEqual(audit.findings, ())

    def check_test_source_duplicates_and_transitive_related_groups_crossing_roles_are_reported(self) -> None:
        repeated_source = "a" * 64
        samples = (
            _sample(
                "dev_a",
                source_sha256=repeated_source,
                biological_lot_id="lot_a",
                specimen_group_id="lineage_bridge",
                capture_session_id="session_a",
            ),
            _sample(
                "dev_b",
                biological_lot_id="lot_b",
                specimen_group_id="lineage_bridge",
                capture_session_id="session_b",
                capture_group_id="capture_bridge",
            ),
            _sample(
                "test_c",
                SplitRole.LOCKED_TEST,
                source_sha256=repeated_source,
                biological_lot_id="lot_c",
                specimen_group_id="lineage_c",
                capture_session_id="session_c",
                capture_group_id="capture_bridge",
            ),
        )
        audit = audit_benchmark_manifest(_ready_manifest(samples=samples))
        findings = {item.code: item for item in audit.findings}

        self.assertFalse(audit.evaluation_eligible)
        self.assertEqual(audit.source_duplicate_group_count, 1)
        self.assertEqual(audit.related_group_leakage_count, 1)
        self.assertEqual(
            findings["source_content_crosses_splits"].sample_ids,
            ("dev_a", "test_c"),
        )
        self.assertEqual(
            findings["related_groups_cross_splits"].sample_ids,
            ("dev_a", "dev_b", "test_c"),
        )
        self.assertEqual(
            findings["related_groups_cross_splits"].group_kinds,
            ("capture_group", "specimen_group"),
        )

    def check_test_each_development_or_reference_access_to_locked_test_blocks_eligibility(self) -> None:
        for index, purpose in enumerate(AccessPurpose):
            with self.subTest(purpose=purpose.value):
                event = AccessRecord(
                    access_id=f"access_{index}",
                    purpose=purpose,
                    sample_ids=("test_1",),
                    actor_id="worker_1",
                )
                audit = audit_benchmark_manifest(_ready_manifest(access_records=(event,)))
                self.assertFalse(audit.evaluation_eligible)
                self.assertEqual(audit.locked_test_access_count, 1)
                self.assertIn("locked_test_access_recorded", audit.eligibility_blockers)
                finding = next(item for item in audit.findings if item.code == "locked_test_access_recorded")
                self.assertEqual(finding.sample_ids, ("test_1",))
                self.assertEqual(finding.access_ids, (f"access_{index}",))
                self.assertEqual(finding.access_purpose, purpose)
                self.assertEqual(finding.locked_test_target_ids, ("test_1",))

    def check_test_access_through_source_alias_and_every_group_relation_is_resolved(self) -> None:
        group_fields = (
            "biological_lot_id",
            "specimen_group_id",
            "physical_specimen_id",
            "capture_session_id",
            "capture_group_id",
        )
        for purpose in AccessPurpose:
            with self.subTest(purpose=purpose.value, relation="source_sha256"):
                shared_source = "e" * 64
                manifest = _ready_manifest(
                    samples=(
                        _sample(
                            "dev_alias",
                            source_sha256=shared_source,
                            biological_lot_id="lot_dev",
                            specimen_group_id="spec_dev",
                            capture_session_id="session_dev",
                        ),
                        _sample(
                            "test_target",
                            SplitRole.LOCKED_TEST,
                            source_sha256=shared_source,
                            biological_lot_id="lot_test",
                            specimen_group_id="spec_test",
                            capture_session_id="session_test",
                        ),
                    ),
                    access_records=(AccessRecord(
                        access_id="source_alias_access",
                        purpose=purpose,
                        sample_ids=("dev_alias",),
                        actor_id="worker_1",
                    ),),
                )
                finding = next(
                    item for item in audit_benchmark_manifest(manifest).findings
                    if item.code == "locked_test_access_recorded"
                )
                self.assertEqual(finding.sample_ids, ("dev_alias",))
                self.assertEqual(finding.locked_test_target_ids, ("test_target",))
                self.assertEqual(finding.group_kinds, ("source_sha256",))
                self.assertEqual(finding.access_purpose, purpose)

            for group_field in group_fields:
                with self.subTest(purpose=purpose.value, relation=group_field):
                    dev_groups = {
                        "biological_lot_id": "lot_dev",
                        "specimen_group_id": "spec_dev",
                        "physical_specimen_id": "physical_dev",
                        "capture_session_id": "session_dev",
                        "capture_group_id": "capture_dev",
                    }
                    test_groups = {
                        "biological_lot_id": "lot_test",
                        "specimen_group_id": "spec_test",
                        "physical_specimen_id": "physical_test",
                        "capture_session_id": "session_test",
                        "capture_group_id": "capture_test",
                    }
                    dev_groups[group_field] = "shared_relation"
                    test_groups[group_field] = "shared_relation"
                    manifest = _ready_manifest(
                        samples=(
                            _sample("dev_related", **dev_groups),
                            _sample("test_related", SplitRole.LOCKED_TEST, **test_groups),
                        ),
                        access_records=(AccessRecord(
                            access_id="group_access",
                            purpose=purpose,
                            sample_ids=("dev_related",),
                            actor_id="worker_1",
                        ),),
                    )
                    audit = audit_benchmark_manifest(manifest)
                    finding = next(
                        item for item in audit.findings
                        if item.code == "locked_test_access_recorded"
                    )
                    self.assertEqual(audit.locked_test_access_count, 1)
                    self.assertEqual(finding.sample_ids, ("dev_related",))
                    self.assertEqual(finding.locked_test_target_ids, ("test_related",))
                    self.assertEqual(finding.group_kinds, (group_field.removesuffix("_id"),))
                    self.assertEqual(finding.access_purpose, purpose)

    def check_test_access_via_transitive_related_group_reports_accessed_and_locked_ids(self) -> None:
        samples = (
            _sample(
                "accessed_dev",
                biological_lot_id="lot_accessed",
                specimen_group_id="shared_specimen_group",
                capture_session_id="session_accessed",
            ),
            _sample(
                "related_dev",
                biological_lot_id="lot_related",
                specimen_group_id="shared_specimen_group",
                capture_session_id="session_related",
                capture_group_id="shared_capture_group",
            ),
            _sample(
                "locked_test",
                SplitRole.LOCKED_TEST,
                biological_lot_id="lot_locked",
                specimen_group_id="specimen_locked",
                capture_session_id="session_locked",
                capture_group_id="shared_capture_group",
            ),
        )
        event = AccessRecord(
            access_id="transitive_access",
            purpose=AccessPurpose.LIBRARY_EXTRACTION,
            sample_ids=("accessed_dev",),
            actor_id="worker_1",
        )
        audit = audit_benchmark_manifest(_ready_manifest(samples=samples, access_records=(event,)))
        finding = next(
            item for item in audit.findings
            if item.code == "locked_test_access_recorded"
        )

        self.assertEqual(audit.locked_test_access_count, 1)
        self.assertEqual(finding.sample_ids, ("accessed_dev",))
        self.assertEqual(finding.locked_test_target_ids, ("locked_test",))
        self.assertEqual(finding.group_kinds, ("capture_group", "specimen_group"))
        self.assertEqual(finding.access_ids, ("transitive_access",))
        self.assertEqual(finding.access_purpose, AccessPurpose.LIBRARY_EXTRACTION)

    def check_test_access_closure_crosses_alternating_source_and_group_edges(self) -> None:
        source_ab = "1" * 64
        source_cd = "2" * 64
        source_ef = "3" * 64
        # The first route starts with a source alias and alternates through group
        # and source identities. The second reverses the first two edge types.
        routes = (
            (
                _sample("a", source_sha256=source_ab, biological_lot_id="lot_a", capture_session_id="session_a"),
                _sample("b", source_sha256=source_ab, biological_lot_id="lot_b", capture_group_id="bridge_1", capture_session_id="session_b"),
                _sample("c", source_sha256=source_cd, biological_lot_id="lot_c", capture_group_id="bridge_1", capture_session_id="session_c"),
                _sample("d", source_sha256=source_cd, biological_lot_id="lot_d", specimen_group_id="bridge_2", capture_session_id="session_d"),
                _sample("e", SplitRole.LOCKED_TEST, source_sha256=source_ef, biological_lot_id="lot_e", specimen_group_id="bridge_2", capture_session_id="session_e"),
            ),
            (
                _sample("a", source_sha256="4" * 64, biological_lot_id="lot_a", specimen_group_id="bridge_1", capture_session_id="session_a"),
                _sample("b", source_sha256="5" * 64, biological_lot_id="lot_b", specimen_group_id="bridge_1", capture_session_id="session_b"),
                _sample("c", SplitRole.LOCKED_TEST, source_sha256="5" * 64, biological_lot_id="lot_c", capture_session_id="session_c"),
            ),
        )
        for route_index, samples in enumerate(routes):
            for purpose_index, purpose in enumerate(AccessPurpose):
                with self.subTest(route=route_index, purpose=purpose.value):
                    event = AccessRecord(
                        access_id=f"mixed_{route_index}_{purpose_index}",
                        purpose=purpose,
                        sample_ids=("a", "b"),
                        actor_id="worker_1",
                    )
                    audit = audit_benchmark_manifest(_ready_manifest(samples=samples, access_records=(event,)))
                    finding = next(
                        item for item in audit.findings
                        if item.code == "locked_test_access_recorded"
                    )

                    self.assertFalse(audit.evaluation_eligible)
                    self.assertEqual(audit.locked_test_access_count, 1)
                    self.assertEqual(finding.sample_ids, ("a", "b"))
                    self.assertEqual(finding.locked_test_target_ids, ("e" if route_index == 0 else "c",))
                    self.assertEqual(finding.access_purpose, purpose)
                    self.assertIn("source_sha256", finding.group_kinds)
                    self.assertTrue({"capture_group", "specimen_group"}.intersection(finding.group_kinds))
                    self.assertEqual(audit.source_duplicate_group_count, 0 if route_index == 0 else 1)
                    self.assertEqual(audit.related_group_leakage_count, 1 if route_index == 0 else 0)

    def check_test_audit_report_bounds_findings_and_names_the_omitted_count(self) -> None:
        records = tuple(
            AccessRecord(
                access_id=f"access_{index}",
                purpose=AccessPurpose.TRAINING,
                sample_ids=("test_1",),
                actor_id="worker_1",
            )
            for index in range(6)
        )
        with patch.object(manifest_module, "MAX_AUDIT_FINDINGS", 2):
            audit = audit_benchmark_manifest(_ready_manifest(access_records=records))

        self.assertEqual(len(audit.findings), 2)
        self.assertEqual(audit.omitted_finding_count, 4)
        self.assertEqual(audit.locked_test_access_count, 6)
        self.assertEqual(audit.to_dict()["omitted_finding_count"], 4)

    def check_test_missing_groups_truth_or_access_log_blocks_evaluation_without_guessing(self) -> None:
        unavailable = TruthIdentity(
            state=TruthState.UNAVAILABLE,
            unavailable_reason="No reviewed truth package is available.",
        )
        samples = (
            _sample("dev_without_groups"),
            _sample("test_without_groups", SplitRole.LOCKED_TEST, truth=unavailable),
        )
        audit = audit_benchmark_manifest(_ready_manifest(
            samples=samples,
            access_log_complete=False,
            access_log_unavailable_reason="Historical access records were not retained.",
        ))

        self.assertFalse(audit.evaluation_eligible)
        self.assertTrue({
            "biological_group_missing",
            "capture_group_missing",
            "locked_test_truth_unavailable",
            "access_log_incomplete",
        }.issubset(audit.eligibility_blockers))
        self.assertEqual(audit.truth_unavailable_count, 1)

    def check_test_strict_schema_rejects_unknown_duplicate_nonfinite_and_bad_role_values(self) -> None:
        manifest = _ready_manifest()
        payload = manifest.to_dict()

        with self.assertRaisesRegex(BenchmarkManifestError, "missing or unknown"):
            BenchmarkManifest.from_dict(payload | {"extra": True})
        with self.assertRaisesRegex(BenchmarkManifestError, "trimmed text"):
            BenchmarkManifest.from_dict(payload | {"study_id": " not trimmed "})
        with self.assertRaisesRegex(BenchmarkManifestError, "one of"):
            BenchmarkManifest.from_dict(
                payload | {"samples": [payload["samples"][0] | {"split_role": "test"}, payload["samples"][1]]}
            )
        with self.assertRaisesRegex(BenchmarkManifestError, "Duplicate JSON field"):
            BenchmarkManifest.from_json('{"format":"a","format":"b"}')
        with self.assertRaisesRegex(BenchmarkManifestError, "Non-finite"):
            BenchmarkManifest.from_json('{"version":NaN}')
        with self.assertRaisesRegex(BenchmarkManifestError, "version"):
            BenchmarkManifest.from_dict(payload | {"version": BENCHMARK_MANIFEST_VERSION + 1})
        self.assertEqual(payload["format"], BENCHMARK_MANIFEST_FORMAT)

    def check_test_unavailable_truth_cannot_carry_partial_identity_and_available_truth_is_complete(self) -> None:
        with self.assertRaisesRegex(BenchmarkManifestError, "cannot claim reviewed"):
            TruthIdentity(
                state=TruthState.UNAVAILABLE,
                annotation_sha256="a" * 64,
                unavailable_reason="not actually unavailable",
            )
        with self.assertRaisesRegex(BenchmarkManifestError, "requires annotation"):
            TruthIdentity(state=TruthState.AVAILABLE, annotation_sha256="a" * 64)
        unavailable = TruthIdentity(
            state=TruthState.UNAVAILABLE,
            unavailable_reason="No reviewed truth package exists.",
        )
        for invalid_revision in ("", 0, False, [], {}):
            with self.subTest(revision=invalid_revision):
                payload = unavailable.to_dict()
                payload["revision"] = invalid_revision
                original = deepcopy(payload)
                with self.assertRaisesRegex(BenchmarkManifestError, "cannot claim reviewed"):
                    TruthIdentity.from_dict(payload)
                self.assertEqual(payload, original)

    def check_test_count_and_byte_limits_apply_to_direct_construction_and_decode(self) -> None:
        one = _sample(
            "dev_one", biological_lot_id="lot_a", specimen_group_id="group_a",
            capture_session_id="session_a",
        )
        two = _sample(
            "test_two", SplitRole.LOCKED_TEST, biological_lot_id="lot_b",
            specimen_group_id="group_b", capture_session_id="session_b",
        )
        with patch.object(manifest_module, "MAX_BENCHMARK_SAMPLES", 1):
            with self.assertRaisesRegex(BenchmarkManifestError, "samples"):
                _ready_manifest(samples=(one, two))
            payload = _ready_manifest(samples=(one,)).to_dict()
            with self.assertRaisesRegex(BenchmarkManifestError, "samples"):
                BenchmarkManifest.from_dict(payload | {"samples": [payload["samples"][0], two.to_dict()]})
        with patch.object(manifest_module, "MAX_BENCHMARK_MANIFEST_BYTES", 128):
            with self.assertRaisesRegex(BenchmarkManifestError, "byte limit"):
                _ready_manifest(samples=(one,))
            with self.assertRaisesRegex(BenchmarkManifestError, "byte limit"):
                BenchmarkManifest.from_json(json.dumps(_ready_manifest(samples=(one,)).to_dict()))

    def check_test_exact_production_byte_limit_constructs_loads_and_saves_identically(self) -> None:
        count = 4575
        samples = tuple(
            _sample(
                f"s{index:05}",
                biological_lot_id="lot",
                specimen_group_id="group",
                capture_session_id="session",
            )
            for index in range(count)
        )
        payload = {
            "format": BENCHMARK_MANIFEST_FORMAT,
            "version": BENCHMARK_MANIFEST_VERSION,
            "study_id": "exact_limit",
            "access_log_complete": False,
            "access_log_unavailable_reason": "",
            "samples": [sample.to_dict() for sample in samples],
            "access_records": [],
        }
        base_size = len(json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8"))
        reason_size = manifest_module.MAX_BENCHMARK_MANIFEST_BYTES - base_size
        self.assertGreater(reason_size, 0)
        self.assertLessEqual(reason_size, manifest_module.MAX_MANIFEST_TEXT_CHARS)
        payload["access_log_unavailable_reason"] = "x" * reason_size

        manifest = BenchmarkManifest.from_dict(payload)
        encoded = manifest.to_json().encode("utf-8")
        self.assertEqual(len(encoded), manifest_module.MAX_BENCHMARK_MANIFEST_BYTES)
        with tempfile.TemporaryDirectory() as temporary:
            destination = manifest.save(Path(temporary) / "at-limit.json")
            self.assertEqual(destination.stat().st_size, len(encoded))
            self.assertEqual(destination.read_bytes(), encoded)
            self.assertEqual(BenchmarkManifest.load(destination), manifest)

    def check_test_exact_lowered_byte_limit_is_shared_by_constructor_load_and_save(self) -> None:
        manifest = _ready_manifest()
        encoded = manifest.to_json().encode("utf-8")
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "at-lowered-limit.json"
            with patch.object(manifest_module, "MAX_BENCHMARK_MANIFEST_BYTES", len(encoded)):
                decoded = BenchmarkManifest.from_json(encoded)
                decoded.save(destination)
                self.assertEqual(destination.stat().st_size, len(encoded))
                self.assertEqual(BenchmarkManifest.load(destination), manifest)
            with patch.object(manifest_module, "MAX_BENCHMARK_MANIFEST_BYTES", len(encoded) - 1):
                with self.assertRaisesRegex(BenchmarkManifestError, "byte limit"):
                    BenchmarkManifest.from_json(encoded)
                with self.assertRaisesRegex(BenchmarkManifestError, "byte limit"):
                    manifest.save(destination)

    def test_workflow_lifecycle(self):
        """9 related BenchmarkManifestTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_round_trip_is_immutable_strict_and_does_not_mutate_payload)
        self.scenario(self.check_test_atomic_file_round_trip_and_audit_are_explicitly_not_science_certification)
        self.scenario(self.check_test_same_group_may_contain_multiple_images_within_one_split)
        self.scenario(self.check_test_source_duplicates_and_transitive_related_groups_crossing_roles_are_reported)
        self.scenario(self.check_test_each_development_or_reference_access_to_locked_test_blocks_eligibility)
        self.scenario(self.check_test_access_through_source_alias_and_every_group_relation_is_resolved)
        self.scenario(self.check_test_access_via_transitive_related_group_reports_accessed_and_locked_ids)
        self.scenario(self.check_test_access_closure_crosses_alternating_source_and_group_edges)
        self.scenario(self.check_test_audit_report_bounds_findings_and_names_the_omitted_count)

    def test_validation_and_recovery(self):
        """6 related BenchmarkManifestTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_missing_groups_truth_or_access_log_blocks_evaluation_without_guessing)
        self.scenario(self.check_test_strict_schema_rejects_unknown_duplicate_nonfinite_and_bad_role_values)
        self.scenario(self.check_test_unavailable_truth_cannot_carry_partial_identity_and_available_truth_is_complete)
        self.scenario(self.check_test_count_and_byte_limits_apply_to_direct_construction_and_decode)
        self.scenario(self.check_test_exact_production_byte_limit_constructs_loads_and_saves_identically)
        self.scenario(self.check_test_exact_lowered_byte_limit_is_shared_by_constructor_load_and_save)


if __name__ == "__main__":
    unittest.main()
