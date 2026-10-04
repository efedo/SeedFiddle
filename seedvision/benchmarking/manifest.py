"""Strict study manifests and metadata-only benchmark leakage audits.

The audit checks declared identities, split roles, grouping and access records.
It cannot prove that the declarations are truthful or certify scientific
performance. Raster files and annotations remain external artifacts identified
by immutable digests; no image data is embedded here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
import heapq
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from seedvision.benchmarking.checkpoints import CoordinateFrame


BENCHMARK_MANIFEST_FORMAT = "seedfiddle.benchmark-manifest"
BENCHMARK_MANIFEST_VERSION = 1
LEAKAGE_AUDIT_VERSION = 1
MAX_BENCHMARK_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_BENCHMARK_SAMPLES = 100_000
MAX_BENCHMARK_ACCESS_RECORDS = 100_000
MAX_ACCESS_SAMPLE_IDS = 100_000
MAX_TRUTH_REVIEWERS = 32
MAX_AUDIT_FINDINGS = 256
MAX_AUDIT_SAMPLE_IDS = 1024
MAX_MANIFEST_TEXT_CHARS = 4096

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_STABLE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")


class BenchmarkManifestError(ValueError):
    """A benchmark manifest is malformed, inconsistent, or over its limits."""


class SplitRole(StrEnum):
    DEVELOPMENT = "development"
    VALIDATION = "validation"
    LOCKED_TEST = "locked_test"


class AccessPurpose(StrEnum):
    TRAINING = "training"
    LIBRARY_EXTRACTION = "library_extraction"
    PARAMETER_DEVELOPMENT = "parameter_development"
    REFERENCE = "reference"


class TruthState(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


def _text(value: Any, name: str, *, stable_id: bool = False) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise BenchmarkManifestError(f"{name} must be non-empty trimmed text.")
    if len(value) > MAX_MANIFEST_TEXT_CHARS:
        raise BenchmarkManifestError(
            f"{name} exceeds the {MAX_MANIFEST_TEXT_CHARS}-character limit."
        )
    if "\x00" in value:
        raise BenchmarkManifestError(f"{name} cannot contain a NUL character.")
    if stable_id and not _STABLE_ID.fullmatch(value):
        raise BenchmarkManifestError(f"{name} must be a stable identifier.")
    return value


def _optional_text(value: Any, name: str) -> str | None:
    return None if value is None else _text(value, name, stable_id=True)


def _digest(value: Any, name: str, *, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise BenchmarkManifestError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _enum(enum_type: type[StrEnum], value: Any, name: str) -> StrEnum:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as error:
        choices = ", ".join(member.value for member in enum_type)
        raise BenchmarkManifestError(f"{name} must be one of: {choices}.") from error


def _exact_object(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise BenchmarkManifestError(f"{name} has missing or unknown fields.")
    # Public decoders may transform child values; never mutate caller-owned JSON.
    return dict(value)


def _canonical_json(value: Any, *, name: str, maximum: int | None = None) -> bytes:
    try:
        encoder = json.JSONEncoder(
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        encoded = bytearray()
        for part in encoder.iterencode(value):
            # Reject obviously overlong chunks before allocating their UTF-8
            # copy; a non-ASCII chunk may encode to at most four bytes/char.
            if maximum is not None and len(part) > maximum:
                raise BenchmarkManifestError(f"{name} exceeds the {maximum}-byte limit.")
            chunk = part.encode("utf-8")
            if maximum is not None and len(encoded) + len(chunk) > maximum:
                raise BenchmarkManifestError(f"{name} exceeds the {maximum}-byte limit.")
            encoded.extend(chunk)
    except BenchmarkManifestError:
        raise
    except (TypeError, ValueError, UnicodeError, RecursionError) as error:
        raise BenchmarkManifestError(f"{name} must contain strict finite JSON values.") from error
    return bytes(encoded)


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BenchmarkManifestError(f"Duplicate JSON field {key!r} is forbidden.")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise BenchmarkManifestError(f"Non-finite JSON number {value!r} is forbidden.")


@dataclass(frozen=True, slots=True)
class TruthIdentity:
    """Identity of reviewed annotations, regions and eligibility, not their pixels."""

    state: TruthState
    annotation_sha256: str | None = None
    reviewed_regions_sha256: str | None = None
    eligibility_sha256: str | None = None
    revision: str | None = None
    reviewer_ids: tuple[str, ...] = ()
    adjudication_sha256: str | None = None
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", _enum(TruthState, self.state, "Truth state"))
        object.__setattr__(self, "annotation_sha256", _digest(
            self.annotation_sha256, "annotation SHA-256", optional=True
        ))
        object.__setattr__(self, "reviewed_regions_sha256", _digest(
            self.reviewed_regions_sha256, "reviewed-region SHA-256", optional=True
        ))
        object.__setattr__(self, "eligibility_sha256", _digest(
            self.eligibility_sha256, "eligibility SHA-256", optional=True
        ))
        object.__setattr__(self, "adjudication_sha256", _digest(
            self.adjudication_sha256, "adjudication SHA-256", optional=True
        ))
        if not isinstance(self.reviewer_ids, tuple) or len(self.reviewer_ids) > MAX_TRUTH_REVIEWERS:
            raise BenchmarkManifestError(
                f"reviewer_ids must be an immutable tuple of at most {MAX_TRUTH_REVIEWERS} IDs."
            )
        reviewers = tuple(_text(item, "reviewer ID", stable_id=True) for item in self.reviewer_ids)
        if len(set(reviewers)) != len(reviewers):
            raise BenchmarkManifestError("reviewer_ids must be unique.")
        object.__setattr__(self, "reviewer_ids", reviewers)
        if self.state is TruthState.AVAILABLE:
            if any(value is None for value in (
                self.annotation_sha256,
                self.reviewed_regions_sha256,
                self.eligibility_sha256,
                self.revision,
            )) or not self.reviewer_ids:
                raise BenchmarkManifestError(
                    "Available truth requires annotation, reviewed-region and eligibility hashes, "
                    "a revision, and reviewer IDs."
                )
            object.__setattr__(self, "revision", _text(self.revision, "truth revision", stable_id=True))
            if self.unavailable_reason is not None:
                raise BenchmarkManifestError("Available truth cannot have an unavailable reason.")
        else:
            if any(value is not None for value in (
                self.annotation_sha256,
                self.reviewed_regions_sha256,
                self.eligibility_sha256,
                self.revision,
                self.adjudication_sha256,
            )) or self.reviewer_ids != ():
                raise BenchmarkManifestError("Unavailable truth cannot claim reviewed truth identities.")
            object.__setattr__(
                self,
                "unavailable_reason",
                _text(self.unavailable_reason, "truth unavailable reason"),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "annotation_sha256": self.annotation_sha256,
            "reviewed_regions_sha256": self.reviewed_regions_sha256,
            "eligibility_sha256": self.eligibility_sha256,
            "revision": self.revision,
            "reviewer_ids": list(self.reviewer_ids),
            "adjudication_sha256": self.adjudication_sha256,
            "unavailable_reason": self.unavailable_reason,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "TruthIdentity":
        data = _exact_object(
            value,
            {
                "state", "annotation_sha256", "reviewed_regions_sha256",
                "eligibility_sha256", "revision", "reviewer_ids",
                "adjudication_sha256", "unavailable_reason",
            },
            "truth identity",
        )
        if not isinstance(data["reviewer_ids"], list):
            raise BenchmarkManifestError("Truth reviewer_ids must be a JSON array.")
        data["reviewer_ids"] = tuple(data["reviewer_ids"])
        return cls(**data)


@dataclass(frozen=True, slots=True)
class BenchmarkSample:
    """One source/frame and its declared dataset role and grouping identities."""

    sample_id: str
    source_sha256: str
    coordinate_frame: CoordinateFrame
    species: str
    split_role: SplitRole
    truth: TruthIdentity
    biological_lot_id: str | None = None
    specimen_group_id: str | None = None
    physical_specimen_id: str | None = None
    capture_session_id: str | None = None
    capture_group_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_id", _text(self.sample_id, "sample ID", stable_id=True))
        object.__setattr__(self, "source_sha256", _digest(self.source_sha256, "source SHA-256"))
        if not isinstance(self.coordinate_frame, CoordinateFrame):
            raise BenchmarkManifestError("coordinate_frame must be an exact CoordinateFrame.")
        for field_name in ("frame_id", "origin", "axes", "units"):
            _text(getattr(self.coordinate_frame, field_name), f"coordinate frame {field_name}")
        object.__setattr__(self, "species", _text(self.species, "species"))
        object.__setattr__(self, "split_role", _enum(SplitRole, self.split_role, "split role"))
        if not isinstance(self.truth, TruthIdentity):
            raise BenchmarkManifestError("truth must be a TruthIdentity.")
        for field_name in (
            "biological_lot_id", "specimen_group_id", "physical_specimen_id",
            "capture_session_id", "capture_group_id",
        ):
            object.__setattr__(self, field_name, _optional_text(getattr(self, field_name), field_name))

    @property
    def identity_sha256(self) -> str:
        """Digest binding the complete immutable sample record."""

        return sha256(_canonical_json(self.to_dict(), name="sample identity")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "source_sha256": self.source_sha256,
            "coordinate_frame": self.coordinate_frame.to_dict(),
            "species": self.species,
            "split_role": self.split_role.value,
            "biological_lot_id": self.biological_lot_id,
            "specimen_group_id": self.specimen_group_id,
            "physical_specimen_id": self.physical_specimen_id,
            "capture_session_id": self.capture_session_id,
            "capture_group_id": self.capture_group_id,
            "truth": self.truth.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "BenchmarkSample":
        data = _exact_object(
            value,
            {
                "sample_id", "source_sha256", "coordinate_frame", "species", "split_role",
                "biological_lot_id", "specimen_group_id", "physical_specimen_id",
                "capture_session_id", "capture_group_id", "truth",
            },
            "benchmark sample",
        )
        try:
            data["coordinate_frame"] = CoordinateFrame.from_dict(data["coordinate_frame"])
            data["truth"] = TruthIdentity.from_dict(data["truth"])
            return cls(**data)
        except (TypeError, ValueError, KeyError, OverflowError) as error:
            if isinstance(error, BenchmarkManifestError):
                raise
            raise BenchmarkManifestError(f"Benchmark sample is invalid: {error}") from error


@dataclass(frozen=True, slots=True)
class AccessRecord:
    """Declared use of one or more manifest samples by a development activity."""

    access_id: str
    purpose: AccessPurpose
    sample_ids: tuple[str, ...]
    actor_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "access_id", _text(self.access_id, "access ID", stable_id=True))
        object.__setattr__(self, "purpose", _enum(AccessPurpose, self.purpose, "access purpose"))
        if not isinstance(self.sample_ids, tuple) or not self.sample_ids:
            raise BenchmarkManifestError("Access sample_ids must be a non-empty immutable tuple.")
        if len(self.sample_ids) > MAX_ACCESS_SAMPLE_IDS:
            raise BenchmarkManifestError(f"An access record may name at most {MAX_ACCESS_SAMPLE_IDS} samples.")
        sample_ids = tuple(_text(item, "access sample ID", stable_id=True) for item in self.sample_ids)
        if len(set(sample_ids)) != len(sample_ids):
            raise BenchmarkManifestError("Access sample_ids must be unique.")
        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(self, "actor_id", _text(self.actor_id, "access actor ID", stable_id=True))

    def to_dict(self) -> dict[str, Any]:
        return {
            "access_id": self.access_id,
            "purpose": self.purpose.value,
            "sample_ids": list(self.sample_ids),
            "actor_id": self.actor_id,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "AccessRecord":
        data = _exact_object(value, {"access_id", "purpose", "sample_ids", "actor_id"}, "access record")
        if not isinstance(data["sample_ids"], list):
            raise BenchmarkManifestError("Access sample_ids must be a JSON array.")
        data["sample_ids"] = tuple(data["sample_ids"])
        return cls(**data)


@dataclass(frozen=True, slots=True)
class BenchmarkManifest:
    """Versioned, immutable source/truth inventory for one comparison study."""

    study_id: str
    samples: tuple[BenchmarkSample, ...]
    access_log_complete: bool
    access_records: tuple[AccessRecord, ...] = ()
    access_log_unavailable_reason: str | None = None
    format: str = BENCHMARK_MANIFEST_FORMAT
    version: int = BENCHMARK_MANIFEST_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "study_id", _text(self.study_id, "study ID", stable_id=True))
        if self.format != BENCHMARK_MANIFEST_FORMAT or type(self.version) is not int or self.version != BENCHMARK_MANIFEST_VERSION:
            raise BenchmarkManifestError("Unsupported benchmark manifest format or version.")
        if not isinstance(self.samples, tuple) or len(self.samples) > MAX_BENCHMARK_SAMPLES:
            raise BenchmarkManifestError(f"Manifest samples must be an immutable tuple of at most {MAX_BENCHMARK_SAMPLES}.")
        if not self.samples:
            raise BenchmarkManifestError("A benchmark manifest must contain at least one sample.")
        if any(not isinstance(item, BenchmarkSample) for item in self.samples):
            raise BenchmarkManifestError("Manifest samples must be BenchmarkSample values.")
        sample_ids = tuple(item.sample_id for item in self.samples)
        if len(set(sample_ids)) != len(sample_ids):
            raise BenchmarkManifestError("Manifest sample IDs must be unique.")
        if type(self.access_log_complete) is not bool:
            raise BenchmarkManifestError("access_log_complete must be Boolean.")
        if not isinstance(self.access_records, tuple) or len(self.access_records) > MAX_BENCHMARK_ACCESS_RECORDS:
            raise BenchmarkManifestError(
                f"access_records must be an immutable tuple of at most {MAX_BENCHMARK_ACCESS_RECORDS}."
            )
        if any(not isinstance(item, AccessRecord) for item in self.access_records):
            raise BenchmarkManifestError("access_records must contain AccessRecord values.")
        access_ids = tuple(item.access_id for item in self.access_records)
        if len(set(access_ids)) != len(access_ids):
            raise BenchmarkManifestError("Access record IDs must be unique.")
        known_samples = set(sample_ids)
        for record in self.access_records:
            unknown = set(record.sample_ids) - known_samples
            if unknown:
                raise BenchmarkManifestError(
                    f"Access record {record.access_id!r} names unknown sample IDs: {', '.join(sorted(unknown)[:8])}."
                )
        if self.access_log_complete:
            if self.access_log_unavailable_reason is not None:
                raise BenchmarkManifestError("A complete access log cannot have an unavailable reason.")
        else:
            object.__setattr__(
                self,
                "access_log_unavailable_reason",
                _text(self.access_log_unavailable_reason, "access-log unavailable reason"),
            )
        _canonical_json(self.to_dict(), name="benchmark manifest", maximum=MAX_BENCHMARK_MANIFEST_BYTES)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "version": self.version,
            "study_id": self.study_id,
            "access_log_complete": self.access_log_complete,
            "access_log_unavailable_reason": self.access_log_unavailable_reason,
            "samples": [item.to_dict() for item in self.samples],
            "access_records": [item.to_dict() for item in self.access_records],
        }

    def to_json(self) -> str:
        return _canonical_json(
            self.to_dict(), name="benchmark manifest", maximum=MAX_BENCHMARK_MANIFEST_BYTES
        ).decode("utf-8")

    @property
    def manifest_sha256(self) -> str:
        """Content digest suitable for freezing the declared study inventory."""

        return sha256(self.to_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, value: Any) -> "BenchmarkManifest":
        data = _exact_object(
            value,
            {
                "format", "version", "study_id", "access_log_complete",
                "access_log_unavailable_reason", "samples", "access_records",
            },
            "benchmark manifest",
        )
        if data["format"] != BENCHMARK_MANIFEST_FORMAT or type(data["version"]) is not int or data["version"] != BENCHMARK_MANIFEST_VERSION:
            raise BenchmarkManifestError("Unsupported benchmark manifest format or version.")
        if not isinstance(data["samples"], list) or len(data["samples"]) > MAX_BENCHMARK_SAMPLES:
            raise BenchmarkManifestError("Benchmark samples must be a bounded JSON array.")
        if not isinstance(data["access_records"], list) or len(data["access_records"]) > MAX_BENCHMARK_ACCESS_RECORDS:
            raise BenchmarkManifestError("Benchmark access_records must be a bounded JSON array.")
        _canonical_json(data, name="benchmark manifest", maximum=MAX_BENCHMARK_MANIFEST_BYTES)
        try:
            return cls(
                study_id=data["study_id"],
                samples=tuple(BenchmarkSample.from_dict(item) for item in data["samples"]),
                access_log_complete=data["access_log_complete"],
                access_records=tuple(AccessRecord.from_dict(item) for item in data["access_records"]),
                access_log_unavailable_reason=data["access_log_unavailable_reason"],
                format=data["format"],
                version=data["version"],
            )
        except (TypeError, ValueError, KeyError, OverflowError, RecursionError) as error:
            if isinstance(error, BenchmarkManifestError):
                raise
            raise BenchmarkManifestError(f"Benchmark manifest is invalid: {error}") from error

    @classmethod
    def from_json(cls, value: str | bytes | bytearray) -> "BenchmarkManifest":
        if isinstance(value, str):
            if len(value) > MAX_BENCHMARK_MANIFEST_BYTES:
                raise BenchmarkManifestError(
                    f"Benchmark manifest exceeds the {MAX_BENCHMARK_MANIFEST_BYTES}-byte limit."
                )
            try:
                raw = value.encode("utf-8")
            except UnicodeError as error:
                raise BenchmarkManifestError("Benchmark manifest is not UTF-8 JSON.") from error
        elif isinstance(value, (bytes, bytearray)):
            if len(value) > MAX_BENCHMARK_MANIFEST_BYTES:
                raise BenchmarkManifestError(
                    f"Benchmark manifest exceeds the {MAX_BENCHMARK_MANIFEST_BYTES}-byte limit."
                )
            raw = bytes(value)
        else:
            raise BenchmarkManifestError("Benchmark manifest JSON must be text or bytes.")
        if len(raw) > MAX_BENCHMARK_MANIFEST_BYTES:
            raise BenchmarkManifestError(
                f"Benchmark manifest exceeds the {MAX_BENCHMARK_MANIFEST_BYTES}-byte limit."
            )
        try:
            payload = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_pairs,
                parse_constant=_reject_nonfinite,
            )
        except BenchmarkManifestError:
            raise
        except (UnicodeError, json.JSONDecodeError, RecursionError, TypeError, ValueError) as error:
            raise BenchmarkManifestError(f"Benchmark manifest is not strict UTF-8 JSON: {error}") from error
        return cls.from_dict(payload)

    @classmethod
    def load(cls, path: Path | str) -> "BenchmarkManifest":
        source = Path(path)
        try:
            if source.stat().st_size > MAX_BENCHMARK_MANIFEST_BYTES:
                raise BenchmarkManifestError(
                    f"Benchmark manifest exceeds the {MAX_BENCHMARK_MANIFEST_BYTES}-byte limit."
                )
            with source.open("rb") as stream:
                raw = stream.read(MAX_BENCHMARK_MANIFEST_BYTES + 1)
        except BenchmarkManifestError:
            raise
        except OSError as error:
            raise BenchmarkManifestError(f"Could not read benchmark manifest {source}: {error}") from error
        return cls.from_json(raw)

    def save(self, path: Path | str) -> Path:
        destination = Path(path)
        # Persist the exact bounded canonical representation. Adding a newline
        # here would make a manifest at the accepted byte limit unsavable.
        encoded = self.to_json().encode("utf-8")
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor = -1
        temporary: Path | None = None
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
            )
            temporary = Path(temporary_name)
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = -1
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            temporary = None
            return destination
        except OSError as error:
            raise BenchmarkManifestError(f"Could not save benchmark manifest {destination}: {error}") from error
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass


@dataclass(frozen=True, slots=True)
class LeakageFinding:
    code: str
    sample_ids: tuple[str, ...]
    split_roles: tuple[SplitRole, ...] = ()
    group_kinds: tuple[str, ...] = ()
    access_ids: tuple[str, ...] = ()
    access_purpose: AccessPurpose | None = None
    locked_test_target_ids: tuple[str, ...] = ()
    omitted_sample_ids: int = 0
    omitted_locked_test_target_ids: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "sample_ids": list(self.sample_ids),
            "split_roles": [item.value for item in self.split_roles],
            "group_kinds": list(self.group_kinds),
            "access_ids": list(self.access_ids),
            "access_purpose": None if self.access_purpose is None else self.access_purpose.value,
            "locked_test_target_ids": list(self.locked_test_target_ids),
            "omitted_sample_ids": self.omitted_sample_ids,
            "omitted_locked_test_target_ids": self.omitted_locked_test_target_ids,
        }


@dataclass(frozen=True, slots=True)
class LeakageAudit:
    """Bounded metadata audit; eligibility is a protocol-readiness gate only."""

    study_id: str
    manifest_sha256: str
    audit_version: int
    sample_count: int
    access_record_count: int
    split_counts: tuple[tuple[SplitRole, int], ...]
    truth_available_count: int
    truth_unavailable_count: int
    source_duplicate_group_count: int
    related_group_leakage_count: int
    locked_test_access_count: int
    evaluation_eligible: bool
    eligibility_blockers: tuple[str, ...]
    findings: tuple[LeakageFinding, ...]
    omitted_finding_count: int
    scientific_validation_status: str = "not_certified"

    def to_dict(self) -> dict[str, Any]:
        return {
            "study_id": self.study_id,
            "manifest_sha256": self.manifest_sha256,
            "audit_version": self.audit_version,
            "sample_count": self.sample_count,
            "access_record_count": self.access_record_count,
            "split_counts": {role.value: count for role, count in self.split_counts},
            "truth_available_count": self.truth_available_count,
            "truth_unavailable_count": self.truth_unavailable_count,
            "source_duplicate_group_count": self.source_duplicate_group_count,
            "related_group_leakage_count": self.related_group_leakage_count,
            "locked_test_access_count": self.locked_test_access_count,
            "evaluation_eligible": self.evaluation_eligible,
            "eligibility_blockers": list(self.eligibility_blockers),
            "findings": [item.to_dict() for item in self.findings],
            "omitted_finding_count": self.omitted_finding_count,
            "scientific_validation_status": self.scientific_validation_status,
        }


def _group_values(sample: BenchmarkSample) -> tuple[tuple[str, str], ...]:
    values = (
        ("biological_lot", sample.biological_lot_id),
        ("specimen_group", sample.specimen_group_id),
        ("physical_specimen", sample.physical_specimen_id),
        ("capture_session", sample.capture_session_id),
        ("capture_group", sample.capture_group_id),
    )
    return tuple((kind, value) for kind, value in values if value is not None)


def audit_benchmark_manifest(manifest: BenchmarkManifest) -> LeakageAudit:
    """Audit source/group leakage, locked-test access and evaluation prerequisites.

    A true ``evaluation_eligible`` value means only that declared metadata passed
    these checks. It is not evidence of representative sampling, truthful access
    logs, annotation correctness, or scientific validity.
    """

    if not isinstance(manifest, BenchmarkManifest):
        raise BenchmarkManifestError("audit_benchmark_manifest requires a BenchmarkManifest.")
    sample_by_id = {item.sample_id: item for item in manifest.samples}
    sample_index_by_id = {
        item.sample_id: index for index, item in enumerate(manifest.samples)
    }
    role_order = (SplitRole.DEVELOPMENT, SplitRole.VALIDATION, SplitRole.LOCKED_TEST)
    split_counts = tuple(
        (role, sum(sample.split_role is role for sample in manifest.samples))
        for role in role_order
    )
    test_samples = tuple(item for item in manifest.samples if item.split_role is SplitRole.LOCKED_TEST)
    blockers: set[str] = set()
    findings: list[LeakageFinding] = []
    finding_total = 0

    def add_finding(
        code: str,
        ids: set[str] | tuple[str, ...],
        *,
        roles: set[SplitRole] | tuple[SplitRole, ...] = (),
        kinds: set[str] | tuple[str, ...] = (),
        access_ids: set[str] | tuple[str, ...] = (),
        access_purpose: AccessPurpose | None = None,
        locked_test_targets: set[str] | tuple[str, ...] = (),
        locked_test_target_count: int | None = None,
        locked_test_targets_sorted: bool = False,
    ) -> None:
        nonlocal finding_total
        finding_total += 1
        if len(findings) >= MAX_AUDIT_FINDINGS:
            return
        ordered_ids = tuple(sorted(ids))
        ordered_targets = (
            tuple(locked_test_targets)
            if locked_test_targets_sorted
            else tuple(sorted(locked_test_targets))
        )
        target_count = len(ordered_targets) if locked_test_target_count is None else locked_test_target_count
        findings.append(LeakageFinding(
            code=code,
            sample_ids=ordered_ids[:MAX_AUDIT_SAMPLE_IDS],
            split_roles=tuple(sorted(roles, key=lambda item: item.value)),
            group_kinds=tuple(sorted(kinds)),
            access_ids=tuple(sorted(access_ids)[:MAX_AUDIT_SAMPLE_IDS]),
            access_purpose=access_purpose,
            locked_test_target_ids=ordered_targets[:MAX_AUDIT_SAMPLE_IDS],
            omitted_sample_ids=max(0, len(ordered_ids) - MAX_AUDIT_SAMPLE_IDS),
            omitted_locked_test_target_ids=max(0, target_count - len(ordered_targets)),
        ))

    if not test_samples:
        blockers.add("no_locked_test_samples")
        add_finding("no_locked_test_samples", ())
    if not manifest.access_log_complete:
        blockers.add("access_log_incomplete")
        add_finding("access_log_incomplete", ())

    # Missing group declarations are an explicit block; no identity is guessed
    # from file names, paths, species labels, or image similarity.
    missing_biological = {
        item.sample_id for item in manifest.samples
        if item.biological_lot_id is None
        and item.specimen_group_id is None
        and item.physical_specimen_id is None
    }
    missing_capture = {
        item.sample_id for item in manifest.samples
        if item.capture_session_id is None and item.capture_group_id is None
    }
    if missing_biological:
        blockers.add("biological_group_missing")
        add_finding("biological_group_missing", missing_biological)
    if missing_capture:
        blockers.add("capture_group_missing")
        add_finding("capture_group_missing", missing_capture)

    missing_truth = {item.sample_id for item in test_samples if item.truth.state is not TruthState.AVAILABLE}
    if missing_truth:
        blockers.add("locked_test_truth_unavailable")
        add_finding("locked_test_truth_unavailable", missing_truth, roles={SplitRole.LOCKED_TEST})

    source_groups: dict[str, list[str]] = {}
    for sample in manifest.samples:
        source_groups.setdefault(sample.source_sha256, []).append(sample.sample_id)
    source_duplicate_group_count = 0
    for sample_ids in source_groups.values():
        roles = {sample_by_id[item].split_role for item in sample_ids}
        if len(sample_ids) > 1 and len(roles) > 1:
            source_duplicate_group_count += 1
            blockers.add("source_content_crosses_splits")
            add_finding("source_content_crosses_splits", set(sample_ids), roles=roles)

    # Union samples sharing any explicit biological or acquisition identity.
    # This catches indirect chains, for example specimen A related to B and B
    # captured in the same session as locked-test specimen C.
    parent = list(range(len(manifest.samples)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    group_members: dict[tuple[str, str], list[int]] = {}
    for index, sample in enumerate(manifest.samples):
        for kind, value in _group_values(sample):
            group_members.setdefault((kind, value), []).append(index)
    for indices in group_members.values():
        first = indices[0]
        for index in indices[1:]:
            union(first, index)
    component_group_kinds: dict[int, set[str]] = {}
    for (kind, _value), indices in group_members.items():
        if len(indices) > 1:
            component_group_kinds.setdefault(find(indices[0]), set()).add(kind)
    components: dict[int, list[int]] = {}
    for index in range(len(manifest.samples)):
        components.setdefault(find(index), []).append(index)
    related_group_leakage_count = 0
    component_by_sample: dict[int, int] = {}
    for root, indices in components.items():
        for index in indices:
            component_by_sample[index] = root
    for indices in components.values():
        if len(indices) < 2:
            continue
        roles = {manifest.samples[index].split_role for index in indices}
        if len(roles) < 2:
            continue
        kinds = component_group_kinds.get(find(indices[0]), set())
        related_group_leakage_count += 1
        blockers.add("related_groups_cross_splits")
        add_finding(
            "related_groups_cross_splits",
            {manifest.samples[index].sample_id for index in indices},
            roles=roles,
            kinds=kinds,
        )

    # Access closure follows every explicit identity edge, including source
    # content and biological/capture groups. Keep it separate from the group-only
    # DSU above so source duplicates do not change the distinct group-leakage
    # count or its findings.
    access_parent = list(range(len(manifest.samples)))

    def access_find(index: int) -> int:
        while access_parent[index] != index:
            access_parent[index] = access_parent[access_parent[index]]
            index = access_parent[index]
        return index

    def access_union(left: int, right: int) -> None:
        left_root, right_root = access_find(left), access_find(right)
        if left_root != right_root:
            access_parent[right_root] = left_root

    for indices in group_members.values():
        first = indices[0]
        for index in indices[1:]:
            access_union(first, index)
    for sample_ids in source_groups.values():
        indices = [sample_index_by_id[sample_id] for sample_id in sample_ids]
        first = indices[0]
        for index in indices[1:]:
            access_union(first, index)

    access_component_kinds: dict[int, set[str]] = {}
    for (kind, _value), indices in group_members.items():
        if len(indices) > 1:
            access_component_kinds.setdefault(access_find(indices[0]), set()).add(kind)
    for sample_ids in source_groups.values():
        if len(sample_ids) > 1:
            root = access_find(sample_index_by_id[sample_ids[0]])
            access_component_kinds.setdefault(root, set()).add("source_sha256")

    locked_tests_by_access_component: dict[int, list[str]] = {}
    for index, sample in enumerate(manifest.samples):
        if sample.split_role is SplitRole.LOCKED_TEST:
            locked_tests_by_access_component.setdefault(access_find(index), []).append(sample.sample_id)
    # Components partition samples, so target counts add without a potentially
    # quadratic union of all locked IDs for every access record.
    locked_tests_by_access_component = {
        root: sorted(sample_ids)
        for root, sample_ids in locked_tests_by_access_component.items()
    }

    locked_target_preview_cache: dict[int, tuple[str, ...]] = {}

    def component_target_preview(root: int, limit: int) -> tuple[str, ...]:
        preview = locked_target_preview_cache.get(root)
        if preview is None:
            preview = tuple(locked_tests_by_access_component[root][:limit])
            locked_target_preview_cache[root] = preview
        return preview

    def first_locked_targets(roots: set[int], limit: int) -> tuple[str, ...]:
        """Return the sorted first IDs from disjoint sorted component lists."""
        if len(roots) == 1:
            return component_target_preview(next(iter(roots)), limit)
        heap: list[tuple[str, int, int]] = []
        lists = [component_target_preview(root, limit) for root in roots]
        for list_index, values in enumerate(lists):
            if values:
                heapq.heappush(heap, (values[0], list_index, 0))
        selected: list[str] = []
        while heap and len(selected) < limit:
            sample_id, list_index, position = heapq.heappop(heap)
            selected.append(sample_id)
            next_position = position + 1
            values = lists[list_index]
            if next_position < len(values):
                heapq.heappush(heap, (values[next_position], list_index, next_position))
        return tuple(selected)

    locked_test_access_ids: set[str] = set()
    for record in manifest.access_records:
        accessed_samples: set[str] = set()
        accessed_components: set[int] = set()
        relationships: set[str] = set()
        for sample_id in record.sample_ids:
            sample = sample_by_id[sample_id]
            root = access_find(sample_index_by_id[sample_id])
            if root in locked_tests_by_access_component:
                accessed_samples.add(sample_id)
                accessed_components.add(root)
            if sample.split_role is SplitRole.LOCKED_TEST:
                relationships.add("direct_locked_test_sample")
        for root in accessed_components:
            relationships.update(access_component_kinds.get(root, set()))
        if accessed_components:
            locked_test_access_ids.add(record.access_id)
            blockers.add("locked_test_access_recorded")
            related_roles = {sample_by_id[item].split_role for item in accessed_samples}
            related_roles.add(SplitRole.LOCKED_TEST)
            target_count = sum(len(locked_tests_by_access_component[root]) for root in accessed_components)
            # Findings beyond the report cap contribute to the omitted count,
            # but expanding their bounded target previews would repeat work for
            # every access record without adding auditable output.
            target_preview = (
                first_locked_targets(accessed_components, MAX_AUDIT_SAMPLE_IDS)
                if len(findings) < MAX_AUDIT_FINDINGS
                else ()
            )
            add_finding(
                "locked_test_access_recorded",
                accessed_samples,
                roles=related_roles,
                kinds=relationships,
                access_ids={record.access_id},
                access_purpose=record.purpose,
                locked_test_targets=target_preview,
                locked_test_target_count=target_count,
                locked_test_targets_sorted=True,
            )

    return LeakageAudit(
        study_id=manifest.study_id,
        manifest_sha256=manifest.manifest_sha256,
        audit_version=LEAKAGE_AUDIT_VERSION,
        sample_count=len(manifest.samples),
        access_record_count=len(manifest.access_records),
        split_counts=split_counts,
        truth_available_count=sum(item.truth.state is TruthState.AVAILABLE for item in manifest.samples),
        truth_unavailable_count=sum(item.truth.state is TruthState.UNAVAILABLE for item in manifest.samples),
        source_duplicate_group_count=source_duplicate_group_count,
        related_group_leakage_count=related_group_leakage_count,
        locked_test_access_count=len(locked_test_access_ids),
        evaluation_eligible=not blockers,
        eligibility_blockers=tuple(sorted(blockers)),
        findings=tuple(findings),
        omitted_finding_count=max(0, finding_total - len(findings)),
    )


__all__ = [
    "AccessPurpose",
    "AccessRecord",
    "BENCHMARK_MANIFEST_FORMAT",
    "BENCHMARK_MANIFEST_VERSION",
    "BenchmarkManifest",
    "BenchmarkManifestError",
    "BenchmarkSample",
    "LEAKAGE_AUDIT_VERSION",
    "LeakageAudit",
    "LeakageFinding",
    "MAX_ACCESS_SAMPLE_IDS",
    "MAX_AUDIT_FINDINGS",
    "MAX_AUDIT_SAMPLE_IDS",
    "MAX_BENCHMARK_ACCESS_RECORDS",
    "MAX_BENCHMARK_MANIFEST_BYTES",
    "MAX_BENCHMARK_SAMPLES",
    "MAX_MANIFEST_TEXT_CHARS",
    "MAX_TRUTH_REVIEWERS",
    "SplitRole",
    "TruthIdentity",
    "TruthState",
    "audit_benchmark_manifest",
]
