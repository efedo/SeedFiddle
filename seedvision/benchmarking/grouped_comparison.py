"""Prespecified paired group comparisons of compact, revision-bound observations.

This engine checks supplied declarations and computes numerical uncertainty. It
does not authenticate study conduct, acquire truth, or certify science. Missing
and failed observations remain in coverage denominators, never become scores.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from itertools import islice
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import numpy as np

from seedvision.benchmarking.manifest import (
    BenchmarkManifest, SplitRole, _group_values, audit_benchmark_manifest,
)

FORMAT = "seedfiddle.grouped-comparison"
VERSION = 1
MAX_INPUT_BYTES = 8 * 1024 * 1024
MAX_REPORT_BYTES = 32 * 1024 * 1024
MAX_RECORDS = 20_000
MAX_METRICS = 32
MAX_STRATA = 32
MAX_GROUPS = 10_000
MAX_REPETITIONS = 10_000
MAX_RESAMPLING_VISITS = 10_000_000
MAX_METRIC_CELLS = 640_000
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")


class GroupedComparisonError(ValueError):
    """Invalid, mismatched, or computationally unsupported study declarations."""


def _text(value, name, *, identifier=False):
    if (type(value) is not str or not value or value != value.strip()
            or len(value) > 4096 or "\0" in value
            or (identifier and not _ID.fullmatch(value))):
        raise GroupedComparisonError(f"Invalid {name}.")
    return value


def _digest(value, name, *, optional=False):
    if optional and value is None:
        return
    if type(value) is not str or not _SHA.fullmatch(value):
        raise GroupedComparisonError(f"Invalid {name} SHA-256.")


def _number(value, name, *, optional=False):
    if optional and value is None:
        return
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise GroupedComparisonError(f"{name} must be finite numeric data.")


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise GroupedComparisonError(f"{name} must be an integer in [{low}, {high}].")


def _json(value, maximum=MAX_INPUT_BYTES):
    try:
        data = bytearray()
        encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"),
                                   ensure_ascii=False, allow_nan=False)
        for chunk in encoder.iterencode(value):
            encoded = chunk.encode('utf-8')
            if len(data) + len(encoded) > maximum:
                raise GroupedComparisonError("Encoded grouped-comparison byte limit exceeded.")
            data.extend(encoded)
    except (TypeError, ValueError, OverflowError) as exc:
        if isinstance(exc, GroupedComparisonError):
            raise
        raise GroupedComparisonError("Invalid finite JSON data.") from exc
    return bytes(data)


def _exact(value, keys, name):
    if type(value) is not dict or set(value) != set(keys.split()):
        raise GroupedComparisonError(f"Unexpected {name} fields.")
    return dict(value)


def _pairs(value, name, *, numeric=False, maximum=MAX_METRICS):
    if type(value) is not tuple or len(value) > maximum:
        raise GroupedComparisonError(f"{name} must be a bounded immutable tuple.")
    names = []
    for pair in value:
        if type(pair) is not tuple or len(pair) != 2:
            raise GroupedComparisonError(f"Invalid {name} entry.")
        _text(pair[0], name, identifier=True)
        if numeric:
            _number(pair[1], name, optional=True)
        else:
            _text(pair[1], name)
        names.append(pair[0])
    if len(set(names)) != len(names):
        raise GroupedComparisonError(f"Duplicate {name} keys.")


@dataclass(frozen=True, slots=True)
class MethodBinding:
    method_id: str
    backend_id: str
    recipe_sha256: str
    model_sha256: str | None = None
    training_evidence_sha256: str | None = None
    training_access_ids: tuple[str, ...] = ()

    def __post_init__(self):
        _text(self.method_id, "method ID", identifier=True)
        _text(self.backend_id, "backend ID", identifier=True)
        for name in ("recipe_sha256", "model_sha256", "training_evidence_sha256"):
            _digest(getattr(self, name), name, optional=name != "recipe_sha256")
        if type(self.training_access_ids) is not tuple or len(self.training_access_ids) > 1024:
            raise GroupedComparisonError("Access IDs must be a bounded immutable tuple.")
        for identifier in self.training_access_ids:
            _text(identifier, "access ID", identifier=True)
        if len(set(self.training_access_ids)) != len(self.training_access_ids):
            raise GroupedComparisonError("Duplicate access IDs.")
        object.__setattr__(self, 'training_access_ids', tuple(sorted(self.training_access_ids)))

    def to_dict(self):
        return {name: (list(value) if name == "training_access_ids" else value)
                for name, value in ((name, getattr(self, name)) for name in self.__dataclass_fields__)}

    @property
    def identity_sha256(self):
        return sha256(_json(self.to_dict())).hexdigest()

    @classmethod
    def from_dict(cls, value):
        data = _exact(value, "method_id backend_id recipe_sha256 model_sha256 "
                      "training_evidence_sha256 training_access_ids", "method")
        if type(data["training_access_ids"]) is not list:
            raise GroupedComparisonError("Access IDs must be an array.")
        data["training_access_ids"] = tuple(data["training_access_ids"])
        return cls(**data)


@dataclass(frozen=True, slots=True)
class MetricGate:
    name: str
    direction: str
    absolute_limit: float
    noninferiority_margin: float

    def __post_init__(self):
        _text(self.name, "metric name", identifier=True)
        if self.direction not in ("higher", "lower"):
            raise GroupedComparisonError("Metric direction must be higher or lower.")
        _number(self.absolute_limit, "absolute limit")
        _number(self.noninferiority_margin, "noninferiority margin")
        if self.noninferiority_margin < 0:
            raise GroupedComparisonError("Noninferiority margin cannot be negative.")

    def to_dict(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value):
        return cls(**_exact(value, "name direction absolute_limit noninferiority_margin", "metric"))


@dataclass(frozen=True, slots=True)
class CriticalStratum:
    name: str
    selectors: tuple[tuple[str, str], ...]

    def __post_init__(self):
        _text(self.name, "stratum name", identifier=True)
        _pairs(self.selectors, "stratum selectors", maximum=16)
        if not self.selectors or self.name == "overall":
            raise GroupedComparisonError("Critical stratum needs selectors and a nonreserved name.")
        object.__setattr__(self, 'selectors', tuple(sorted(self.selectors)))

    def to_dict(self):
        return {"name": self.name, "selectors": dict(self.selectors)}

    @classmethod
    def from_dict(cls, value):
        data = _exact(value, "name selectors", "stratum")
        if type(data["selectors"]) is not dict:
            raise GroupedComparisonError("Selectors must be an object.")
        return cls(data["name"], tuple(sorted(data["selectors"].items())))


@dataclass(frozen=True, slots=True)
class ComparisonPlan:
    manifest_sha256: str
    reference: MethodBinding
    candidate: MethodBinding
    metrics: tuple[MetricGate, ...]
    critical_strata: tuple[CriticalStratum, ...] = ()
    split_role: str = "locked_test"
    aggregation: str = "group_mean"
    minimum_independent_groups: int = 2
    minimum_coverage: float = 1.0
    repetitions: int = 1000
    seed: int = 0
    format: str = "seedfiddle.grouped-comparison-plan"
    version: int = 1

    def __post_init__(self):
        if self.format != 'seedfiddle.grouped-comparison-plan' or type(self.version) is not int or self.version != 1:
            raise GroupedComparisonError('Unsupported comparison plan format/version.')
        _digest(self.manifest_sha256, "manifest")
        if not isinstance(self.reference, MethodBinding) or not isinstance(self.candidate, MethodBinding):
            raise GroupedComparisonError("Plan needs exact method bindings.")
        if self.reference.method_id == self.candidate.method_id:
            raise GroupedComparisonError("Comparison method IDs must differ.")
        for items, kind, maximum in ((self.metrics, MetricGate, MAX_METRICS),
                                     (self.critical_strata, CriticalStratum, MAX_STRATA)):
            if type(items) is not tuple or len(items) > maximum or any(not isinstance(i, kind) for i in items):
                raise GroupedComparisonError("Plan collections must be bounded typed tuples.")
            names = [i.name for i in items]
            if len(set(names)) != len(names):
                raise GroupedComparisonError("Duplicate plan names.")
        if not self.metrics:
            raise GroupedComparisonError("At least one prespecified metric is required.")
        if self.split_role not in {i.value for i in SplitRole} or self.aggregation not in ("group_mean", "image_mean"):
            raise GroupedComparisonError("Unsupported split or aggregation.")
        _integer(self.minimum_independent_groups, "minimum groups", 2, MAX_GROUPS)
        _integer(self.repetitions, "repetitions", 100, MAX_REPETITIONS)
        _integer(self.seed, "seed", 0, 2**63 - 1)
        _number(self.minimum_coverage, "minimum coverage")
        if not 0 <= self.minimum_coverage <= 1:
            raise GroupedComparisonError("Coverage must lie in [0,1].")
        _json(self.to_dict())

    def to_dict(self):
        return {"manifest_sha256": self.manifest_sha256,
                "format": self.format, "version": self.version,
                "reference": self.reference.to_dict(), "candidate": self.candidate.to_dict(),
                "metrics": [i.to_dict() for i in self.metrics],
                "critical_strata": [i.to_dict() for i in self.critical_strata],
                **{name: getattr(self, name) for name in ("split_role", "aggregation",
                    "minimum_independent_groups", "minimum_coverage", "repetitions", "seed")}}

    @property
    def plan_sha256(self):
        return sha256(_json(self.to_dict())).hexdigest()

    @classmethod
    def from_dict(cls, value):
        data = _exact(value, "manifest_sha256 reference candidate metrics critical_strata split_role "
                      "aggregation minimum_independent_groups minimum_coverage repetitions seed format version", "plan")
        data["reference"] = MethodBinding.from_dict(data["reference"])
        data["candidate"] = MethodBinding.from_dict(data["candidate"])
        for key, kind in (("metrics", MetricGate), ("critical_strata", CriticalStratum)):
            if type(data[key]) is not list:
                raise GroupedComparisonError("Plan collections must be arrays.")
            data[key] = tuple(kind.from_dict(i) for i in data[key])
        return cls(**data)


def truth_identity_sha256(sample):
    """Bind exactly the manifest's declared truth, including unavailable state."""
    return sha256(_json(sample.truth.to_dict())).hexdigest()


@dataclass(frozen=True, slots=True)
class ComparisonRecord:
    sample_id: str
    sample_sha256: str
    truth_sha256: str
    method_id: str
    method_sha256: str
    output_revision: str | None
    status: str
    metrics: tuple[tuple[str, float | None], ...]
    strata: tuple[tuple[str, str], ...] = ()
    reason: str | None = None
    gallery_links: tuple[str, ...] = ()

    def __post_init__(self):
        _text(self.sample_id, "sample ID", identifier=True)
        _text(self.method_id, "method ID", identifier=True)
        for name in ("sample_sha256", "truth_sha256", "method_sha256", "output_revision"):
            _digest(getattr(self, name), name, optional=name == "output_revision")
        if self.status not in ("available", "failed", "abstained", "unavailable"):
            raise GroupedComparisonError("Unsupported observation status.")
        _pairs(self.metrics, "metrics", numeric=True)
        _pairs(self.strata, "strata", maximum=16)
        object.__setattr__(self, 'metrics', tuple(sorted(self.metrics)))
        object.__setattr__(self, 'strata', tuple(sorted(self.strata)))
        if self.status == "available" and self.output_revision is None:
            raise GroupedComparisonError("Available observations require an output revision.")
        if self.status != "available" and any(v is not None for _, v in self.metrics):
            raise GroupedComparisonError("Uncompleted observations cannot claim metric scores.")
        if self.reason is not None:
            _text(self.reason, "reason")
        if (self.status != "available" or any(v is None for _, v in self.metrics)) and self.reason is None:
            raise GroupedComparisonError("Unavailable observations/metrics need a reason.")
        if type(self.gallery_links) is not tuple or len(self.gallery_links) > 8:
            raise GroupedComparisonError("Gallery links must be a bounded tuple.")
        for link in self.gallery_links:
            _text(link, "gallery link")

    def to_dict(self):
        return {"sample_id": self.sample_id, "sample_sha256": self.sample_sha256,
                "truth_sha256": self.truth_sha256, "method_id": self.method_id,
                "method_sha256": self.method_sha256, "output_revision": self.output_revision,
                "status": self.status, "metrics": dict(self.metrics), "strata": dict(self.strata),
                "reason": self.reason, "gallery_links": list(self.gallery_links)}

    @classmethod
    def from_dict(cls, value):
        data = _exact(value, "sample_id sample_sha256 truth_sha256 method_id method_sha256 "
                      "output_revision status metrics strata reason gallery_links", "observation")
        for key in ("metrics", "strata"):
            if type(data[key]) is not dict:
                raise GroupedComparisonError("Observation metrics/strata must be objects.")
            data[key] = tuple(sorted(data[key].items()))
        if type(data["gallery_links"]) is not list:
            raise GroupedComparisonError("Gallery links must be an array.")
        data["gallery_links"] = tuple(data["gallery_links"])
        return cls(**data)


def _independent_groups(manifest):
    # Same declared group identities as the leakage audit, plus source duplicates.
    parents = list(range(len(manifest.samples)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    seen = {}
    for i, sample in enumerate(manifest.samples):
        for key in (("source", sample.source_sha256), *_group_values(sample)):
            if key in seen:
                a, b = find(i), find(seen[key])
                parents[max(a, b)] = min(a, b)
            else:
                seen[key] = i
    members = {}
    for i, sample in enumerate(manifest.samples):
        members.setdefault(find(i), []).append(sample.sample_id)
    result = {}
    for ids in members.values():
        identity = sha256(_json(sorted(ids))).hexdigest()
        result.update((identifier, identity) for identifier in ids)
    return result


def _finite(value):
    value = float(value)
    if not math.isfinite(value):
        raise GroupedComparisonError("Comparison arithmetic is outside its finite numeric domain.")
    return value


def _sum(values):
    try:
        return _finite(math.fsum(values))
    except (OverflowError, ValueError) as exc:
        if isinstance(exc, GroupedComparisonError):
            raise
        raise GroupedComparisonError('Comparison arithmetic is outside its finite numeric domain.') from exc


def _mean(values):
    return _sum(float(v) / len(values) for v in values)


def evaluate_grouped_comparison(manifest, records, plan):
    """Compute paired percentile bounds; all gates are numerical, never promotion.

    Absolute gates use all available candidate observations and their one-sided
    95% bound. Comparative gates use only jointly available pairs and
    use candidate-minus-reference: lower 5% >= -margin for higher, upper 95%
    <= margin for lower. Every prespecified critical stratum must pass.
    """
    if not isinstance(manifest, BenchmarkManifest) or not isinstance(plan, ComparisonPlan):
        raise GroupedComparisonError("Expected BenchmarkManifest and ComparisonPlan.")
    if manifest.manifest_sha256 != plan.manifest_sha256:
        raise GroupedComparisonError("Plan manifest identity mismatch.")
    try:
        records = tuple(islice(iter(records), MAX_RECORDS + 1))
    except TypeError as exc:
        raise GroupedComparisonError('Observations must be a typed iterable.') from exc
    if len(records) > MAX_RECORDS or any(not isinstance(i, ComparisonRecord) for i in records):
        raise GroupedComparisonError("Observation collection exceeds its typed record limit.")
    if len(records) * len(plan.metrics) > MAX_METRIC_CELLS:
        raise GroupedComparisonError("Metric-cell budget exceeded.")
    _json([i.to_dict() for i in records])
    methods = {i.method_id: i for i in (plan.reference, plan.candidate)}
    accesses = {i.access_id for i in manifest.access_records}
    if any(set(i.training_access_ids) - accesses for i in methods.values()):
        raise GroupedComparisonError("Method names undeclared access records.")
    samples = {i.sample_id: i for i in manifest.samples if i.split_role.value == plan.split_role}
    if not samples:
        raise GroupedComparisonError("Selected split has no manifest samples.")
    if len(samples) * 2 > MAX_RECORDS or len(samples) * 2 * len(plan.metrics) > MAX_METRIC_CELLS:
        raise GroupedComparisonError("Planned observation/metric-cell budget exceeded.")
    lookup = {}
    names = {i.name for i in plan.metrics}
    for record in records:
        sample = samples.get(record.sample_id)
        method = methods.get(record.method_id)
        if sample is None or method is None:
            raise GroupedComparisonError("Observation is outside the selected sample/method inventory.")
        if (record.sample_sha256 != sample.identity_sha256
                or record.truth_sha256 != truth_identity_sha256(sample)
                or record.method_sha256 != method.identity_sha256):
            raise GroupedComparisonError("Observation sample/truth/method identity mismatch.")
        if set(dict(record.metrics)) != names:
            raise GroupedComparisonError("Observation metrics differ from prespecified endpoints.")
        if sample.truth.state.value != "available" and any(v is not None for _, v in record.metrics):
            raise GroupedComparisonError("Unavailable truth cannot support metric scores.")
        key = (record.sample_id, record.method_id)
        if key in lookup:
            raise GroupedComparisonError("Duplicate sample/method observation.")
        lookup[key] = record
    groups = _independent_groups(manifest)
    selected_groups = sorted({groups[i] for i in samples})
    if len(selected_groups) > MAX_GROUPS:
        raise GroupedComparisonError("Independent-group limit exceeded.")
    budget = 2 * len(selected_groups) * plan.repetitions * len(plan.metrics) * (1 + len(plan.critical_strata))
    if budget > MAX_RESAMPLING_VISITS:
        raise GroupedComparisonError("Paired resampling work budget exceeded.")
    audit = audit_benchmark_manifest(manifest)
    images = []
    strata_keys = {key for s in plan.critical_strata for key, _ in s.selectors}
    unknown_strata = []
    for identifier, sample in sorted(samples.items()):
        present = [lookup.get((identifier, m)) for m in methods]
        declared = [dict(i.strata) for i in present if i is not None]
        if declared and any(i != declared[0] for i in declared[1:]):
            raise GroupedComparisonError("Paired observations disagree on stratum facts.")
        facts = declared[0] if declared else {}
        if "species" in facts and facts["species"] != sample.species:
            raise GroupedComparisonError("Stratum species disagrees with manifest.")
        facts = {**facts, "species": sample.species}
        if strata_keys - set(facts):
            unknown_strata.append(identifier)
        observations = {}
        for method_id in methods:
            record = lookup.get((identifier, method_id))
            observations[method_id] = record.to_dict() if record else {
                "status": "not_recorded", "reason": "No observation was supplied.",
                "metrics": dict.fromkeys(sorted(names)), "output_revision": None, "gallery_links": []}
        images.append({"sample_id": identifier, "sample_sha256": sample.identity_sha256,
                       "truth_sha256": truth_identity_sha256(sample), "group_id": groups[identifier],
                       "strata": facts, "observations": observations})
    scopes = [("overall", images)] + [(s.name, [i for i in images
        if all(i["strata"].get(k) == v for k, v in s.selectors)]) for s in plan.critical_strata]
    def outcome_counts(selected):
        counts = {m: dict.fromkeys(('available', 'failed', 'abstained', 'unavailable', 'not_recorded'), 0)
                  for m in methods}
        for image in selected:
            for m in methods:
                counts[m][image['observations'][m]['status']] += 1
        return counts
    grouped_images = {}
    for image in images:
        grouped_images.setdefault(image['group_id'], []).append(image)
    group_summaries = [{'group_id': g, 'sample_ids': [i['sample_id'] for i in members],
                        'outcome_counts': outcome_counts(members)}
                       for g, members in sorted(grouped_images.items())]
    rng = np.random.Generator(np.random.PCG64(plan.seed))
    summaries = []
    for scope, selected in scopes:
        endpoints = []
        for metric in plan.metrics:
            paired = []
            candidate_values = {}
            available = {m: 0 for m in methods}
            for image in selected:
                obs = image["observations"]
                valid = {m: obs[m]["status"] == "available" and obs[m]["metrics"][metric.name] is not None
                         for m in methods}
                for m in methods:
                    available[m] += int(valid[m])
                if valid[plan.candidate.method_id]:
                    candidate_values.setdefault(image["group_id"], []).append(
                        obs[plan.candidate.method_id]["metrics"][metric.name])
                if all(valid.values()):
                    paired.append((image["group_id"], obs[plan.reference.method_id]["metrics"][metric.name],
                                   obs[plan.candidate.method_id]["metrics"][metric.name]))
            by_group = {}
            for group_id, reference, candidate in paired:
                by_group.setdefault(group_id, []).append((reference, candidate))
            aggregates = [(g, _mean([x[0] for x in values]), _mean([x[1] for x in values]), len(values))
                          for g, values in sorted(by_group.items())]
            absolute_groups = [(g, _mean(values), len(values))
                               for g, values in sorted(candidate_values.items())]
            coverage = len(paired) / len(selected) if selected else 0.0
            candidate_coverage = available[plan.candidate.method_id] / len(selected) if selected else 0.0
            common_blockers = []
            if not audit.evaluation_eligible:
                common_blockers.append("manifest_audit_ineligible")
            if scope != "overall" and unknown_strata:
                common_blockers.append("stratum_membership_not_recorded")
            paired_blockers, candidate_blockers = list(common_blockers), list(common_blockers)
            if len(aggregates) < plan.minimum_independent_groups:
                paired_blockers.append("insufficient_paired_groups")
            if coverage < plan.minimum_coverage:
                paired_blockers.append("insufficient_paired_coverage")
            if len(absolute_groups) < plan.minimum_independent_groups:
                candidate_blockers.append("insufficient_candidate_groups")
            if candidate_coverage < plan.minimum_coverage:
                candidate_blockers.append("insufficient_candidate_coverage")
            blockers = sorted(set(paired_blockers + candidate_blockers))
            result = {"metric": metric.name, "direction": metric.direction, "paired_images": len(paired),
                      "total_images": len(selected), "paired_coverage": coverage,
                      "candidate_coverage": candidate_coverage,
                      "available_images": available, "independent_groups": len(aggregates),
                      "paired_independent_groups": len(aggregates),
                      "candidate_independent_groups": len(absolute_groups),
                      "groups": [{"group_id": g, "reference": r, "candidate": c, "paired_images": n}
                                 for g, r, c, n in aggregates], "blockers": blockers,
                      "candidate_groups": [{"group_id": g, "candidate": c, "available_images": n}
                                           for g, c, n in absolute_groups],
                      "paired_blockers": paired_blockers, "candidate_blockers": candidate_blockers,
                      "reference_estimate": None, "candidate_estimate": None, "difference_estimate": None,
                      "candidate_absolute_estimate": None,
                      "candidate_one_sided_bound": None, "difference_one_sided_bound": None,
                      "absolute_pass": None, "noninferiority_pass": None, "state": "inconclusive"}
            if aggregates:
                array = np.asarray([[r, c, n] for _, r, c, n in aggregates], dtype=np.float64)
                def aggregate(draw):
                    weights = draw[:, 2] if plan.aggregation == "image_mean" else np.ones(len(draw))
                    weights = weights / weights.sum()
                    ref = _sum(float(v) for v in draw[:, 0] * weights)
                    cand = _sum(float(v) for v in draw[:, 1] * weights)
                    return ref, cand, _finite(cand - ref)
                r, c, delta = aggregate(array)
                result.update(reference_estimate=r, candidate_estimate=c, difference_estimate=delta)
                if not paired_blockers:
                    deltas = []
                    for _ in range(plan.repetitions):
                        draw = array[rng.integers(0, len(array), size=len(array))]
                        _, _, difference = aggregate(draw)
                        deltas.append(difference)
                    quantile = .05 if metric.direction == "higher" else .95
                    db = _finite(np.quantile(deltas, quantile, method="linear"))
                    comparative = db >= -metric.noninferiority_margin if metric.direction == "higher" else db <= metric.noninferiority_margin
                    result.update(difference_one_sided_bound=db, noninferiority_pass=comparative)
            if absolute_groups:
                absolute_array = np.asarray([[c, n] for _, c, n in absolute_groups], dtype=np.float64)
                def absolute_aggregate(draw):
                    weights = draw[:, 1] if plan.aggregation == "image_mean" else np.ones(len(draw))
                    weights = weights / weights.sum()
                    return _sum(float(v) for v in draw[:, 0] * weights)
                result['candidate_absolute_estimate'] = absolute_aggregate(absolute_array)
                if not candidate_blockers:
                    distribution = [absolute_aggregate(absolute_array[rng.integers(
                        0, len(absolute_array), size=len(absolute_array))]) for _ in range(plan.repetitions)]
                    cb = _finite(np.quantile(distribution, .05 if metric.direction == 'higher' else .95, method='linear'))
                    result.update(candidate_one_sided_bound=cb, absolute_pass=(
                        cb >= metric.absolute_limit if metric.direction == 'higher' else cb <= metric.absolute_limit))
            if not blockers:
                result['state'] = 'pass' if result['absolute_pass'] and result['noninferiority_pass'] else 'fail'
            endpoints.append(result)
        summaries.append({"name": scope, "total_images": len(selected),
                          "outcome_counts": outcome_counts(selected), "metrics": endpoints})
    states = [e["state"] for scope in summaries for e in scope["metrics"]]
    report = {"format": FORMAT, "version": VERSION, "scientifically_validated": False,
              "interpretation": "Numerical gates on supplied declarations; not scientific certification or promotion.",
              "manifest_sha256": manifest.manifest_sha256, "plan_sha256": plan.plan_sha256,
              "records_sha256": sha256(_json([i.to_dict() for i in sorted(records, key=lambda x: (x.sample_id, x.method_id))])).hexdigest(),
              "plan": plan.to_dict(), "manifest_audit": audit.to_dict(),
              "resampling": {"unit": "connected biological/acquisition/source groups", "algorithm": "PCG64 paired percentile bootstrap",
                             "confidence": .95, "absolute_rule": "candidate one-sided bound", "joint_methods": True,
                             "metric_population": "noninferiority: jointly available pairs; absolute: all available candidate observations; both coverage denominators include every planned image"},
              "images": images, "groups": group_summaries, "strata": summaries,
              "unknown_stratum_samples": unknown_strata,
              "state": "inconclusive" if "inconclusive" in states else "pass" if all(s == "pass" for s in states) else "fail"}
    _json(report, MAX_REPORT_BYTES)
    return report


def _decode(data):
    if type(data) is not bytes or len(data) > MAX_INPUT_BYTES:
        raise GroupedComparisonError("Input byte limit exceeded.")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise GroupedComparisonError("Duplicate JSON fields.")
            result[key] = value
        return result
    def nonfinite(_):
        raise GroupedComparisonError('Nonfinite JSON.')
    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=pairs,
                          parse_constant=nonfinite)
    except (UnicodeError, ValueError, RecursionError) as exc:
        if isinstance(exc, GroupedComparisonError):
            raise
        raise GroupedComparisonError("Invalid grouped comparison JSON.") from exc


def load_comparison_inputs(path):
    """Strict bounded file containing exactly manifest, plan, and records."""
    with Path(path).open("rb") as stream:
        payload = _decode(stream.read(MAX_INPUT_BYTES + 1))
    data = _exact(payload, "manifest plan records", "comparison input")
    if type(data["records"]) is not list or len(data["records"]) > MAX_RECORDS:
        raise GroupedComparisonError("Invalid observation collection.")
    return (BenchmarkManifest.from_dict(data["manifest"]),
            tuple(ComparisonRecord.from_dict(i) for i in data["records"]),
            ComparisonPlan.from_dict(data["plan"]))


def export_grouped_comparison(path, manifest, records, plan):
    """Recompute/validate before atomic publication; preserve prior destination."""
    report = evaluate_grouped_comparison(manifest, records, plan)
    data = _json(report, MAX_REPORT_BYTES)
    target = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return report
