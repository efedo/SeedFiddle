"""Bind supplied classifier artifacts to an audited grouped evaluation.

This validates declared training provenance and image membership; it does not
authenticate study conduct, infer provenance from names, or certify science.
No original training images, masks, pickle files, or CUDA tensors are opened.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from seedvision.benchmarking.grouped_comparison import MethodBinding
from seedvision.benchmarking.manifest import (
    AccessPurpose, BenchmarkManifest, SplitRole, audit_benchmark_manifest,
)

MAX_MODEL_BYTES = 8 * 1024 * 1024
MAX_TRAINING_BYTES = 512 * 1024
MAX_RECEIPT_BYTES = 32 * 1024 * 1024
MAX_MEMBERSHIP_SAMPLE_IDS = 200_000


class ClassifierEvaluationError(ValueError):
    """An artifact or training declaration cannot support this evaluation."""


def _read(path, maximum):
    try:
        with Path(path).open('rb') as stream:
            raw = stream.read(maximum + 1)
    except (OSError, TypeError, ValueError) as error:
        raise ClassifierEvaluationError('Cannot read selected classifier evidence.') from error
    if not 0 < len(raw) <= maximum:
        raise ClassifierEvaluationError('Classifier evidence is empty or exceeds its byte limit.')
    return raw


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ClassifierEvaluationError('Duplicate training evidence JSON key.')
        result[key] = value
    return result


def _nonfinite(value):
    raise ClassifierEvaluationError('Nonfinite training evidence JSON number.')


def _artifacts(family, model_path, training_path):
    raw = _read(model_path, MAX_MODEL_BYTES)
    model_sha = sha256(raw).hexdigest()
    if family == 'naive_bayes':
        from seedvision.cuda.plantcv_naive_bayes import parse_plantcv_pdf
        from seedvision.model_io.naive_bayes_provenance import parse_training_record
        model = parse_plantcv_pdf(raw)
        if training_path is None:
            raise ClassifierEvaluationError('Bayes evaluation requires typed training evidence.')
        evidence = _read(training_path, MAX_TRAINING_BYTES)
        record = parse_training_record(evidence, model)
        metadata = record.metadata
        identity = {'tables_sha256': model.tables_sha256,
                    'class_names': list(model.class_names)}
        sources = ([row['image']['sha256'] for row in metadata['occurrences']]
                   if metadata['kind'] == 'native_bayes_directory_training' else None)
    else:
        from seedvision.model_io import kmeans_model_artifact as codec
        artifact = codec.parse_kmeans_artifact(raw)
        metadata, evidence = artifact.provenance, artifact.provenance_json
        if training_path is not None:
            selected = _read(training_path, MAX_TRAINING_BYTES)
            try:
                supplied = json.loads(selected, object_pairs_hook=_pairs, parse_constant=_nonfinite)
            except (UnicodeError, RecursionError) as error:
                raise ClassifierEvaluationError('Invalid training evidence JSON.') from error
            canonical = codec._canonical_metadata(supplied)
            if selected != canonical:
                raise ClassifierEvaluationError('Training evidence must use canonical JSON.')
            codec._validate_provenance(supplied, artifact.model)
            if evidence is not None and selected != evidence:
                raise ClassifierEvaluationError('Selected evidence differs from embedded training evidence.')
            metadata, evidence = supplied, selected
        if metadata is None:
            raise ClassifierEvaluationError('K-means evaluation requires typed training evidence.')
        identity = {'centres_sha256': artifact.model.centres_sha256,
                    'patch_size': artifact.model.patch_size, 'channels': artifact.model.channels}
        sources = ([row['source_sha256'] for row in metadata['manifest']['occurrences']
                    if row['status'] == 'accepted']
                   if metadata['kind'] == 'native_training' else None)
    return model_sha, sha256(evidence).hexdigest(), metadata['kind'], identity, sources


def bind_classifier_evaluation(manifest, *, family, model_path, training_path=None,
                               method_id, backend_id, recipe_sha256, training_access_ids):
    """Return a frozen MethodBinding and canonical, finite JSON audit receipt.

    Families are ``naive_bayes`` and ``kmeans``. Selected access records must
    have TRAINING purpose and contain development samples only. Every known
    consumed image must match a source SHA in those records. Converted models
    and Bayes sample tables can bind their typed bytes but return an explicitly
    ineligible receipt because their source-image membership is unavailable.
    Callers must require ``evaluation_eligible is True`` before evaluation.
    """
    if family not in ('naive_bayes', 'kmeans'):
        raise ClassifierEvaluationError('Unsupported classifier family.')
    if not isinstance(manifest, BenchmarkManifest):
        raise ClassifierEvaluationError('A typed BenchmarkManifest is required.')
    manifest = BenchmarkManifest.from_dict(manifest.to_dict())
    preliminary = MethodBinding(method_id, backend_id, recipe_sha256,
                                training_access_ids=training_access_ids)
    if not preliminary.training_access_ids:
        raise ClassifierEvaluationError('Classifier evaluation requires training access IDs.')
    audit = audit_benchmark_manifest(manifest)
    if not audit.evaluation_eligible:
        raise ClassifierEvaluationError('Benchmark manifest is not evaluation eligible: '
                                        + ', '.join(audit.eligibility_blockers))
    records = {record.access_id: record for record in manifest.access_records}
    samples = {sample.sample_id: sample for sample in manifest.samples}
    selected = []
    covered = set()
    for access_id in preliminary.training_access_ids:
        record = records.get(access_id)
        if record is None or record.purpose != AccessPurpose.TRAINING:
            raise ClassifierEvaluationError('Selected training access ID is absent or not training.')
        if any(samples[sample_id].split_role != SplitRole.DEVELOPMENT
               for sample_id in record.sample_ids):
            raise ClassifierEvaluationError('Training access may contain development samples only.')
        covered.update(record.sample_ids)
        selected.append(record.to_dict())
    model_sha, evidence_sha, kind, numeric_identity, sources = _artifacts(
        family, model_path, training_path)
    binding = MethodBinding(method_id, backend_id, recipe_sha256, model_sha,
                            evidence_sha, preliminary.training_access_ids)
    membership = None
    if sources is not None:
        by_source = {}
        for sample_id in sorted(covered):
            by_source.setdefault(samples[sample_id].source_sha256, []).append(sample_id)
        if not sources or any(source not in by_source for source in sources):
            raise ClassifierEvaluationError('Consumed training image is outside selected training access.')
        if sum(len(by_source[source]) for source in sources) > MAX_MEMBERSHIP_SAMPLE_IDS:
            raise ClassifierEvaluationError('Repeated source membership exceeds its receipt work limit.')
        membership = [{'source_sha256': source, 'sample_ids': by_source[source]}
                      for source in sources]
    payload = {
        'format': 'seedfiddle.classifier-evaluation-binding', 'version': 1,
        'family': family, 'binding': binding.to_dict(),
        'manifest_sha256': manifest.manifest_sha256, 'manifest_audit': audit.to_dict(),
        'training_kind': kind, 'numeric_identity': numeric_identity,
        'training_access_records': selected,
        'source_membership': membership,
        'source_membership_state': 'verified_declared_membership' if sources is not None else 'unavailable',
        'evaluation_eligible': sources is not None,
        'unavailability_reason': None if sources is not None else 'typed evidence lacks consumed image source hashes',
        'scientifically_validated': False,
    }
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    if len(text.encode('utf-8')) > MAX_RECEIPT_BYTES:
        raise ClassifierEvaluationError('Classifier evaluation receipt exceeds its byte limit.')
    return binding, text
