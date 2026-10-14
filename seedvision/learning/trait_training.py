"""Explicit native trait evidence, CUDA materialization and atomic model saving."""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from seedvision.annotation.coordinates import validated_transform
from seedvision.learning.trait_classifier import (
    FrozenTraitModel, TraitDefinition, TraitExample, TraitTrainingEvidence,
    TraitVocabulary, _canonical, _consistent_truth, _disjoint, _integer, _number,
    _pairs, _text, check_trait_cancellation, train_trait_model,
)
from seedvision.learning.trait_features import TRAIT_FEATURE_SCHEMA, extract_trait_features
from seedvision.persistence.reference_regions import (
    ReferenceRegionStore, validated_source_sha256,
)

_MAX_SOURCE_BYTES = 512 * 1024 * 1024
_MAX_ARCHIVE_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class TraitTrainingConfiguration:
    iterations: int = 200
    learning_rate: float = .03
    l2: float = .01
    temperature_bounds: tuple[float, float] = (.05, 20.)
    calibration_iterations: int = 65
    abstention_threshold: float = .8

    def __post_init__(self):
        _integer(self.iterations, 'Training iterations', 1, 10000)
        _number(self.learning_rate, 'Learning rate', positive=True, maximum=1)
        _number(self.l2, 'L2', minimum=0)
        _integer(self.calibration_iterations, 'Calibration iterations', 2, 1024)
        _number(self.abstention_threshold, 'Abstention threshold', minimum=0, maximum=1)
        if not isinstance(self.temperature_bounds, (tuple, list)) or len(self.temperature_bounds) != 2:
            raise ValueError('Two temperature bounds are required.')
        object.__setattr__(self, 'temperature_bounds', tuple(self.temperature_bounds))
        lower, upper = self.temperature_bounds
        _number(lower, 'Lower temperature', positive=True)
        _number(upper, 'Upper temperature', positive=True)
        if lower >= upper:
            raise ValueError('Temperature bounds must increase.')


@dataclass(frozen=True, slots=True)
class TraitTrainingRequest:
    manifest_path: Path
    output_path: Path
    configuration: TraitTrainingConfiguration = TraitTrainingConfiguration()
    activate_after_training: bool = False
    manifest_sha256: str | None = None

    def __post_init__(self):
        object.__setattr__(self, 'manifest_path', Path(self.manifest_path).expanduser().resolve())
        object.__setattr__(self, 'output_path', Path(self.output_path).expanduser().resolve())
        if not isinstance(self.configuration, TraitTrainingConfiguration) or type(self.activate_after_training) is not bool:
            raise ValueError('Explicit trait controls and activation choice are required.')
        validated_source_sha256(self.manifest_sha256, 'manifest SHA256')
        if self.manifest_path == self.output_path:
            raise ValueError('Model output must differ from the evidence manifest.')


@dataclass(frozen=True, slots=True)
class TraitManifest:
    canonical_json: str
    content_sha256: str
    directory: Path

    @property
    def payload(self):
        return json.loads(self.canonical_json)


def _exact(value, keys, name):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f'{name} requires exactly: {", ".join(keys)}.')


def _vocabulary(value):
    _exact(value, ('revision', 'definition_source', 'definition_source_sha256', 'traits'), 'Vocabulary')
    if not isinstance(value['traits'], list):
        raise ValueError('Vocabulary traits must be a list.')
    for item in value['traits']:
        _exact(item, ('identifier', 'definition', 'evidence_reference'), 'Trait definition')
    return TraitVocabulary(value['revision'], value['definition_source'], value['definition_source_sha256'],
                           tuple(TraitDefinition(**item) for item in value['traits']))


def read_trait_manifest(path, *, expected_sha256=None):
    """Metadata preflight does not access any example's image/archive input."""
    path = Path(path).resolve()
    with path.open('rb') as stream:
        content = stream.read(16 * 1024 * 1024 + 1)
    if len(content) > 16 * 1024 * 1024:
        raise ValueError('Trait manifest exceeds 16 MiB.')
    digest = sha256(content).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError('Evidence manifest changed after the training request.')
    payload = json.loads(content.decode('utf-8'), object_pairs_hook=_pairs,
                         parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON constant.')))
    _exact(payload, ('format', 'version', 'feature_schema', 'vocabulary', 'examples'), 'Trait manifest')
    if payload['format'] != 'seedfiddle-trait-training' or type(payload['version']) is not int or payload['version'] != 1:
        raise ValueError('Unsupported trait manifest format/version.')
    if payload['feature_schema'] != TRAIT_FEATURE_SCHEMA:
        raise ValueError('Trait manifest does not match the CUDA feature schema.')
    vocabulary = _vocabulary(payload['vocabulary'])
    examples = payload['examples']
    if not isinstance(examples, list) or not 2 <= len(examples) <= 10000:
        raise ValueError('Supply two to ten thousand trait examples.')
    splits = {'train': [], 'calibration': []}
    for item in examples:
        _exact(item, ('example_id', 'physical_id', 'group_id', 'trait_id', 'truth_provenance',
                      'permitted_use', 'split', 'input'), 'Trait example')
        fact = TraitExample(**{key: item[key] for key in TraitExample.__dataclass_fields__})
        if item['split'] not in splits:
            raise ValueError('Use train or calibration splits.')
        splits[item['split']].append(fact)
        if not fact.permitted_use:
            continue  # Denied input contents and paths are never interpreted.
        if fact.trait_id not in {trait.identifier for trait in vocabulary.traits}:
            raise ValueError('Included trait ID is absent from the supplied vocabulary.')
        source = item['input']
        if not isinstance(source, dict):
            raise ValueError('Included trait input must be an object.')
        if source.get('kind') == 'compact_features':
            _exact(source, ('kind', 'values', 'provenance'), 'Compact trait input')
            _text(source['provenance'], 'Supplied feature provenance')
            if not isinstance(source['values'], list) or len(source['values']) != 20:
                raise ValueError('Compact input requires twenty schema-ordered values.')
            for value in source['values']:
                _number(value, 'Supplied feature value')
        elif source.get('kind') == 'reference_seed':
            _exact(source, ('kind', 'image_path', 'image_sha256', 'archive_path', 'archive_sha256', 'seed_id'), 'Reference seed input')
            for name in ('image_path', 'archive_path'):
                _text(source[name], name)
                if '\x00' in source[name]:
                    raise ValueError('Input paths cannot contain NUL.')
            for name in ('image_sha256', 'archive_sha256'):
                if validated_source_sha256(source[name], name) is None:
                    raise ValueError('Included inputs require explicit file digests.')
            _integer(source['seed_id'], 'Reference seed ID', 1, 2147483647)
        else:
            raise ValueError('Choose compact_features or reference_seed input.')
    for facts in splits.values():
        if not any(fact.permitted_use for fact in facts):
            raise ValueError('Both splits require permitted examples.')
        if len({fact.example_id for fact in facts}) != len(facts):
            raise ValueError('Example IDs must be unique.')
        _consistent_truth(facts)
    _disjoint(tuple(splits['train']), tuple(splits['calibration']))
    if {fact.trait_id for fact in splits['train'] if fact.permitted_use} != {trait.identifier for trait in vocabulary.traits}:
        raise ValueError('Every defined class needs permitted training examples.')
    return TraitManifest(_canonical(payload), digest, path.parent)


def _reference_features(source, fact, directory, device, cancellation_requested):
    image_path = (directory / source['image_path']).resolve()
    archive_path = (directory / source['archive_path']).resolve()
    snapshots = []
    for path, expected, limit in ((image_path, source['image_sha256'], _MAX_SOURCE_BYTES),
                                 (archive_path, source['archive_sha256'], _MAX_ARCHIVE_BYTES)):
        check_trait_cancellation(cancellation_requested)
        with path.open('rb') as stream:
            content = stream.read(limit + 1)
        if len(content) > limit:
            raise ValueError(f'Compressed input exceeds its byte limit: {path.name}.')
        if sha256(content).hexdigest() != expected:
            raise ValueError(f'Input SHA256 mismatch: {path.name}.')
        snapshots.append(content)
    check_trait_cancellation(cancellation_requested)
    image_content, archive_content = snapshots
    bundle = ReferenceRegionStore(directory).load_project_archive_snapshot(image_path, archive_path,
        image_content=image_content, archive_content=archive_content)
    if bundle is None or bundle.annotated_seeds is None:
        raise ValueError('Reference seed requires an existing validated label archive.')
    if bundle.source_to_corrected is None:
        raise ValueError('Reference seed requires a recorded source-to-corrected transform.')
    matrix = validated_transform(bundle.source_to_corrected)
    annotation = next((item for item in bundle.seed_annotations if item.seed_id == source['seed_id']), None)
    if annotation is None or not annotation.shape_reviewed or annotation.outline_visibility != 'complete' or annotation.shape_exclusion_reason:
        raise ValueError('Reference seed requires a reviewed complete eligible outline.')
    if annotation.physical_seed_id is not None and annotation.physical_seed_id != fact.physical_id:
        raise ValueError('Reference physical seed ID disagrees with supplied evidence.')
    mask = bundle.annotated_seeds == source['seed_id']
    if not mask.any() or mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
        raise ValueError('Reference seed is missing or touches the corrected canvas boundary.')
    # This CPU mask is an existing archive input, never a downloaded analysis raster.
    mask_sha = sha256(_canonical({'shape': list(mask.shape), 'transform': matrix.tolist()}).encode()
                      + np.packbits(mask).tobytes()).hexdigest()
    image = cv2.imdecode(np.frombuffer(image_content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Cannot decode the reference source image.')
    check_trait_cancellation(cancellation_requested)
    gpu_image = torch.as_tensor(image, device=device).permute(2, 0, 1)[None].to(torch.float64) / 255
    height, width = bundle.shape
    yy, xx = torch.meshgrid(torch.arange(height, device=device, dtype=torch.float64),
                            torch.arange(width, device=device, dtype=torch.float64), indexing='ij')
    points = torch.stack((xx, yy, torch.ones_like(xx)), -1) @ torch.linalg.inv(torch.as_tensor(matrix, device=device)).T
    denominator = points[..., 2]
    safe = torch.where(denominator != 0, denominator, torch.ones_like(denominator))
    x, y = points[..., 0] / safe, points[..., 1] / safe
    valid = (denominator != 0) & torch.isfinite(x) & torch.isfinite(y) & (x >= 0) & (y >= 0) & (x <= image.shape[1] - 1) & (y <= image.shape[0] - 1)
    gpu_mask = torch.as_tensor(mask, device=device)
    if bool((gpu_mask & ~valid).any().item()):
        raise ValueError('Reviewed seed lies outside decoded source support.')
    grid = torch.stack((x / max(image.shape[1] - 1, 1) * 2 - 1,
                        y / max(image.shape[0] - 1, 1) * 2 - 1), -1)
    grid = torch.where(valid[..., None], grid, torch.zeros_like(grid))
    corrected = F.grid_sample(gpu_image, grid[None], mode='bilinear', padding_mode='border', align_corners=True)[0].permute(1, 2, 0)
    labels = gpu_mask.to(torch.int64)  # Selected seed becomes ID1 without changing its pixels.
    frame = _canonical({'image_sha256': source['image_sha256'], 'shape': list(bundle.shape), 'transform': matrix.tolist()})
    batch = extract_trait_features(corrected, labels, (1,), (1,), encoding='bgr_unit', image_frame=frame, label_frame=frame)
    if batch.rejected:
        raise ValueError(f'Reference features rejected: {batch.rejected[0][1]}.')
    audit = dict(kind='reference_seed', image_sha256=source['image_sha256'], archive_sha256=source['archive_sha256'],
                 seed_id=source['seed_id'], selected_mask_sha256=mask_sha, shape=list(bundle.shape),
                 source_to_corrected=matrix.tolist(), annotation_snapshot_source_sha256=bundle.annotation_snapshot_source_sha256,
                 generation_state='unknown' if bundle.annotation_snapshot_source_sha256 is None else 'recorded_matching',
                 consumed_bytes='immutable hash-matching image/archive snapshots',
                 extractor=TRAIT_FEATURE_SCHEMA, representation='full recorded corrected canvas; reviewed complete visible seed pixels')
    return batch.values[0], audit, (source['image_sha256'], mask_sha)


def materialize_trait_evidence(manifest, *, device='cuda', cancellation_requested=None, progress_callback=None):
    payload = manifest.payload
    vocabulary = _vocabulary(payload['vocabulary'])
    device = torch.device(device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise ValueError('Native trait training requires CUDA.')
    rows = {'train': [], 'calibration': []}
    facts = {'train': [], 'calibration': []}
    audit, actual_inputs = [], {}
    for index, item in enumerate(payload['examples']):
        check_trait_cancellation(cancellation_requested)
        fact = TraitExample(**{key: item[key] for key in TraitExample.__dataclass_fields__})
        split = item['split']
        facts[split].append(fact)
        if not fact.permitted_use:
            rows[split].append(torch.zeros(20, dtype=torch.float64, device=device))
            audit.append(dict(example_id=fact.example_id, split=split, included=False, reason='use_not_permitted'))
            continue
        source = item['input']
        try:
            if source['kind'] == 'compact_features':
                values = torch.tensor(source['values'], dtype=torch.float64, device=device)
                lineage = dict(kind='compact_features', provenance=source['provenance'],
                               representation='supplied schema-ordered features; extractor lineage not verified')
            else:
                values, lineage, actual = _reference_features(source, fact, manifest.directory, device, cancellation_requested)
                if actual in actual_inputs and actual_inputs[actual] != split:
                    raise ValueError('Actual source/selected-mask evidence overlaps training and calibration.')
                actual_inputs[actual] = split
        except ValueError as error:
            raise ValueError(f'Example {fact.example_id}: {error}') from error
        rows[split].append(values)
        audit.append(dict(example_id=fact.example_id, split=split, included=True, **lineage))
        if progress_callback is not None:
            progress_callback('evidence', index + 1, len(payload['examples']))
    lineage_sha = sha256(_canonical(audit).encode()).hexdigest()
    evidence = {}
    for split in rows:
        provenance = _canonical(dict(manifest_sha256=manifest.content_sha256, input_audit_sha256=lineage_sha,
                                    split=split, feature_schema=TRAIT_FEATURE_SCHEMA,
                                    authority='supplied definitions, truth and grouping; no scientific validation'))
        evidence[split] = TraitTrainingEvidence(torch.stack(rows[split]), tuple(facts[split]), provenance)
    return evidence['train'], evidence['calibration'], vocabulary, audit


def load_frozen_trait_model(path):
    with Path(path).open('rb') as stream:
        content = stream.read(4 * 1024 * 1024 + 1)
    if len(content) > 4 * 1024 * 1024:
        raise ValueError('Frozen trait model exceeds 4 MiB.')
    return FrozenTraitModel.from_json(content.decode('utf-8'))


def activate_trait_model(pipeline, path, *, abstention_threshold=None):
    """Validate first; enable/path/threshold changes are one UI transaction."""
    path = Path(path).resolve()
    model = load_frozen_trait_model(path)
    threshold = model.payload['development_diagnostics']['abstention_threshold'] if abstention_threshold is None else abstention_threshold
    _number(threshold, 'Activation threshold', minimum=0, maximum=1)
    values = {'trait_model_path': str(path), 'trait_abstention_threshold': threshold}
    proposed = deepcopy(pipeline)
    proposed.set_parameters('trait_classifier', values)
    proposed.set_enabled('trait_classifier', True)
    snapshot = {key: (dict(node.parameters), node.enabled, node.status, node.status_detail)
                for key, node in pipeline.nodes.items()}
    revision, suspended = pipeline.revision, set(pipeline._connection_suspended)
    try:
        affected = set(pipeline.set_parameters('trait_classifier', values))
        affected.update(pipeline.set_enabled('trait_classifier', True))
    except Exception:
        for key, (parameters, enabled, status, detail) in snapshot.items():
            node = pipeline.nodes[key]
            node.parameters = parameters
            node.enabled, node.status, node.status_detail = enabled, status, detail
        pipeline.revision = revision
        pipeline._connection_suspended = suspended
        raise
    return tuple(sorted(affected)), model


def save_frozen_trait_model(model, path, *, cancellation_requested=None):
    """A failure/stop before replace preserves the old destination byte-for-byte."""
    path = Path(path)
    content = model.to_json()
    FrozenTraitModel.from_json(content)  # Validate bounded envelope before creating output.
    check_trait_cancellation(cancellation_requested)
    owned_path = None
    try:
        descriptor, owned_path = tempfile.mkstemp(prefix='.trait-model-', suffix='.json.tmp', dir=path.parent)
        with os.fdopen(descriptor, 'w', encoding='utf-8', newline='\n') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        check_trait_cancellation(cancellation_requested)
        os.replace(owned_path, path)
        owned_path = None
    finally:
        if owned_path is not None:
            Path(owned_path).unlink(missing_ok=True)


def train_trait_manifest(request, *, cancellation_requested=None, progress_callback=None):
    check_trait_cancellation(cancellation_requested)
    manifest = read_trait_manifest(request.manifest_path, expected_sha256=request.manifest_sha256)
    # Never overwrite permitted evidence inputs with the published model.
    for item in manifest.payload['examples']:
        if item['permitted_use'] and item['input']['kind'] == 'reference_seed':
            for name in ('image_path', 'archive_path'):
                if (manifest.directory / item['input'][name]).resolve() == request.output_path:
                    raise ValueError('Model output must differ from evidence input paths.')
    training, calibration, vocabulary, audit = materialize_trait_evidence(manifest,
        cancellation_requested=cancellation_requested, progress_callback=progress_callback)
    config = request.configuration
    model = train_trait_model(training, calibration, vocabulary, iterations=config.iterations,
        learning_rate=config.learning_rate, l2=config.l2, temperature_bounds=config.temperature_bounds,
        calibration_iterations=config.calibration_iterations, abstention_threshold=config.abstention_threshold,
        cancellation_requested=cancellation_requested, progress_callback=progress_callback)
    payload = model.payload
    payload['native_training_evidence'] = dict(version=1, manifest_sha256=manifest.content_sha256,
        input_audit=audit, definition_authority='supplied, not independently verified',
        generation_policy='known mismatches rejected; missing history remains unknown', scientifically_validated=False)
    model = FrozenTraitModel(_canonical(payload))
    save_frozen_trait_model(model, request.output_path, cancellation_requested=cancellation_requested)
    # A late stop keeps committed output and suppresses UI activation.
    return dict(output_path=str(request.output_path), model_sha256=model.model_sha256,
                manifest_sha256=manifest.content_sha256, diagnostics=model.payload['development_diagnostics'],
                saved=True, cancellation_after_save=bool(cancellation_requested and cancellation_requested()),
                scientifically_validated=False)
