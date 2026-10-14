"""Grouped CUDA trait classification with frozen JSON models and explicit abstention.

Training/calibration diagnostics establish software behavior, not biological
validation. Expert definitions and truth provenance must be supplied explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import re

import torch
from torch.nn import functional as F

from seedvision.learning.trait_features import (
    TRAIT_FEATURE_NAMES, TRAIT_FEATURE_SCHEMA, TraitFeatureBatch,
)


def _text(value, name):
    if not isinstance(value, str) or not value.strip() or len(value) > 4096:
        raise ValueError(f'{name} must be a nonempty bounded string.')
    return value


def _number(value, name, *, minimum=None, maximum=None, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite JSON number.')
    if (positive and value <= 0) or (minimum is not None and value < minimum) or (
            maximum is not None and value > maximum):
        raise ValueError(f'{name} is outside its supported range.')
    return float(value)


def _integer(value, name, minimum=1, maximum=100000):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f'{name} must be a bounded integer.')
    return value


def _canonical(payload):
    return json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f'Duplicate JSON key {key!r}.')
        result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class TraitDefinition:
    identifier: str
    definition: str
    evidence_reference: str

    def __post_init__(self):
        if not isinstance(self.identifier, str) or re.fullmatch(r'[a-z][a-z0-9_]{0,63}', self.identifier) is None:
            raise ValueError('Trait IDs must be stable lowercase identifiers.')
        _text(self.definition, 'Trait definition')
        _text(self.evidence_reference, 'Trait definition evidence')


@dataclass(frozen=True, slots=True)
class TraitVocabulary:
    revision: str
    definition_source: str
    definition_source_sha256: str
    traits: tuple[TraitDefinition, ...]

    def __post_init__(self):
        object.__setattr__(self, 'traits', tuple(self.traits))
        _text(self.revision, 'Vocabulary revision')
        _text(self.definition_source, 'Vocabulary source')
        if not isinstance(self.definition_source_sha256, str) or re.fullmatch(r'[0-9a-f]{64}', self.definition_source_sha256) is None:
            raise ValueError('Vocabulary source needs an explicit SHA256.')
        if not 2 <= len(self.traits) <= 64 or any(not isinstance(x, TraitDefinition) for x in self.traits):
            raise ValueError('Supply between two and sixty-four defined traits.')
        if len({x.identifier for x in self.traits}) != len(self.traits):
            raise ValueError('Trait vocabulary IDs must be unique.')

    def payload(self):
        return {'revision': self.revision, 'definition_source': self.definition_source,
                'definition_source_sha256': self.definition_source_sha256,
                'traits': [dict(identifier=x.identifier, definition=x.definition,
                                evidence_reference=x.evidence_reference) for x in self.traits]}


@dataclass(frozen=True, slots=True)
class TraitExample:
    example_id: str
    physical_id: str
    group_id: str
    trait_id: str
    truth_provenance: str
    permitted_use: bool

    def __post_init__(self):
        for name in ('example_id', 'physical_id', 'group_id', 'trait_id', 'truth_provenance'):
            _text(getattr(self, name), name)
        if type(self.permitted_use) is not bool:
            raise ValueError('Evidence use permission must be explicitly Boolean.')

    def payload(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class TraitTrainingEvidence:
    values: torch.Tensor
    facts: tuple[TraitExample, ...]
    provenance: str
    schema: str = TRAIT_FEATURE_SCHEMA

    def __post_init__(self):
        object.__setattr__(self, 'facts', tuple(self.facts))
        _text(self.provenance, 'Feature evidence provenance')
        if self.schema != TRAIT_FEATURE_SCHEMA:
            raise ValueError('Training features must use the selected pixel schema.')
        if not isinstance(self.values, torch.Tensor) or not self.values.is_cuda or not self.values.is_floating_point():
            raise ValueError('Training features must remain on CUDA.')
        if not 1 <= len(self.facts) <= 100000 or self.values.shape != (len(self.facts), 20):
            raise ValueError('Training facts must match bounded twenty-column CUDA rows.')
        if any(not isinstance(x, TraitExample) for x in self.facts):
            raise ValueError('Every feature row needs explicit evidence facts.')
        if len({x.example_id for x in self.facts}) != len(self.facts):
            raise ValueError('Training example IDs must be unique.')


def _audit(facts):
    return {'included': [x.payload() for x in facts if x.permitted_use],
            'excluded': [dict(x.payload(), exclusion_reason='use_not_permitted')
                         for x in facts if not x.permitted_use]}


def _consistent_truth(facts):
    labels = {}
    for fact in facts:
        if not fact.permitted_use:
            continue
        if fact.physical_id in labels and labels[fact.physical_id] != fact.trait_id:
            raise ValueError('Permitted views of one physical seed have contradictory trait labels.')
        labels[fact.physical_id] = fact.trait_id


def _disjoint(left, right):
    for key in ('example_id', 'group_id', 'physical_id'):
        if {getattr(x, key) for x in left}.intersection(getattr(x, key) for x in right):
            raise ValueError(f'Trait evidence splits overlap by {key}.')


def _included(evidence, vocabulary):
    indices = [index for index, fact in enumerate(evidence.facts) if fact.permitted_use]
    if not indices:
        raise ValueError('Trait evidence has no permitted examples.')
    classes = {x.identifier: index for index, x in enumerate(vocabulary.traits)}
    if any(evidence.facts[index].trait_id not in classes for index in indices):
        raise ValueError('Permitted evidence labels must map to supplied trait definitions.')
    values = evidence.values[indices].detach().to(torch.float64)
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError('Permitted feature evidence must be finite.')
    targets = torch.tensor([classes[evidence.facts[index].trait_id] for index in indices],
                           dtype=torch.int64, device=values.device)
    return values, targets


def _evidence_sha256(evidence, included_values):
    """Hash permitted compact per-seed features, not full-resolution rasters."""
    payload = {'feature_schema': evidence.schema, 'provenance': evidence.provenance,
               'included_facts': [x.payload() for x in evidence.facts if x.permitted_use],
               'included_feature_values': included_values.tolist()}
    return sha256(_canonical(payload).encode('utf-8')).hexdigest()


@torch.no_grad()
def _diagnostics(logits, targets, temperature, threshold):
    calibrated_logits = logits / temperature
    if not bool(torch.isfinite(calibrated_logits).all().item()):
        raise ValueError('Trait diagnostics are outside their finite calibrated numeric range.')
    probabilities = calibrated_logits.softmax(dim=1)
    confidence, predicted = probabilities.max(dim=1)
    correct = predicted == targets
    accepted = confidence >= threshold
    count = targets.numel()
    accepted_count = int(accepted.sum().item())
    brier = (probabilities - F.one_hot(targets, probabilities.shape[1])).square().sum(dim=1).mean()
    bins = []
    for index in range(10):
        lower, upper = index / 10, (index + 1) / 10
        mask = (confidence >= lower) & (confidence < upper if index < 9 else confidence <= upper)
        size = int(mask.sum().item())
        bins.append({'lower_inclusive': lower, 'upper': upper,
                     'upper_inclusive': index == 9, 'count': size,
                     'mean_confidence': float(confidence[mask].mean().item()) if size else None,
                     'observed_correct_fraction': float(correct[mask].double().mean().item()) if size else None})
    return {'count': count, 'nll': float(F.cross_entropy(calibrated_logits, targets).item()),
            'multiclass_brier': float(brier.item()), 'reliability_bins': bins,
            'abstention_threshold': threshold, 'accepted_count': accepted_count,
            'coverage': accepted_count / count,
            'accepted_error': float((~correct[accepted]).double().mean().item()) if accepted_count else None,
            'scientifically_validated': False,
            'interpretation': 'supplied-evidence diagnostics; predictive probabilities, not biological validation'}


def _validate_diagnostics(diagnostics, expected_count):
    if not isinstance(diagnostics, dict):
        raise ValueError('Development diagnostics must be a JSON object.')
    _text(diagnostics['interpretation'], 'Diagnostic interpretation')
    count = _integer(diagnostics['count'], 'Diagnostic count')
    if count != expected_count:
        raise ValueError('Development diagnostic count differs from included calibration evidence.')
    accepted = _integer(diagnostics['accepted_count'], 'Accepted diagnostic count', 0, count)
    coverage = _number(diagnostics['coverage'], 'Coverage', minimum=0, maximum=1)
    if coverage != accepted / count:
        raise ValueError('Development diagnostic coverage is inconsistent.')
    _number(diagnostics['abstention_threshold'], 'Diagnostic threshold', minimum=0, maximum=1)
    if accepted:
        _number(diagnostics['accepted_error'], 'Accepted error', minimum=0, maximum=1)
    elif diagnostics['accepted_error'] is not None:
        raise ValueError('Zero-coverage error must be unavailable.')
    _number(diagnostics['nll'], 'Diagnostic NLL', minimum=0)
    _number(diagnostics['uncalibrated_nll'], 'Uncalibrated diagnostic NLL', minimum=0)
    _number(diagnostics['multiclass_brier'], 'Diagnostic Brier', minimum=0, maximum=2)
    if diagnostics['scientifically_validated'] is not False:
        raise ValueError('Development diagnostics cannot certify science.')
    bins = diagnostics['reliability_bins']
    if not isinstance(bins, list) or len(bins) != 10:
        raise ValueError('Development diagnostics require ten declared reliability bins.')
    total = 0
    for index, item in enumerate(bins):
        lower = _number(item['lower_inclusive'], 'Reliability lower bound')
        upper = _number(item['upper'], 'Reliability upper bound')
        if lower != index / 10 or upper != (index + 1) / 10 or item['upper_inclusive'] is not (index == 9):
            raise ValueError('Reliability bin boundaries disagree with their declared convention.')
        size = _integer(item['count'], 'Reliability bin count', 0, count)
        total += size
        if size:
            _number(item['mean_confidence'], 'Reliability confidence', minimum=lower, maximum=upper)
            _number(item['observed_correct_fraction'], 'Reliability correctness', minimum=0, maximum=1)
        elif item['mean_confidence'] is not None or item['observed_correct_fraction'] is not None:
            raise ValueError('Empty reliability bins cannot invent statistics.')
    if total != count:
        raise ValueError('Reliability bin counts do not cover calibration evidence.')


def _validate_payload(payload):
    if not isinstance(payload, dict) or type(payload.get('schema_version')) is not int or payload['schema_version'] != 1:
        raise ValueError('Unsupported frozen trait model schema.')
    if payload.get('feature_schema') != TRAIT_FEATURE_SCHEMA or payload.get('feature_names') != list(TRAIT_FEATURE_NAMES):
        raise ValueError('Frozen model feature schema/order is incompatible.')
    if payload.get('scientifically_validated') is not False or payload.get('computation_dtype') != 'float64':
        raise ValueError('Frozen trait models cannot assert scientific validation.')
    try:
        v = payload['vocabulary']
        vocabulary = TraitVocabulary(v['revision'], v['definition_source'], v['definition_source_sha256'],
            tuple(TraitDefinition(**item) for item in v['traits']))
        mean, scale = payload['standardization']['mean'], payload['standardization']['scale']
        constant = payload['standardization']['constant_columns']
        weights, bias = payload['weights'], payload['bias']
        if len(mean) != 20 or len(scale) != 20 or len(weights) != len(vocabulary.traits) or len(bias) != len(weights):
            raise ValueError('Frozen trait model matrix dimensions disagree.')
        for value in mean: _number(value, 'Feature mean')
        for value in scale: _number(value, 'Feature scale', positive=True)
        if not isinstance(constant, list) or any(type(x) is not int or not 0 <= x < 20 for x in constant) or len(set(constant)) != len(constant):
            raise ValueError('Constant feature indices are invalid.')
        if any(scale[index] != 1 for index in constant):
            raise ValueError('Exactly constant training features use unit scale.')
        for row in weights:
            if not isinstance(row, list) or len(row) != 20: raise ValueError('Invalid trait weight row.')
            for value in row: _number(value, 'Trait weight')
        for value in bias: _number(value, 'Trait bias')
        calibration = payload['temperature_calibration']
        _number(calibration['temperature'], 'Temperature', positive=True)
        lower, upper = calibration['bounds']
        _number(lower, 'Temperature lower bound', positive=True)
        _number(upper, 'Temperature upper bound', positive=True)
        if not lower < upper or not lower <= calibration['temperature'] <= upper:
            raise ValueError('Temperature is outside its search bounds.')
        _integer(calibration['iterations'], 'Calibration search iterations', 2, 1024)
        if (calibration['method'] != 'bounded log-grid NLL minimization; smallest-temperature tie'
                or calibration['includes_uncalibrated_candidate'] is not (lower <= 1 <= upper)):
            raise ValueError('Unsupported temperature search convention.')
        config = payload['training_configuration']
        _integer(config['iterations'], 'Training iterations', 1, 10000)
        _number(config['learning_rate'], 'Learning rate', positive=True, maximum=1)
        _number(config['l2'], 'L2', minimum=0)
        if config['l2_convention'] != 'mean cross entropy + 0.5*l2*sum(weights squared)':
            raise ValueError('Unsupported trait regularization convention.')
        if config['optimizer'] != 'Adam; zero initialization':
            raise ValueError('Unsupported trait optimizer convention.')
        splits = payload['split_audit']
        included = []
        for name in ('training', 'calibration'):
            split = splits[name]
            if not isinstance(split['included'], list) or not 1 <= len(split['included']) <= 100000:
                raise ValueError('A frozen model needs permitted evidence in each split.')
            facts = tuple(TraitExample(**item) for item in split['included'])
            if any(not x.permitted_use or x.trait_id not in {t.identifier for t in vocabulary.traits} for x in facts):
                raise ValueError('Frozen included evidence contradicts permission/vocabulary.')
            if len({x.example_id for x in facts}) != len(facts):
                raise ValueError('Frozen included evidence IDs are not unique.')
            if not isinstance(split['excluded'], list) or len(split['excluded']) > 100000:
                raise ValueError('Frozen excluded evidence must be bounded.')
            excluded = []
            for item in split['excluded']:
                item = dict(item)
                if item.pop('exclusion_reason') != 'use_not_permitted':
                    raise ValueError('Unsupported evidence exclusion.')
                fact = TraitExample(**item)
                if fact.permitted_use: raise ValueError('Excluded evidence cannot be permitted.')
                excluded.append(fact)
            all_facts = facts + tuple(excluded)
            if len({x.example_id for x in all_facts}) != len(all_facts):
                raise ValueError('Frozen evidence IDs are not unique across inclusion decisions.')
            included.append(all_facts)
            _consistent_truth(all_facts)
        _disjoint(*included)
        if {x.trait_id for x in included[0] if x.permitted_use} != {x.identifier for x in vocabulary.traits}:
            raise ValueError('Every frozen class needs permitted training evidence.')
        for name in ('training_provenance', 'calibration_provenance'):
            _text(payload[name], name)
        for name in ('training', 'calibration'):
            value = payload['evidence_sha256'][name]
            if not isinstance(value, str) or re.fullmatch(r'[0-9a-f]{64}', value) is None:
                raise ValueError('Frozen evidence identity requires explicit SHA256 values.')
        _validate_diagnostics(payload['development_diagnostics'],
                              len(splits['calibration']['included']))
        # Every nested value must still be finite JSON, including audit/diagnostics.
        canonical = _canonical(payload)
    except (KeyError, TypeError, OverflowError) as error:
        raise ValueError('Incomplete or malformed frozen trait model.') from error
    return canonical


@dataclass(frozen=True, slots=True)
class FrozenTraitModel:
    canonical_json: str

    def __post_init__(self):
        if not isinstance(self.canonical_json, str) or len(self.canonical_json.encode('utf-8')) > 4 * 1024 * 1024:
            raise ValueError('Frozen trait model JSON exceeds its bounded size.')
        payload = json.loads(self.canonical_json, object_pairs_hook=_pairs)
        object.__setattr__(self, 'canonical_json', _validate_payload(payload))

    @property
    def model_sha256(self):
        return sha256(self.canonical_json.encode('utf-8')).hexdigest()

    @property
    def payload(self):
        return json.loads(self.canonical_json)

    def to_json(self):
        return _canonical({'model_sha256': self.model_sha256, 'model': self.payload})

    @classmethod
    def from_json(cls, text):
        if not isinstance(text, str) or len(text.encode('utf-8')) > 4 * 1024 * 1024:
            raise ValueError('Frozen trait model JSON exceeds its bounded size.')
        envelope = json.loads(text, object_pairs_hook=_pairs)
        if not isinstance(envelope, dict) or set(envelope) != {'model_sha256', 'model'}:
            raise ValueError('Invalid frozen trait model envelope.')
        result = cls(_canonical(envelope['model']))
        if envelope['model_sha256'] != result.model_sha256:
            raise ValueError('Frozen trait model content does not match its SHA256.')
        return result


class TraitTrainingCancelled(RuntimeError):
    """A requested stop before publishing a new frozen model."""


def check_trait_cancellation(cancellation_requested):
    if cancellation_requested is not None and cancellation_requested():
        raise TraitTrainingCancelled('Trait training cancelled.')


def train_trait_model(training, calibration, vocabulary, *, iterations=200,
                      learning_rate=.03, l2=.01, temperature_bounds=(.05, 20.),
                      calibration_iterations=65, abstention_threshold=.8,
                      cancellation_requested=None, progress_callback=None):
    """Fit only permitted training rows, then calibrate on isolated evidence."""
    check_trait_cancellation(cancellation_requested)
    if not isinstance(training, TraitTrainingEvidence) or not isinstance(calibration, TraitTrainingEvidence) or not isinstance(vocabulary, TraitVocabulary):
        raise ValueError('Training requires explicit feature evidence and vocabulary.')
    if training.schema != TRAIT_FEATURE_SCHEMA or calibration.schema != TRAIT_FEATURE_SCHEMA:
        raise ValueError('Training and calibration feature schemas must match the selected extractor.')
    _integer(iterations, 'Training iterations', 1, 10000)
    _number(learning_rate, 'Learning rate', positive=True, maximum=1)
    _number(l2, 'L2', minimum=0)
    _integer(calibration_iterations, 'Calibration iterations', 2, 1024)
    _number(abstention_threshold, 'Abstention threshold', minimum=0, maximum=1)
    if not isinstance(temperature_bounds, (tuple, list)) or len(temperature_bounds) != 2:
        raise ValueError('Temperature search needs two positive ordered bounds.')
    lower, upper = temperature_bounds
    _number(lower, 'Temperature lower bound', positive=True)
    _number(upper, 'Temperature upper bound', positive=True)
    if not lower < upper: raise ValueError('Temperature search bounds must increase.')
    _disjoint(training.facts, calibration.facts)
    _consistent_truth(training.facts)
    _consistent_truth(calibration.facts)
    if training.values.device != calibration.values.device:
        raise ValueError('Training and calibration must share a CUDA device.')
    x, y = _included(training, vocabulary)
    cx, cy = _included(calibration, vocabulary)
    if len(set(y.tolist())) != len(vocabulary.traits):
        raise ValueError('Every defined class requires permitted training examples.')
    evidence_sha256 = {'training': _evidence_sha256(training, x),
                       'calibration': _evidence_sha256(calibration, cx)}
    mean = x.mean(dim=0)
    scale = (x - mean).square().mean(dim=0).sqrt()
    constant = scale == 0
    scale = torch.where(constant, torch.ones_like(scale), scale)
    x, cx = (x - mean) / scale, (cx - mean) / scale
    if not bool(torch.isfinite(torch.cat((mean, scale, x.flatten(), cx.flatten()))).all().item()):
        raise ValueError('Trait normalization is outside its finite numeric range.')
    weights = torch.zeros((len(vocabulary.traits), 20), dtype=torch.float64, device=x.device, requires_grad=True)
    bias = torch.zeros(len(vocabulary.traits), dtype=torch.float64, device=x.device, requires_grad=True)
    optimizer = torch.optim.Adam((weights, bias), lr=learning_rate)
    for index in range(iterations):
        check_trait_cancellation(cancellation_requested)
        optimizer.zero_grad()
        loss = F.cross_entropy(x @ weights.T + bias, y) + .5 * l2 * weights.square().sum()
        if not bool(torch.isfinite(loss).item()):
            raise ValueError('Trait optimization became nonfinite; no model was produced.')
        loss.backward()
        optimizer.step()
        if progress_callback is not None:
            progress_callback('training', index + 1, iterations)
    with torch.no_grad():
        logits = cx @ weights.T + bias
        if not bool(torch.isfinite(logits).all().item()) or not bool(torch.isfinite(weights).all().item()):
            raise ValueError('Trait optimization is outside its finite numeric range.')
        temperatures = torch.logspace(math.log10(lower), math.log10(upper), calibration_iterations,
                                      dtype=torch.float64, device=x.device)
        temperatures[0], temperatures[-1] = lower, upper
        if lower <= 1 <= upper:
            temperatures = torch.cat((temperatures, temperatures.new_tensor([1.])))
        temperatures = temperatures.unique(sorted=True)
        losses = []
        for index, t in enumerate(temperatures):
            check_trait_cancellation(cancellation_requested)
            losses.append(F.cross_entropy(logits / t, cy))
            if progress_callback is not None:
                progress_callback('calibration', index + 1, len(temperatures))
        losses = torch.stack(losses)
        if not bool(torch.isfinite(losses).all().item()):
            raise ValueError('Trait temperature calibration became nonfinite.')
        # Sorted candidates plus argmin choose the smallest temperature on a tie.
        temperature = float(temperatures[losses.argmin()].item())
        diagnostics = _diagnostics(logits, cy, temperature, abstention_threshold)
        diagnostics['uncalibrated_nll'] = float(F.cross_entropy(logits, cy).item())
        payload = {'schema_version': 1, 'feature_schema': TRAIT_FEATURE_SCHEMA,
            'feature_names': list(TRAIT_FEATURE_NAMES), 'vocabulary': vocabulary.payload(),
            'standardization': {'mean': mean.tolist(), 'scale': scale.tolist(),
                                'constant_columns': torch.nonzero(constant).flatten().tolist()},
            'weights': weights.tolist(), 'bias': bias.tolist(), 'computation_dtype': 'float64',
            'temperature_calibration': {'temperature': temperature, 'bounds': [lower, upper],
                'iterations': calibration_iterations, 'method': 'bounded log-grid NLL minimization; smallest-temperature tie',
                'includes_uncalibrated_candidate': lower <= 1 <= upper},
            'training_configuration': {'iterations': iterations, 'learning_rate': learning_rate,
                'l2': l2, 'l2_convention': 'mean cross entropy + 0.5*l2*sum(weights squared)',
                'optimizer': 'Adam; zero initialization'},
            'split_audit': {'training': _audit(training.facts), 'calibration': _audit(calibration.facts)},
            'training_provenance': training.provenance, 'calibration_provenance': calibration.provenance,
            'evidence_sha256': evidence_sha256,
            'development_diagnostics': diagnostics, 'scientifically_validated': False}
    check_trait_cancellation(cancellation_requested)
    return FrozenTraitModel(_canonical(payload))


def _logits(model, values):
    payload = model.payload
    device = values.device
    mean = torch.tensor(payload['standardization']['mean'], dtype=torch.float64, device=device)
    scale = torch.tensor(payload['standardization']['scale'], dtype=torch.float64, device=device)
    weights = torch.tensor(payload['weights'], dtype=torch.float64, device=device)
    bias = torch.tensor(payload['bias'], dtype=torch.float64, device=device)
    return ((values.to(torch.float64) - mean) / scale) @ weights.T + bias


@torch.no_grad()
def predict_traits(model, batch, *, abstention_threshold=.8):
    """Return compact per-seed probabilities and actual threshold abstention."""
    _number(abstention_threshold, 'Abstention threshold', minimum=0, maximum=1)
    if not isinstance(batch, TraitFeatureBatch): raise ValueError('Prediction needs a trait feature batch.')
    if model is not None and not isinstance(model, FrozenTraitModel):
        raise ValueError('Prediction needs a validated frozen trait model.')
    reason = 'no_frozen_model' if model is None else (
        'incompatible_feature_schema' if batch.schema != TRAIT_FEATURE_SCHEMA else None)
    model_sha = model.model_sha256 if model is not None else None
    rows = [dict(seed_id=identifier, model_sha256=model_sha, probabilities=None, confidence=None,
                 predictive_entropy=None, predicted_trait_id=None, abstained=True,
                 abstention_reason=why, eligibility_reason=why)
            for identifier, why in batch.rejected]
    if reason:
        rows.extend(dict(seed_id=identifier, model_sha256=model_sha, probabilities=None,
            confidence=None, predictive_entropy=None, predicted_trait_id=None, abstained=True,
            abstention_reason=reason, eligibility_reason=None) for identifier in batch.identifiers)
    elif batch.identifiers:
        payload = model.payload
        logits = _logits(model, batch.values)
        calibrated_logits = logits / payload['temperature_calibration']['temperature']
        valid = torch.isfinite(batch.values).all(dim=1) & torch.isfinite(calibrated_logits).all(dim=1)
        safe_logits = torch.where(valid[:, None], calibrated_logits, torch.zeros_like(logits))
        probabilities = safe_logits.softmax(dim=1)
        confidence, classes = probabilities.max(dim=1)
        entropy = -(probabilities * probabilities.clamp_min(torch.finfo(torch.float64).tiny).log()).sum(dim=1)
        labels = [x['identifier'] for x in payload['vocabulary']['traits']]
        for identifier, ok, probs, conf, cls, ent in zip(batch.identifiers, valid.tolist(),
                probabilities.tolist(), confidence.tolist(), classes.tolist(), entropy.tolist(), strict=True):
            why = 'invalid_features' if not ok else ('confidence_below_threshold' if conf < abstention_threshold else None)
            rows.append(dict(seed_id=identifier, model_sha256=model_sha,
                probabilities=dict(zip(labels, probs, strict=True)) if ok else None,
                confidence=conf if ok else None, predictive_entropy=ent if ok else None,
                predicted_trait_id=labels[cls] if why is None else None,
                abstained=why is not None, abstention_reason=why, eligibility_reason=None))
    return {'schema_version': 1, 'feature_schema': batch.schema, 'model_sha256': model_sha,
            'feature_provenance': json.loads(batch.provenance_json),
            'calibration_provenance': model.payload['calibration_provenance'] if model else None,
            'abstention_threshold': abstention_threshold, 'predictions': sorted(rows, key=lambda x: x['seed_id']),
            'scientifically_validated': False,
            'probability_semantics': 'calibrated predictive probabilities; not confidence intervals or proof of correctness'}


@torch.no_grad()
def evaluate_trait_model(model, evidence, *, abstention_threshold=.8):
    """Evaluate an already frozen model without allowing split leakage or refit."""
    if not isinstance(model, FrozenTraitModel) or not isinstance(evidence, TraitTrainingEvidence):
        raise ValueError('Evaluation requires a frozen model and explicit evidence.')
    _number(abstention_threshold, 'Abstention threshold', minimum=0, maximum=1)
    payload = model.payload
    if evidence.schema != payload['feature_schema']:
        raise ValueError('Evaluation features do not match the frozen model feature schema.')
    _consistent_truth(evidence.facts)
    for name in ('training', 'calibration'):
        split = payload['split_audit'][name]
        old = tuple(TraitExample(**item) for item in split['included'])
        old += tuple(TraitExample(**{key: value for key, value in item.items() if key != 'exclusion_reason'})
                     for item in split['excluded'])
        _disjoint(old, evidence.facts)
    v = payload['vocabulary']
    vocabulary = TraitVocabulary(v['revision'], v['definition_source'], v['definition_source_sha256'],
                                 tuple(TraitDefinition(**item) for item in v['traits']))
    values, targets = _included(evidence, vocabulary)
    logits = _logits(model, values)
    if not bool(torch.isfinite(logits).all().item()):
        raise ValueError('Evaluation logits are outside their finite numeric range.')
    result = _diagnostics(logits, targets, payload['temperature_calibration']['temperature'], abstention_threshold)
    result.update(model_sha256=model.model_sha256, evidence_provenance=evidence.provenance,
                  split_audit=_audit(evidence.facts), evaluation_state='supplied_disjoint_evidence')
    return result
