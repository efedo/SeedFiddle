"""Cached native classifier inference and immutable result-frame association."""
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from seedvision.learning.trait_classifier import FrozenTraitModel, predict_traits
from seedvision.learning.trait_features import TRAIT_FEATURE_SCHEMA, TraitFeatureBatch, extract_trait_features

_MAX_MODEL_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class TraitClassifierSettings:
    trait_model_path: str = 'None'
    trait_abstention_threshold: float = .8

    def __post_init__(self):
        if not isinstance(self.trait_model_path, str) or not self.trait_model_path.strip():
            raise ValueError('Trait model path must be explicit, or None.')
        value = self.trait_abstention_threshold
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Trait abstention threshold must be finite in [0,1].')


@dataclass(frozen=True, slots=True)
class TraitInferenceResult:
    canonical_json: str

    @property
    def report(self):
        return json.loads(self.canonical_json)


def trait_input_binding(method, instances, corrected_bgr, crop_offset, affine):
    corrected = np.ascontiguousarray(corrected_bgr)
    labels = None if instances is None else np.asarray(instances.labels, dtype=np.int32)
    return {'selected_method': method,
        'label_sha256': None if labels is None else sha256(labels.tobytes()).hexdigest(),
        'label_shape': None if labels is None else list(labels.shape),
        'instance_source_shape': None if labels is None else list(getattr(instances, 'source_shape', labels.shape)),
        'corrected_image_sha256': sha256(corrected.tobytes()).hexdigest(),
        'corrected_image_shape': list(corrected.shape), 'corrected_image_dtype': corrected.dtype.str,
        'crop_offset': list(crop_offset), 'source_to_corrected': np.asarray(affine).tolist(),
        'feature_schema': TRAIT_FEATURE_SCHEMA,
        'label_mapping': 'recorded instance source_shape to corrected crop; nearest-neighbour labels',
        'eligibility_policy': 'visible instance pixels inside valid corrected crop; no physical-completeness claim'}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _abstention(identifiers, reason, *, model=None):
    return {'schema_version': 1, 'feature_schema': TRAIT_FEATURE_SCHEMA,
        'model_sha256': model.model_sha256 if model else None,
        'predictions': [dict(seed_id=value, model_sha256=model.model_sha256 if model else None,
            probabilities=None, confidence=None, predictive_entropy=None,
            predicted_trait_id=None, abstained=True, abstention_reason=reason,
            eligibility_reason=reason if reason not in {'no_frozen_model', 'invalid_model', 'model_file_missing'} else None)
            for value in identifiers], 'scientifically_validated': False,
        'probability_semantics': 'calibrated predictive probabilities; not confidence intervals or proof of correctness'}


def _read_model_bytes(path):
    with path.open('rb') as stream:
        contents = stream.read(_MAX_MODEL_BYTES + 1)
    if len(contents) > _MAX_MODEL_BYTES:
        raise ValueError('Frozen model exceeds its supported size.')
    return contents


def _model(settings, root):
    if settings.trait_model_path.strip().casefold() == 'none':
        return None, None, None, 'no_frozen_model'
    path = Path(settings.trait_model_path).expanduser()
    path = path.resolve() if path.is_absolute() else (Path(root) / path).resolve()
    if not path.is_file():
        return None, str(path), None, 'model_file_missing'
    try:
        if path.stat().st_size > _MAX_MODEL_BYTES:
            raise ValueError('Frozen model exceeds its supported size.')
        contents = _read_model_bytes(path)
        return FrozenTraitModel.from_json(contents.decode('utf-8')), str(path), sha256(contents).hexdigest(), None
    except (ValueError, UnicodeError, OSError):
        return None, str(path), sha256(contents).hexdigest() if 'contents' in locals() else None, 'invalid_model'


def _tensor(value):
    return value.gpu_tensor() if hasattr(value, 'gpu_tensor') else value


def _tensor_signature(value):
    tensor = _tensor(value)
    if not isinstance(tensor, torch.Tensor):
        return None
    try:
        version = tensor._version
    except RuntimeError:
        # Inference-mode tensors have no mutation counter. Recompute rather
        # than treating their identity as proof that their pixels are unchanged.
        version = object()
    header = (str(tensor.device), tensor.dtype, tensor.layout, tuple(tensor.shape))
    if tensor.layout == torch.strided:
        storage = tensor.untyped_storage()
        header += (tuple(tensor.stride()), tensor.storage_offset(), storage.data_ptr(), storage.nbytes())
    return (id(tensor), version, header)


@torch.no_grad()
def run_trait_classifier(method, instances, *, corrected_bgr, crop_offset, affine,
                         gpu_source, gpu_valid, settings, root, cache):
    """Reuse feature extraction and calibrated scores on threshold-only edits.

    Existing compact/display instance labels upload once; no image or mask is
    downloaded for classification. A recorded source_shape must match the crop.
    """
    binding = trait_input_binding(method, instances, corrected_bgr, crop_offset, affine)
    identifiers = () if instances is None else tuple(int(x) for x in np.unique(instances.labels) if x > 0)
    model, path, artifact_sha, unavailable = _model(settings, root)
    model_key = (path, artifact_sha, model.model_sha256 if model else None, unavailable)
    feature_key = (_canonical(binding), _tensor_signature(gpu_source), _tensor_signature(gpu_valid))
    score_key = (feature_key, model_key)
    decision_key = (score_key, settings.trait_abstention_threshold)
    if cache.get('trait_classifier.decision_key') == decision_key:
        return cache['trait_classifier.result'], False
    if cache.get('trait_classifier.score_key') != score_key:
        if unavailable:
            scored = _abstention(identifiers, unavailable)
        elif instances is None:
            scored = _abstention((), 'no_instance_result', model=model)
        else:
            if cache.get('trait_classifier.feature_key') != feature_key:
                source, valid = _tensor(gpu_source), _tensor(gpu_valid)
                batch = None
                reason = None
                if (not isinstance(source, torch.Tensor) or not source.is_cuda or source.ndim != 4
                        or source.shape[0] != 1 or source.shape[1] != 3
                        or not isinstance(valid, torch.Tensor) or not valid.is_cuda
                        or valid.device != source.device or tuple(valid.shape[:-2]) not in ((), (1, 1))):
                    reason = 'gpu_inputs_unavailable'
                elif binding['instance_source_shape'] != list(source.shape[-2:]):
                    reason = 'incompatible_frame'
                elif (len(crop_offset) != 2 or min(crop_offset) < 0
                        or crop_offset[0] + source.shape[-1] > corrected_bgr.shape[1]
                        or crop_offset[1] + source.shape[-2] > corrected_bgr.shape[0]):
                    reason = 'incompatible_frame'
                elif (np.asarray(affine).shape != (3, 3) or not np.isfinite(affine).all()
                        or not np.isfinite(np.linalg.det(affine))
                        or abs(float(np.linalg.det(affine))) <= np.finfo(np.float64).eps):
                    reason = 'invalid_transform'
                else:
                    labels = torch.as_tensor(np.array(instances.labels, dtype=np.int32, copy=True), device=source.device)
                    labels = F.interpolate(labels[None, None].double(), size=source.shape[-2:], mode='nearest')[0, 0].long()
                    valid = valid.reshape(valid.shape[-2:]).bool()
                    if valid.shape != labels.shape:
                        reason = 'incompatible_support_frame'
                    else:
                        rejected, eligible = [], []
                        for identifier in identifiers:
                            mask = labels == identifier
                            if not bool(mask.any().item()):
                                rejected.append((identifier, 'missing_after_mapping'))
                            elif bool((mask & ~valid).any().item()):
                                rejected.append((identifier, 'outside_valid_crop'))
                            elif bool((mask[0].any() | mask[-1].any() | mask[:, 0].any() | mask[:, -1].any()).item()):
                                rejected.append((identifier, 'crop_boundary_instance'))
                            else:
                                eligible.append(identifier)
                        frame = sha256(_canonical(binding).encode('utf-8')).hexdigest()
                        batch = extract_trait_features(source[0].permute(1, 2, 0).double() / 255,
                            labels, tuple(eligible), tuple(eligible), encoding='bgr_unit',
                            image_frame=frame, label_frame=frame)
                        batch = TraitFeatureBatch(batch.values, batch.identifiers,
                            batch.rejected + tuple(rejected), batch.provenance_json)
                cache['trait_classifier.features'] = batch
                cache['trait_classifier.feature_unavailability'] = reason
                cache['trait_classifier.feature_key'] = feature_key
            batch = cache['trait_classifier.features']
            if batch is None:
                scored = _abstention(identifiers, cache['trait_classifier.feature_unavailability'], model=model)
            else:
                scored = predict_traits(model, batch, abstention_threshold=0)
        cache['trait_classifier.scores_json'] = _canonical(scored)
        cache['trait_classifier.score_key'] = score_key
    result = json.loads(cache['trait_classifier.scores_json'])
    for row in result['predictions']:
        if row['probabilities'] is not None and row['confidence'] < settings.trait_abstention_threshold:
            row.update(predicted_trait_id=None, abstained=True, abstention_reason='confidence_below_threshold')
    result.update(abstention_threshold=settings.trait_abstention_threshold,
        input_binding=binding, model_path=path, model_artifact_sha256=artifact_sha,
        frozen_model=model.payload if model else None,
        model_unavailability_reason=unavailable)
    frozen = TraitInferenceResult(_canonical(result))
    cache['trait_classifier.result'] = frozen
    cache['trait_classifier.decision_key'] = decision_key
    return frozen, True


def trait_inference_report(result, method, instances, corrected_bgr, crop_offset, affine):
    """Export stored predictions only when they still describe exported labels."""
    if result is None:
        return None
    report = result.report
    binding = trait_input_binding(method, instances, corrected_bgr, crop_offset, affine)
    if report['input_binding'] != binding:
        raise ValueError('Trait classifier inputs changed; reanalyze before export.')
    if report['model_artifact_sha256'] is not None:
        path = Path(report['model_path'])
        try:
            digest = sha256(_read_model_bytes(path)).hexdigest()
        except (OSError, ValueError) as error:
            raise ValueError('Trait classifier model changed; reanalyze before export.') from error
        if digest != report['model_artifact_sha256']:
            raise ValueError('Trait classifier model changed; reanalyze before export.')
    elif report['model_path'] is not None and report['model_unavailability_reason'] == 'model_file_missing':
        if Path(report['model_path']).is_file():
            raise ValueError('Trait classifier model became available; reanalyze before export.')
    elif report['model_path'] is not None and report['model_unavailability_reason'] == 'invalid_model':
        current, _, digest, reason = _model(TraitClassifierSettings(report['model_path']), Path('.'))
        if current is not None or digest is not None or reason != 'invalid_model':
            raise ValueError('Trait classifier model changed; reanalyze before export.')
    return report
