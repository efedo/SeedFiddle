"""Factual evaluation eligibility and immutable input identities."""
from dataclasses import asdict
from hashlib import sha256
import json
import numpy as np

from seedvision.learning.data import LearningManifest, audit_manifest, file_sha256, resolve_sample_path


def dataset_identity(path, *, splits=None):
    manifest = LearningManifest.load(path)
    records = []
    for sample in manifest.samples:
        if splits is not None and sample.split not in splits:
            continue
        components = {name: file_sha256(resolve_sample_path(path, value)) for name in (
            'features','instances','image','pattern_boundary','pattern_valid','physical_boundary','physical_valid')
            if (value := getattr(sample, name))}
        with np.load(resolve_sample_path(path,sample.features),allow_pickle=False) as archive:
            features = np.ascontiguousarray(archive['features'])
            feature_content = sha256(str((features.shape,str(features.dtype))).encode()+features.tobytes()).hexdigest()
        records.append({'id':sample.identifier,'group':sample.group,'split':sample.split,
            'source_sha256':sample.provenance.get('source_sha256'), 'components':components, 'feature_content_sha256': feature_content})
    encoded = json.dumps(records,sort_keys=True).encode()
    return {'sha256':sha256(encoded).hexdigest(),'samples':records}


def audit_evaluation(path, split, optimize_decoder, protocol='development'):
    if protocol not in {'development','frozen','reference_assisted'}:
        raise ValueError('Unknown evaluation protocol.')
    if optimize_decoder and split != 'validation':
        raise ValueError('Decoder optimization is permitted only on the validation split.')
    audit = audit_manifest(path)
    if not audit['valid']:
        raise ValueError('Evaluation dataset audit failed: ' + '; '.join(audit['errors']))
    if protocol != 'development':
        for sample in LearningManifest.load(path).samples:
            if sample.split != split:
                continue
            provenance = sample.provenance
            if not sample.reviewed or not sample.annotation_author or not sample.annotation_revision:
                raise ValueError('Independent evaluation requires reviewed, attributed annotation revisions.')
            if provenance.get('analysis_mode') != protocol:
                raise ValueError('Independent evaluation rejects image-local adaptation and unknown preprocessing provenance. Re-export under the declared deployment protocol.')
            required = ('source_sha256','preprocessing_recipe_sha256','target_access_audit')
            if any(not provenance.get(key) for key in required):
                raise ValueError('Independent evaluation requires source, frozen preprocessing and target-access provenance.')
            if provenance['target_access_audit'] != 'targets_excluded_from_all_preprocessing':
                raise ValueError('Held-out targets may not influence preprocessing, scale, libraries or feature assembly.')
            if protocol == 'reference_assisted' and not provenance.get('reference_sources_sha256'):
                raise ValueError('Reference-assisted evaluation requires separately identified deployment references.')
        audit['protocol_warning'] = 'Recorded provenance is auditable evidence, not automatic proof of correct study conduct.'
    return audit


def checkpoint_membership(payload, evaluation_identity):
    training = payload.get('training_metadata',{}).get('development_identity')
    if training is None:
        return 'Unknown: checkpoint does not record immutable development membership.'
    for held in evaluation_identity['samples']:
        for used in training['samples']:
            if held['group'] == used['group'] or (
                held.get('source_sha256') and held['source_sha256'] == used.get('source_sha256')) or (
                held['components']['features'] == used['components']['features']) or (
                held.get('feature_content_sha256') and held['feature_content_sha256'] == used.get('feature_content_sha256')):
                raise ValueError('Evaluation overlaps checkpoint development sources/groups.')
    return 'No recorded source/group overlap with checkpoint development data.'


def decoder_identity(settings):
    return sha256(json.dumps(asdict(settings),sort_keys=True).encode()).hexdigest()


def audit_decoder_source(path, settings, identity, *, independent=False):
    from pathlib import Path
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    if payload.get('decoder_settings',payload) != asdict(settings):
        raise ValueError('Frozen decoder source does not match the evaluated settings.')
    membership = payload.get('dataset_identity')
    if independent:
        if membership is not None:
            if payload.get('split') != 'validation':
                raise ValueError('A tuned decoder must be selected on validation data.')
            checkpoint_membership({'training_metadata':{'development_identity':membership}},identity)
        elif payload.get('selection_protocol') != 'predeclared':
            raise ValueError('Independent custom decoder settings need a validation selection report or explicit predeclared provenance.')
    return {'sha256':file_sha256(path),'selection_dataset_identity':membership,
        'selection_protocol':payload.get('selection_protocol',payload.get('split','unknown'))}


def aggregate_records(records):
    def summary(items):
        tp,fp,fn = (sum(item[key] for item in items) for key in ('true_positives','false_positives','false_negatives'))
        return {'images':len(items),'true_positives':tp,'false_positives':fp,'false_negatives':fn,
            'micro_precision':tp/(tp+fp) if tp+fp else None,
            'micro_recall':tp/(tp+fn) if tp+fn else None,
            'micro_f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            'mean_absolute_count_error':float(np.mean([item['absolute_count_error'] for item in items])),
            'empty_truth_images':sum(item['true_instances']==0 for item in items),
            'empty_false_positive_instances':sum(item['predicted_instances'] for item in items if item['true_instances']==0)}
    return {'micro':summary(records), 'by_group':{group:summary([item for item in records if item['group']==group])
        for group in sorted({item['group'] for item in records})},
        'metric_policy':'Both-empty image F1/PQ=1; no matched objects means IoU unavailable; relative count error unavailable for empty truth; report absolute count errors and empty false positives separately. Unavailable dense metrics are null.'}
