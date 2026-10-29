"""Authoritative, reviewable image results shared by desktop and batch consumers."""
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import csv
import json
import cv2
import numpy as np

from seedvision.learning.data import file_sha256


RESULT_REPORT_VERSION = 3
RESULT_REVISION_CONTRACT = 'seedfiddle.result-revision-v7'


def _prediction_array(values, name, *, ndim, dtype=None):
    """Snapshot compact prediction facts once for hashing and reporting."""
    array = np.asarray(values)
    if array.dtype.kind not in 'biuf' or array.ndim != ndim:
        raise ValueError(f'{name} must be a {ndim}-dimensional numeric array.')
    snapshot = np.array(array, order='C', copy=True)
    if not np.isfinite(snapshot).all():
        raise ValueError(f'{name} must contain finite prediction values.')
    if dtype is not None:
        snapshot = snapshot.astype(dtype, copy=False)
    snapshot.flags.writeable = False
    identity = {'shape': list(snapshot.shape), 'dtype': snapshot.dtype.str,
                'sha256': sha256(snapshot.tobytes()).hexdigest()}
    return snapshot, identity


def select_result(analysis, enabled=None):
    """Prefer explicitly enabled learned branches, then procedural instances; never legacy proposals."""
    from seedvision.learning.instance_selection import select_instances
    return select_instances(unet=getattr(analysis, 'unet_instances', None),
        stardist=getattr(analysis, 'stardist_instances', None),
        cellpose=getattr(analysis, 'cellpose_sam_instances', None),
        procedural=getattr(analysis, 'procedural_instances', None), enabled=enabled,
        foreground_available=getattr(getattr(analysis, 'layers', None),
                                     'foreground_colour_profile', None) is not None)


def result_report(analysis, *, enabled=None, settings=None, provenance=None, review=None,
                  execution_provenance=None):
    method, instances, unavailable = select_result(analysis, enabled)
    source = analysis.image_path
    source_digest = file_sha256(source) if source is not None and Path(source).is_file() else None
    if getattr(analysis,'source_sha256',None) is not None and analysis.source_sha256 != source_digest:
        raise ValueError('The source changed after analysis. Reanalyze before reviewing or exporting.')
    transform = np.asarray(analysis.calibration.affine_matrix).tolist()
    from seedvision.export.calibration_facts import calibration_facts
    recorded_calibration = calibration_facts(analysis.calibration)
    from seedvision.export.artifact_identity import (verified_artifact_facts, semantic_settings,
        semantic_artifact, semantic_trait_report, semantic_checkpoint, execution_settings)
    artifact_capture = getattr(analysis, 'artifact_selections_json', None)
    artifacts = {} if artifact_capture is None else verified_artifact_facts(artifact_capture)
    identity_settings = settings if artifact_capture is None else semantic_settings(settings, artifacts)
    from seedvision.export.backend_outcomes import verified_native_outcomes, semantic_native_outcomes
    backend_outcomes = verified_native_outcomes(getattr(analysis, 'backend_outcomes_json', None), analysis)
    identity = {'revision_contract': RESULT_REVISION_CONTRACT,
        'source_sha256': source_digest, 'method': method, 'settings': identity_settings,
        'source_to_corrected': transform, 'crop_offset': list(analysis.crop_offset),
        'calibration_facts': recorded_calibration,
        'provenance': provenance or {},
        'result_state': 'unavailable' if unavailable else 'available_for_review',
        'unavailable_reason': unavailable}
    identity['backend_outcomes'] = semantic_native_outcomes(backend_outcomes)
    if artifact_capture is not None:
        identity['native_artifact_selections'] = {node: semantic_artifact(record, node)
                                                 for node, record in artifacts.items()}
    captured = getattr(analysis, 'analysis_inputs_json', None)
    if captured is not None:
        from seedvision.export.analysis_inputs import analysis_input_facts
        identity['analysis_inputs'] = analysis_input_facts(captured)
    from seedvision import __version__
    package = Path(__file__).resolve().parents[1]
    identity['implementation'] = {'version':__version__, 'files':{name:file_sha256(package/name) for name in (
        'segmentation/baseline.py','segmentation/procedural.py','cuda/layers.py','calibration/image.py',
        'learning/pipeline.py','learning/features.py','export/results.py',
        'calibration/neutral.py','export/calibration_facts.py',
        'measurement/shape_model.py','measurement/geometry.py','measurement/descriptions.py',
        'measurement/lineage.py','export/reviewed_measurements.py','export/shape_models.py',
        'learning/instance_selection.py','learning/trait_pipeline.py',
        'learning/trait_features.py','learning/trait_classifier.py',
        'export/analysis_inputs.py','export/review.py','export/artifact_identity.py',
        'export/backend_outcomes.py')}}
    corrected = np.ascontiguousarray(analysis.calibration.corrected_bgr)
    identity['corrected_image_sha256'] = sha256(corrected.tobytes()).hexdigest()
    identity['corrected_image_shape'] = list(corrected.shape)
    identity['corrected_image_dtype'] = corrected.dtype.str
    from seedvision.export.reviewed_measurements import reviewed_measurements_report
    reviewed_measurements = reviewed_measurements_report(
        getattr(analysis, 'seed_measurement_summary', None), identity)
    identity['reviewed_measurements'] = reviewed_measurements
    from seedvision.export.shape_models import shape_model_report
    shape_model = shape_model_report(
        getattr(analysis, 'seed_dimensions_shape_model', None), identity, reviewed_measurements)
    identity['shape_model'] = shape_model
    from seedvision.learning.trait_pipeline import trait_inference_report
    trait_classifier = trait_inference_report(getattr(analysis, 'trait_classifier', None),
        method, instances, corrected, analysis.crop_offset, analysis.calibration.affine_matrix)
    identity['trait_classifier'] = semantic_trait_report(trait_classifier, artifacts)
    trait_rows = {} if trait_classifier is None else {
        row['seed_id']: row for row in trait_classifier['predictions']}
    rows = []
    candidate_diagnostics = {}
    rejected_manual_centres = []
    if instances is not None:
        checkpoint_id = getattr(instances,'checkpoint_id','')
        identity['checkpoint_id'] = semantic_checkpoint(checkpoint_id, method, artifacts)
        if '#sha256=' in checkpoint_id:
            checkpoint_path, expected = checkpoint_id.rsplit('#sha256=',1)
            if not Path(checkpoint_path).is_file() or file_sha256(checkpoint_path) != expected:
                raise ValueError('Checkpoint changed after inference. Recompute before reviewing or exporting.')
        labels = np.asarray(instances.labels, np.int32)
        identity['label_sha256'] = sha256(labels.tobytes()).hexdigest()
        shape = getattr(instances, 'source_shape', labels.shape)
        identity['label_shape'] = list(labels.shape)
        identity['instance_source_shape'] = list(shape)
        sy, sx = shape[0]/labels.shape[0], shape[1]/labels.shape[1]
        ox, oy = analysis.crop_offset
        scores, score_identity = _prediction_array(
            instances.instance_confidences, 'instance_confidences', ndim=1)
        candidate_scores, candidate_score_identity = _prediction_array(
            getattr(instances, 'candidate_scores', ()), 'candidate_scores', ndim=1)
        candidate_selected, candidate_selected_identity = _prediction_array(
            getattr(instances, 'candidate_selected', ()), 'candidate_selected', ndim=1, dtype=bool)
        if len(candidate_scores) != len(candidate_selected):
            raise ValueError('Candidate scores and selections must have matching lengths.')
        rejected_xy = getattr(instances, 'rejected_manual_centres_xy', ())
        if np.asarray(rejected_xy).size == 0:
            rejected_xy = np.asarray(rejected_xy).reshape(0, 2)
        rejected_xy, rejected_xy_identity = _prediction_array(
            rejected_xy, 'rejected_manual_centres_xy', ndim=2)
        rejected_reasons = tuple(getattr(instances, 'rejected_manual_centre_reasons', ()))
        if rejected_xy.shape[1] != 2 or len(rejected_xy) != len(rejected_reasons):
            raise ValueError('Rejected manual centres require one XY pair and reason per centre.')
        if any(not isinstance(reason, str) for reason in rejected_reasons):
            raise ValueError('Rejected manual centre reasons must be strings.')
        identity['prediction_facts'] = {
            'instance_confidences': score_identity,
            'candidate_scores': candidate_score_identity,
            'candidate_selected': candidate_selected_identity,
            'rejected_manual_centres_xy': rejected_xy_identity,
            'rejected_manual_centre_reasons': list(rejected_reasons)}
        candidate_diagnostics = {'candidate_scores': candidate_scores.tolist(),
            'selected': candidate_selected.tolist(),
            'interpretation':'Unselected hypotheses can overlap accepted candidates; they are not necessarily distinct rejected seeds. Review the alternative-candidate overlay and stratify errors by size, occlusion and phenotype.'}
        rejected_manual_centres = [
            {'xy_px': list(map(float, xy)), 'reason': reason}
            for xy, reason in zip(rejected_xy, rejected_reasons, strict=True)]
        if labels.size and int(labels.max()) > len(scores):
            raise ValueError('Instance confidences must cover every positive label ID.')
        for identifier in np.unique(labels[labels > 0]):
            yy, xx = np.nonzero(labels == identifier)
            points = np.column_stack(((xx+.5)*sx, (yy+.5)*sy)).astype(np.float32)
            sides = sorted(cv2.minAreaRect(points)[1], reverse=True) if len(points) > 1 else [0., 0.]
            rows.append({'seed_id': int(identifier), 'review': 'unreviewed',
                'centre_x_px': float(points[:,0].mean()+ox), 'centre_y_px': float(points[:,1].mean()+oy),
                'visible_area_px2': float(len(points)*sx*sy),
                'visible_major_extent_px': float(sides[0]+min(sx,sy)),
                'visible_minor_extent_px': float(sides[1]+min(sx,sy)),
                'uncalibrated_instance_score': float(scores[int(identifier)-1]),
                'physical_length_mm': None, 'physical_area_mm2': None,
                'trait_assignment': 'abstained: no validated object-level classifier',
                'inferred_full_shape': None})
            if int(identifier) in trait_rows:
                prediction = trait_rows[int(identifier)]
                rows[-1]['trait_assignment'] = (prediction['predicted_trait_id'] if not prediction['abstained']
                    else 'abstained: ' + prediction['abstention_reason'])
                rows[-1]['trait_prediction'] = prediction
    revision = sha256(json.dumps(identity,sort_keys=True,default=str,allow_nan=False).encode()).hexdigest()
    if review and review.get('result_revision') == revision:
        for row in rows:
            state = review.get('decisions',{}).get(str(row['seed_id']), 'unreviewed')
            if state in {'accepted','excluded','unreviewed'}:
                row['review'] = state
    report = {'schema_version': RESULT_REPORT_VERSION, 'result_revision': revision, 'identity': identity,
        'source': str(source) if source else None, 'method': method,
        'state': 'unavailable' if unavailable else 'available_for_review', 'unavailable_reason': unavailable,
        'proposed_count': None if unavailable else len(rows),
        'accepted_count': sum(row['review'] == 'accepted' for row in rows),
        'excluded_count': sum(row['review'] == 'excluded' for row in rows),
        'unreviewed_count': sum(row['review'] == 'unreviewed' for row in rows),
        'units': 'corrected-image pixels; visible 2-D projected regions',
        'metric_measurements': 'withheld: acquisition geometry has not been independently validated',
        'score_semantics': 'uncalibrated heuristic/model scores, not probabilities of correctness',
        'scientifically_validated': False, 'warnings': list(analysis.warnings), 'instances': rows,
        'candidate_diagnostics': candidate_diagnostics,
        'rejected_manual_centres': rejected_manual_centres}
    report['backend_outcomes'] = dict(backend_outcomes, run_revision=revision,
                                     selected_backend=method)
    if execution_provenance is not None:
        report['execution_provenance'] = json.loads(json.dumps(execution_provenance, allow_nan=False))
    if artifact_capture is not None:
        report['native_artifact_execution'] = {'contract': 'seedfiddle.native-artifact-execution-v1',
            'selections': artifacts, 'settings': execution_settings(settings, artifacts)}
    if reviewed_measurements is not None:
        report['reviewed_measurements'] = reviewed_measurements
    if shape_model is not None:
        report['shape_model'] = shape_model
    if trait_classifier is not None:
        report['trait_classifier'] = trait_classifier
    return report


def annotated_image(analysis, report):
    image = analysis.calibration.corrected_bgr.copy()
    for row in report['instances']:
        colour = {'accepted': (60,210,60), 'excluded': (60,60,230), 'unreviewed': (0,210,240)}[row['review']]
        xy = (round(row['centre_x_px']),round(row['centre_y_px']))
        cv2.circle(image,xy,7,colour,2)
        cv2.putText(image,str(row['seed_id']),xy,cv2.FONT_HERSHEY_SIMPLEX,.6,colour,2)
    return image


def export_report(directory, analysis, report):
    """Create a unique immutable revision directory; never overwrite prior reports."""
    from uuid import uuid4
    destination = Path(directory) / (report['result_revision'][:16]+'-'+uuid4().hex[:8])
    destination.mkdir(parents=True,exist_ok=False)
    fields = list(report['instances'][0]) if report['instances'] else ['seed_id','review']
    with (destination/'instances.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerows(report['instances'])
    if not cv2.imwrite(str(destination/'annotated.png'),annotated_image(analysis,report)):
        raise OSError('Could not write annotated image.')
    report = dict(report,components={name:file_sha256(destination/name) for name in ('instances.csv','annotated.png')})
    (destination/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return destination
