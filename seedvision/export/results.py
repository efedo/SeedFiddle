"""Authoritative, reviewable image results shared by desktop and batch consumers."""
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import csv
import json
import cv2
import numpy as np

from seedvision.learning.data import file_sha256


def select_result(analysis, enabled=None):
    """Prefer explicitly enabled learned branches, then procedural instances; never legacy proposals."""
    for name in ('unet_instances', 'stardist_instances', 'procedural_instances'):
        value = getattr(analysis, name, None)
        if value is not None and (enabled is None or name in enabled):
            if name == 'procedural_instances' and getattr(analysis.layers, 'foreground_colour_profile', None) is None:
                return name, None, 'Foreground evidence is missing. Apply Foreground references or select a supported species library.'
            return name, value, ''
    return None, None, 'No enabled instance analysis has completed. Run an instance node.'


def result_report(analysis, *, enabled=None, settings=None, provenance=None, review=None):
    method, instances, unavailable = select_result(analysis, enabled)
    source = analysis.image_path
    source_digest = file_sha256(source) if source is not None and Path(source).is_file() else None
    if getattr(analysis,'source_sha256',None) is not None and analysis.source_sha256 != source_digest:
        raise ValueError('The source changed after analysis. Reanalyze before reviewing or exporting.')
    transform = np.asarray(analysis.calibration.affine_matrix).tolist()
    identity = {'source_sha256': source_digest, 'method': method, 'settings': settings,
        'source_to_corrected': transform, 'provenance': provenance or {}}
    from seedvision import __version__
    package = Path(__file__).resolve().parents[1]
    identity['implementation'] = {'version':__version__, 'files':{name:file_sha256(package/name) for name in (
        'segmentation/baseline.py','segmentation/procedural.py','cuda/layers.py','calibration/image.py',
        'learning/pipeline.py','learning/features.py','export/results.py')}}
    if source_digest is None:
        identity['corrected_image_sha256'] = sha256(np.ascontiguousarray(analysis.calibration.corrected_bgr).tobytes()).hexdigest()
    rows = []
    if instances is not None:
        checkpoint_id = getattr(instances,'checkpoint_id','')
        identity['checkpoint_id'] = checkpoint_id
        if '#sha256=' in checkpoint_id:
            checkpoint_path, expected = checkpoint_id.rsplit('#sha256=',1)
            if not Path(checkpoint_path).is_file() or file_sha256(checkpoint_path) != expected:
                raise ValueError('Checkpoint changed after inference. Recompute before reviewing or exporting.')
        labels = np.asarray(instances.labels, np.int32)
        identity['label_sha256'] = sha256(labels.tobytes()).hexdigest()
        shape = getattr(instances, 'source_shape', labels.shape)
        sy, sx = shape[0]/labels.shape[0], shape[1]/labels.shape[1]
        ox, oy = analysis.crop_offset
        scores = np.asarray(instances.instance_confidences)
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
    revision = sha256(json.dumps(identity,sort_keys=True,default=str).encode()).hexdigest()
    if review and review.get('result_revision') == revision:
        for row in rows:
            state = review.get('decisions',{}).get(str(row['seed_id']), 'unreviewed')
            if state in {'accepted','excluded','unreviewed'}:
                row['review'] = state
    return {'schema_version': 1, 'result_revision': revision, 'identity': identity,
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
        'candidate_diagnostics': {} if instances is None else {
            'candidate_scores': np.asarray(getattr(instances,'candidate_scores',())).tolist(),
            'selected': np.asarray(getattr(instances,'candidate_selected',()),dtype=bool).tolist(),
            'interpretation':'Unselected hypotheses can overlap accepted candidates; they are not necessarily distinct rejected seeds. Review the alternative-candidate overlay and stratify errors by size, occlusion and phenotype.'},
        'rejected_manual_centres': [] if instances is None else [
            {'xy_px': list(map(float,xy)), 'reason': reason} for xy,reason in zip(
                getattr(instances,'rejected_manual_centres_xy',()),
                getattr(instances,'rejected_manual_centre_reasons',()),strict=True)]}


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
