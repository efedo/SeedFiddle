"""Study preparation and descriptive diagnostics; never certify a method automatically."""
from collections import defaultdict
import numpy as np


def score_reliability(scores, outcomes, *, bins=10):
    """Calibration table for independent, binary correctness outcomes."""
    scores = np.asarray(scores,float)
    outcomes = np.asarray(outcomes,float)
    if scores.ndim != 1 or scores.shape != outcomes.shape or not len(scores):
        raise ValueError('Provide equally sized, non-empty score and correctness vectors.')
    if not np.isfinite(scores).all() or np.any((scores<0)|(scores>1)) or not np.isin(outcomes,(0,1)).all():
        raise ValueError('Scores must lie in [0,1], outcomes must be binary.')
    if not 2 <= bins <= 100:
        raise ValueError('Use 2–100 calibration bins.')
    assignment = np.minimum((scores*bins).astype(int),bins-1)
    table = []
    for index in range(bins):
        selected = assignment==index
        table.append({'lower':index/bins,'upper':(index+1)/bins,'n':int(selected.sum()),
            'mean_score':float(scores[selected].mean()) if selected.any() else None,
            'observed_correctness':float(outcomes[selected].mean()) if selected.any() else None})
    return {'n':len(scores),'brier_score':float(np.mean((scores-outcomes)**2)),
        'expected_calibration_error':sum(row['n']/len(scores)*abs(row['mean_score']-row['observed_correctness']) for row in table if row['n']),
        'bins':table,'scientifically_validated':False,
        'caveat':'Use independent grouped data, report group bootstrap intervals and lock any score calibrator before final testing.'}


def assess_metric_geometry(observations, *, maximum_relative_error):
    """Compare independent reference lengths at named positions and directions."""
    if not 0 < maximum_relative_error < 1:
        raise ValueError('Supply a predeclared acceptable relative error between zero and one.')
    rows = []
    locations = defaultdict(set)
    for item in observations:
        expected, measured = float(item['reference_mm']),float(item['measured_mm'])
        direction = item['direction']
        if direction not in {'horizontal','vertical'} or expected<=0 or measured<=0 or not np.isfinite([expected,measured]).all():
            raise ValueError('Geometry observations need positive independent lengths and horizontal/vertical direction.')
        locations[item['location']].add(direction)
        rows.append(dict(item,relative_error=(measured-expected)/expected))
    coverage = len(locations)>=3 and all(value=={'horizontal','vertical'} for value in locations.values())
    return {'coverage_sufficient_for_screening':coverage,'maximum_relative_error':maximum_relative_error,
        'within_predeclared_bound': bool(coverage and rows and max(abs(row['relative_error']) for row in rows)<=maximum_relative_error),
        'observations':rows,'scientifically_validated':False,
        'caveat':'Screening only. Establish camera distortion, plane/elevation, traceability, repeatability and the acquisition envelope before enabling physical measurements.'}


def stratified_errors(records, fields=('species','lot','session','density','overlap','pattern','glare','size_bin')):
    """Retain rare/atypical strata and rejected cases instead of averaging them away."""
    report = {}
    for field in fields:
        groups = defaultdict(list)
        for record in records:
            groups[str(record.get(field,'unknown'))].append(record)
        report[field] = {name:{'n':len(items),'mean_absolute_count_error':float(np.mean([
            abs(item['predicted_count']-item['true_count']) for item in items])),
            'false_positives':sum(item.get('false_positives',0) for item in items),
            'false_negatives':sum(item.get('false_negatives',0) for item in items),
            'rejected_true_seeds':sum(item.get('rejected_true_seeds',0) for item in items)} for name,items in groups.items()}
    return report
