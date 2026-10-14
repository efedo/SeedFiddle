"""Exact-revision review lookup shared by desktop and saved-project batch."""
import json
from pathlib import Path
import re


MAX_REVIEW_BYTES = 1024 * 1024


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate review key.')
        result[key] = value
    return result


def lookup_result_review(root, revision, *, project_path=None):
    """Return (review, state), distinguishing absent and unusable evidence."""
    if not isinstance(revision, str) or not re.fullmatch('[0-9a-f]{64}', revision):
        return None, 'unavailable'
    path = Path(root) / 'result-reviews' / (revision + '.json')
    try:
        with path.open('rb') as stream:
            raw = stream.read(MAX_REVIEW_BYTES + 1)
    except FileNotFoundError:
        # Restored history belongs to this private project, never the shared
        # workspace. A present primary (including unusable evidence) wins.
        if project_path is None:
            return None, 'not_recorded'
        parent = Path(project_path).resolve().parent
        if not parent.is_relative_to(Path(root).resolve()) or parent == Path(root).resolve():
            return None, 'not_recorded'
        try:
            with (parent / 'result-reviews' / (revision + '.json')).open('rb') as stream:
                raw = stream.read(MAX_REVIEW_BYTES + 1)
        except FileNotFoundError:
            return None, 'not_recorded'
    if len(raw) > MAX_REVIEW_BYTES:
        return None, 'malformed'
    try:
        review = json.loads(raw, object_pairs_hook=_unique_object)
    except (ValueError, UnicodeError):
        return None, 'malformed'
    if not isinstance(review, dict):
        return None, 'malformed'
    if review.get('result_revision') != revision:
        return None, 'incompatible'
    decisions = review.get('decisions')
    if not isinstance(decisions, dict) or any(
            not isinstance(key, str) or not re.fullmatch('[1-9][0-9]*', key)
            or not isinstance(state, str) or state not in {'accepted', 'excluded', 'unreviewed'}
            for key, state in decisions.items()):
        return None, 'malformed'
    return review, 'matched'
