"""Complete JSON comparison with bounded diagnostics for large acceptance records."""
import math


_MISSING = object()


def _kind(value):
    if value is _MISSING:
        return 'missing'
    if value is None:
        return 'null'
    if isinstance(value, bool):
        return 'boolean'
    if isinstance(value, str):
        return 'string'
    if isinstance(value, int):
        return 'integer'
    if isinstance(value, float):
        return 'number' if math.isfinite(value) else 'invalid nonfinite number'
    if isinstance(value, dict):
        return 'object'
    if isinstance(value, list):
        return 'array'
    return 'invalid ' + type(value).__name__


def _display(value, maximum):
    kind = _kind(value)
    if kind == 'string':
        text = repr(value[:maximum])
    elif kind in ('object', 'array'):
        text = f'<{kind}: {len(value)} entries>'
    elif kind.startswith('invalid') or kind == 'missing':
        text = '<' + kind + '>'
    else:
        text = repr(value)
    return text if len(text) <= maximum else text[:maximum - 3] + '...'


def json_difference_summary(actual, expected, *, maximum_samples=50, maximum_value_characters=240):
    """Count every structural/value difference; retain only bounded examples.

    Object key order has no JSON meaning; array order does. String subclasses
    (including TorchVersion) remain JSON strings, while booleans and numeric
    representations retain distinct types. Invalid JSON values never compare equal.
    """
    if type(maximum_samples) is not int or not 1 <= maximum_samples <= 1000:
        raise ValueError('maximum_samples must be an integer in [1,1000]')
    if type(maximum_value_characters) is not int or not 16 <= maximum_value_characters <= 4096:
        raise ValueError('maximum_value_characters must be an integer in [16,4096]')
    summary = {'total_changes': 0, 'samples': []}

    def change(path, reason, left, right):
        summary['total_changes'] += 1
        if len(summary['samples']) < maximum_samples:
            summary['samples'].append({
                'path': path or '/', 'reason': reason,
                'actual': _display(left, maximum_value_characters),
                'expected': _display(right, maximum_value_characters),
            })

    def visit(left, right, path):
        left_kind, right_kind = _kind(left), _kind(right)
        if left_kind != right_kind or left_kind.startswith('invalid'):
            change(path, 'type', left, right)
        elif left_kind == 'object':
            if any(not isinstance(key, str) for key in left) or any(not isinstance(key, str) for key in right):
                change(path, 'nonstring object key', left, right)
                return
            for key in sorted(left.keys() | right.keys()):
                escaped = key.replace('~', '~0').replace('/', '~1')
                visit(left.get(key, _MISSING), right.get(key, _MISSING), path + '/' + escaped)
        elif left_kind == 'array':
            if len(left) != len(right):
                change(path, 'array length', left, right)
            for index in range(max(len(left), len(right))):
                visit(left[index] if index < len(left) else _MISSING,
                      right[index] if index < len(right) else _MISSING, path + '/' + str(index))
        elif left != right:
            change(path, 'value', left, right)

    visit(actual, expected, '')
    return summary


def assert_json_equal(testcase, actual, expected, *, maximum_samples=50, maximum_value_characters=240):
    """Fail on any difference without rendering an unbounded unittest dict diff."""
    summary = json_difference_summary(actual, expected, maximum_samples=maximum_samples,
                                      maximum_value_characters=maximum_value_characters)
    if summary['total_changes']:
        import json
        testcase.fail(f"JSON acceptance mismatch: {summary['total_changes']} changes; "
                      f"first {len(summary['samples'])}: " + json.dumps(summary['samples'], ensure_ascii=True))
