"""Lazy single-launch-point benchmark adapters; no desktop or GPU imports here."""
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import tempfile

MAX_CONFIG_BYTES = 8 * 1024 * 1024


def _decode(raw):
    if len(raw) > MAX_CONFIG_BYTES:
        raise ValueError('Benchmark configuration exceeds its byte limit.')
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate benchmark JSON fields.')
            result[key] = value
        return result
    def finite(text):
        value = float(text)
        if not math.isfinite(value):
            raise ValueError('Benchmark JSON contains nonfinite numeric data.')
        return value
    def constant(_):
        raise ValueError('Benchmark JSON contains nonfinite numeric data.')
    try:
        return json.loads(raw.decode('utf-8'), object_pairs_hook=pairs,
                          parse_float=finite, parse_constant=constant)
    except (UnicodeError, RecursionError) as error:
        raise ValueError('Malformed benchmark JSON.') from error


def _read(path):
    path = Path(path)
    with path.open('rb') as stream:
        raw = stream.read(MAX_CONFIG_BYTES + 1)
    return path, raw, _decode(raw)


def _path(value, base):
    if type(value) is not str or not value.strip() or '\0' in value or len(value) > 4096:
        raise ValueError('Artifact/source paths must be nonempty bounded strings.')
    path = Path(value)
    return path if path.is_absolute() else base / path


def _publish(report, output, maximum):
    # Streaming encoding bounds admission before replacing any destination.
    encoded = bytearray()
    for piece in json.JSONEncoder(sort_keys=True, separators=(',', ':'),
                                   ensure_ascii=False, allow_nan=False).iterencode(report):
        piece = piece.encode('utf-8')
        if len(encoded) + len(piece) > maximum:
            raise ValueError('Benchmark report exceeds its byte limit.')
        encoded.extend(piece)
    output = Path(output)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_comparison(input_path, output_path, *, classifier_evidence=None):
    from seedvision.benchmarking import grouped_comparison as grouped
    manifest, records, plan = grouped.load_comparison_inputs(input_path)
    required_evidence = {method.method_id for method in (plan.reference, plan.candidate)
                         if method.model_sha256 is not None or method.training_evidence_sha256 is not None}
    if classifier_evidence is None:
        if required_evidence:
            raise ValueError('Classifier-bearing plan methods require --classifier-evidence with typed model/training provenance.')
        return grouped.export_grouped_comparison(output_path, manifest, records, plan)
    path, raw, evidence = _read(classifier_evidence)
    if type(evidence) is not dict or set(evidence) != {'bindings'}:
        raise ValueError('Classifier evidence requires exactly a bindings array.')
    bindings = evidence['bindings']
    if type(bindings) is not list or not 1 <= len(bindings) <= 2:
        raise ValueError('Classifier evidence needs one or two unique method bindings.')
    from seedvision.benchmarking.classifier_evaluation import bind_classifier_evaluation
    expected = {method.method_id: method for method in (plan.reference, plan.candidate)}
    seen, receipts = set(), []
    for entry in bindings:
        required = {'method_id', 'family', 'model_path'}
        if type(entry) is not dict or not required <= set(entry) or set(entry) - required - {'training_path'}:
            raise ValueError('Unexpected classifier evidence fields.')
        identifier = entry['method_id']
        if type(identifier) is not str or identifier not in expected or identifier in seen:
            raise ValueError('Unknown or duplicate classifier method binding.')
        if entry['family'] not in ('naive_bayes', 'kmeans'):
            raise ValueError('Unsupported classifier family.')
        method = expected[identifier]
        actual, receipt = bind_classifier_evaluation(manifest, family=entry['family'],
            model_path=_path(entry['model_path'], path.parent),
            training_path=(_path(entry['training_path'], path.parent) if entry.get('training_path') is not None else None),
            method_id=method.method_id, backend_id=method.backend_id,
            recipe_sha256=method.recipe_sha256, training_access_ids=method.training_access_ids)
        if actual != method:
            raise ValueError('Classifier evidence differs from the prespecified method binding.')
        if type(receipt) is not str:
            raise ValueError('Classifier adapter returned an invalid receipt.')
        decoded = _decode(receipt.encode('utf-8'))
        if type(decoded) is not dict or decoded.get('evaluation_eligible') is not True:
            raise ValueError('Classifier training evidence is not eligible for this evaluation.')
        if decoded.get('scientifically_validated') is not False:
            raise ValueError('Classifier evidence cannot claim scientific validation.')
        receipts.append({'method_id': identifier, 'method_sha256': method.identity_sha256, 'receipt': decoded})
        seen.add(identifier)
    if required_evidence - seen:
        raise ValueError('Classifier evidence omits a classifier-bearing plan method.')
    result = grouped.evaluate_grouped_comparison(manifest, records, plan)
    result['classifier_evidence'] = {'configuration_sha256': sha256(raw).hexdigest(), 'receipts': receipts}
    _publish(result, output_path, grouped.MAX_REPORT_BYTES)
    return result


def run_resources(input_path, output_path):
    path, _, configuration = _read(input_path)
    required = {'recipe', 'source'}
    optional = {'repeats', 'warmups', 'modes', 'device', 'maximum_elapsed_seconds',
                'maximum_peak_bytes', 'model_asset_paths', 'correction_records'}
    if type(configuration) is not dict or not required <= set(configuration) or set(configuration) - required - optional:
        raise ValueError('Unexpected native resource configuration fields.')
    if type(configuration['recipe']) is not dict:
        raise ValueError('Native recipe must be its exact serialized object.')
    controls = {key: value for key, value in configuration.items() if key in optional}
    for key, low in (('repeats', 1), ('warmups', 0)):
        if key in controls and (type(controls[key]) is not int or not low <= controls[key] <= 64):
            raise ValueError(f'Invalid benchmark {key}.')
    if 'modes' in controls:
        modes = controls['modes']
        if type(modes) is not list or not modes or any(type(i) is not str for i in modes) or len(set(modes)) != len(modes) or set(modes) - {'cold', 'warm'}:
            raise ValueError('Benchmark modes must be distinct cold/warm values.')
        controls['modes'] = tuple(modes)
    if 'device' in controls and (type(controls['device']) is not str or not re.fullmatch(r'cuda(?::[0-9]+)?', controls['device'])):
        raise ValueError('Native resource requests require a CUDA device; no CPU substitute.')
    if 'maximum_elapsed_seconds' in controls:
        value = controls['maximum_elapsed_seconds']
        if value is not None:
            try:
                valid = type(value) in (int, float) and math.isfinite(value) and value > 0
            except OverflowError:
                valid = False
            if not valid:
                raise ValueError('Invalid elapsed threshold.')
    if 'maximum_peak_bytes' in controls:
        value = controls['maximum_peak_bytes']
        if value is not None and (type(value) is not int or not 1 <= value <= 2**63 - 1):
            raise ValueError('Invalid peak threshold.')
    if 'model_asset_paths' in controls:
        assets = controls['model_asset_paths']
        if type(assets) is not dict or len(assets) > 16 or any(type(k) is not str or not k or len(k) > 128 for k in assets):
            raise ValueError('Invalid model asset paths.')
        controls['model_asset_paths'] = {key: _path(value, path.parent) for key, value in assets.items()}
    if 'correction_records' in controls:
        values = controls['correction_records']
        if type(values) is not list or len(values) > 1024 or any(type(i) is not dict for i in values):
            raise ValueError('Invalid correction observations.')
        controls['correction_records'] = tuple(values)
    from seedvision.persistence.native_recipes import native_recipes_from_payload
    recipe, = native_recipes_from_payload([configuration['recipe']])
    from seedvision.benchmarking.performance import benchmark_native_recipe, export_performance_report
    result = benchmark_native_recipe(recipe, _path(configuration['source'], path.parent), **controls)
    export_performance_report(result, output_path)
    return result
