"""Detached terminal evidence for native runs that produced no authoritative result."""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import logging
import os
from pathlib import Path
import shutil
from uuid import uuid4

import numpy as np

from seedvision.export.analysis_inputs import _array_fact
from seedvision.export.artifact_identity import (capture_artifact_selections,
    semantic_artifact, verified_artifact_facts)
from seedvision.learning.data import file_sha256
from seedvision.timing import AnalysisCancelled


FORMAT = 'seedfiddle-native-run-outcome'
CONTRACT = 'seedfiddle.native-run-outcome-v1'
LIMIT = 4 * 1024 * 1024


class AnalysisInputIntegrityError(ValueError):
    """A source/reference binding violation that blocks independent continuation."""


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _source_digest(path):
    try:
        return file_sha256(path)
    except OSError:
        return None


def _request(path, args, kwargs):
    from seedvision.segmentation.baseline import BaselineSettings
    from seedvision.calibration.image import CalibrationSettings
    from seedvision.calibration.geometry import DishDetectionSettings
    from seedvision.visualization import AnalysisLayerSettings, AdvancedAnalysisSettings
    from seedvision.segmentation.procedural import ProceduralInstanceSettings
    from seedvision.learning.pipeline import UNetPipelineSettings, StarDistPipelineSettings
    from seedvision.learning.cellpose_sam import CellposeSamSettings
    from seedvision.learning.trait_pipeline import TraitClassifierSettings
    defaults = {'settings': BaselineSettings, 'calibration_settings': CalibrationSettings,
        'dish_settings': DishDetectionSettings, 'layer_settings': AnalysisLayerSettings,
        'advanced_settings': AdvancedAnalysisSettings, 'procedural_settings': ProceduralInstanceSettings,
        'unet_settings': UNetPipelineSettings, 'stardist_settings': StarDistPipelineSettings,
        'cellpose_settings': CellposeSamSettings, 'trait_settings': TraitClassifierSettings}
    supplied = dict(kwargs)
    if len(args) > 8:
        raise TypeError('Too many positional analysis settings.')
    for name, value in zip(defaults, args):
        if name in supplied:
            raise TypeError(f'Duplicate analysis setting {name}.')
        supplied[name] = value
    settings = {name: asdict(factory() if supplied.get(name) is None else supplied[name])
                for name, factory in defaults.items()}
    enabled = supplied.get('enabled_nodes')
    root = Path(supplied.get('learning_root') or Path.cwd())
    artifact_capture = capture_artifact_selections(root, selections={
        'unet_instances': settings['unet_settings']['checkpoint_path'],
        'stardist_instances': settings['stardist_settings']['checkpoint_path'],
        'cellpose_sam_instances': settings['cellpose_settings']['pretrained_model'],
        'trait_classifier': settings['trait_settings']['trait_model_path']}, enabled=enabled or ())
    artifacts = verified_artifact_facts(artifact_capture)
    for node, setting, field in (('unet_instances', 'unet_settings', 'checkpoint_path'),
            ('stardist_instances', 'stardist_settings', 'checkpoint_path'),
            ('cellpose_sam_instances', 'cellpose_settings', 'pretrained_model'),
            ('trait_classifier', 'trait_settings', 'trait_model_path')):
        settings[setting][field] = semantic_artifact(artifacts[node], node)
    library = supplied.get('species_library')
    manifest = None if library is None else library.manifest
    shape_bank = supplied.get('library_shape_bank')
    source = _source_digest(path)
    facts = {'contract': 'seedfiddle.supplied-analysis-request-v1', 'settings': settings,
        'enabled_nodes': None if enabled is None else sorted(enabled),
        'supplied_masks': {name: _array_fact(None if supplied.get(name) is None else
            np.asarray(supplied[name], dtype=np.uint16 if name == 'seed_instance_annotations' else bool)) for name in (
            'background_reference_mask', 'foreground_reference_mask', 'background_exclusion_mask',
            'foreground_exclusion_mask', 'seed_instance_annotations')},
        'supplied_reference_transform': _array_fact(None if supplied.get('reference_transform') is None else
                                                   np.asarray(supplied['reference_transform'], dtype=float)),
        'supplied_reference_source_bindings': {name: supplied.get(name) for name in (
            'reference_archive_source_sha256', 'reference_annotation_snapshot_source_sha256')},
        'supplied_annotations': [asdict(value) for value in supplied.get('seed_instance_traits', ())],
        'supplied_reference_points': {name: supplied.get(name, ()) for name in (
            'background_reference_points', 'foreground_reference_points')},
        'manual_seed_centres': None if supplied.get('manual_seed_centres') is None else asdict(supplied['manual_seed_centres']),
        'species': supplied.get('species', 'unknown'),
        'biological_context': None if supplied.get('biological_context') is None else asdict(supplied['biological_context']),
        'trait_vocabulary': {'species': supplied.get('seed_trait_species', ''),
            'coat_patterns': supplied.get('seed_trait_coat_patterns', ()),
            'conditions': supplied.get('seed_trait_conditions', ())},
        'background_colour_enabled': supplied.get('background_colour_enabled', True),
        'species_library': None if manifest is None else {'library_id': manifest.library_id,
            'version': manifest.version, 'species_id': manifest.species_id, 'sha256': manifest.content_sha256},
        'library_shape_bank_sha256': None if shape_bank is None else sha256(_json(asdict(shape_bank)).encode()).hexdigest()}
    return json.loads(_json(facts)), source, artifact_capture


@dataclass(frozen=True, slots=True)
class NativeRunOutcome:
    image_path: Path
    payload_json: str

    def __post_init__(self):
        payload = self.payload
        if (payload['format'] != FORMAT or not isinstance(self.image_path, Path)
                or str(self.image_path) != payload['execution']['source_path']):
            raise ValueError('Terminal outcome locator does not match its captured execution.')

    @property
    def payload(self):
        return load_run_outcome(self.payload_json)


def run_native_analysis(path, *args, analysis_runner=None, cancel_outcome_receiver=None, **kwargs):
    """Successful results are unchanged; exceptions become explicit terminal evidence."""
    from seedvision.segmentation.baseline import analyze_path
    from seedvision.export.preflight_outcomes import begin_preflight
    path = Path(path)
    attempt = begin_preflight(path, 'image', 'request_capture')
    try:
        request, source, artifact_capture = _request(path, args, kwargs)
        _validate_request(request)
    except Exception as error:
        return attempt.outcome(error)
    callback = kwargs.get('progress_callback')
    last_started = None

    def cancelled_before_install():
        return _terminal_outcome(path, request, source, artifact_capture,
            'cancelled', 'analysis_cancelled', last_started, 'AnalysisCancelled',
            'Analysis cancelled before result installation.')

    if cancel_outcome_receiver is not None:
        # A desktop worker can preserve the same request if Stop arrives after
        # this wrapper returns but before it emits a completed result.
        cancel_outcome_receiver(cancelled_before_install)

    def progress(node, state):
        nonlocal last_started
        if state == 'started':
            last_started = node
        if callback is not None:
            callback(node, state)

    try:
        # The same checks occur at the direct API boundary. They remain typed
        # here even when a caller supplies a runner for testing/integration.
        from seedvision.measurement.lineage import reference_source_identity_state
        for name, expected in request['supplied_reference_source_bindings'].items():
            try:
                reference_source_identity_state(source, expected, name)
            except ValueError as error:
                raise AnalysisInputIntegrityError(str(error)) from error
        result = (analysis_runner or analyze_path)(path, *args, **dict(kwargs, progress_callback=progress))
        cancellation = kwargs.get('cancellation_requested')
        if cancellation is not None and cancellation():
            raise AnalysisCancelled('Analysis cancelled after calculation.')
        if source is not None and _source_digest(path) != source:
            raise AnalysisInputIntegrityError('Source image changed during calculation.')
        return result
    except AnalysisCancelled as error:
        state, reason = 'cancelled', 'analysis_cancelled'
        failure = (type(error).__name__, str(error) or type(error).__name__)
    except AnalysisInputIntegrityError as error:
        state, reason = 'rejected', 'input_integrity_failed'
        failure = (type(error).__name__, str(error) or type(error).__name__)
    except Exception as error:
        logging.getLogger(__name__).exception('Native analysis did not complete; image=%s', path)
        state, reason = ('unavailable', 'source_unavailable') if source is None else ('failed', 'analysis_exception')
        import torch
        if source is not None and (isinstance(error, torch.cuda.OutOfMemoryError) or
                (isinstance(error, RuntimeError) and any(text in str(error).lower()
                 for text in ('cuda', 'device-side assert', 'illegal memory access')))):
            reason = 'device_failure'
        failure = (type(error).__name__, str(error) or type(error).__name__)
    # Exception variables are cleared by Python after an except block.
    # Keep the actual diagnostic explicitly before leaving that boundary.
    return _terminal_outcome(path, request, source, artifact_capture, state, reason, last_started, *failure)


def _terminal_outcome(path, request, source, artifact_capture, state, reason, last_started,
                      error_type=None, diagnostic=None):
    current_source = _source_digest(path)
    if state != 'cancelled' and source is not None and source != current_source:
        state, reason = 'rejected', 'source_changed'
    try:
        verified_artifact_facts(artifact_capture)
        dependency_verification = 'unchanged'
    except ValueError:
        dependency_verification = 'changed'
        if state != 'cancelled':
            state, reason = 'rejected', 'dependency_changed'
    except OSError:
        dependency_verification = 'unavailable'
        if state != 'cancelled':
            state, reason = 'rejected', 'dependency_changed'
    from seedvision import __version__
    package = Path(__file__).resolve().parents[1]
    identity = {'contract': CONTRACT, 'request': request,
        'source_at_request': {'sha256': source, 'identity_state': 'observed' if source else 'unavailable',
                              'opaque_locator': None if source else str(path.absolute())},
        'source_at_completion': current_source, 'dependency_verification_at_completion': dependency_verification,
        'state': state, 'reason_code': reason,
        'last_started_node': last_started, 'error_type': error_type,
        'calibration_facts': None, 'prediction_revision': None, 'backend_outcomes': None,
        'implementation': {'version': __version__, 'files': {name: file_sha256(package / name) for name in (
            'export/run_outcomes.py', 'segmentation/baseline.py', 'export/analysis_inputs.py',
            'export/artifact_identity.py', 'timing.py')}}}
    payload = {'format': FORMAT, 'version': 1, 'identity': identity,
        'outcome_revision': sha256(_json(identity).encode()).hexdigest(),
        'diagnostic': diagnostic, 'execution': {'source_path': str(path), 'artifact_selections': json.loads(artifact_capture)}}
    raw = _json(payload)
    load_run_outcome(raw)
    return NativeRunOutcome(path, raw)


def _unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('Duplicate run outcome key.')
        value[key] = item
    return value


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def load_run_outcome(raw):
    """Validate historical evidence without claiming its old inputs are current."""
    if not isinstance(raw, str) or len(raw.encode('utf-8')) > LIMIT:
        raise ValueError('Run outcome exceeds its bounded JSON contract.')
    try:
        payload = json.loads(raw, object_pairs_hook=_unique,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (TypeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError('Invalid run outcome JSON.') from error
    if isinstance(payload, dict) and payload.get('format') == 'seedfiddle-native-preflight-outcome':
        from seedvision.export.preflight_outcomes import validate_preflight_payload
        return validate_preflight_payload(payload)
    if (not isinstance(payload, dict) or set(payload) != {'format', 'version', 'identity', 'outcome_revision', 'diagnostic', 'execution'}
            or payload['format'] != FORMAT or type(payload['version']) is not int or payload['version'] != 1):
        raise ValueError('Unsupported run outcome format.')
    identity = payload['identity']
    if not isinstance(identity, dict) or set(identity) != {'contract', 'request', 'source_at_request',
            'source_at_completion', 'dependency_verification_at_completion',
            'state', 'reason_code', 'last_started_node', 'error_type',
            'calibration_facts', 'prediction_revision', 'backend_outcomes', 'implementation'}:
        raise ValueError('Invalid run outcome identity.')
    reasons = {'failed': {'analysis_exception', 'device_failure'}, 'cancelled': {'analysis_cancelled'},
               'rejected': {'input_integrity_failed', 'source_changed', 'dependency_changed'},
               'unavailable': {'source_unavailable'}}
    if (identity['contract'] != CONTRACT or not isinstance(identity['state'], str) or identity['state'] not in reasons
            or not isinstance(identity['reason_code'], str) or identity['reason_code'] not in reasons[identity['state']]
            or any(identity[name] is not None for name in ('calibration_facts', 'prediction_revision', 'backend_outcomes'))):
        raise ValueError('Terminal outcomes cannot claim completed predictions or calibration.')
    source = identity['source_at_request']
    if not isinstance(source, dict) or set(source) != {'sha256', 'identity_state', 'opaque_locator'}:
        raise ValueError('Invalid run outcome source facts.')
    if source['sha256'] is None:
        if source['identity_state'] != 'unavailable' or not isinstance(source['opaque_locator'], str) or not source['opaque_locator']:
            raise ValueError('An unavailable source retains its opaque selection identity.')
    elif not _sha(source['sha256']) or source['identity_state'] != 'observed' or source['opaque_locator'] is not None:
        raise ValueError('Invalid observed source identity.')
    if identity['source_at_completion'] is not None and not _sha(identity['source_at_completion']):
        raise ValueError('Invalid completion source identity.')
    if identity['dependency_verification_at_completion'] not in ('unchanged', 'changed', 'unavailable'):
        raise ValueError('Invalid completion dependency verification.')
    if identity['state'] == 'failed' and (source['sha256'] is None or source['sha256'] != identity['source_at_completion']):
        raise ValueError('A computation failure requires a stable observed source.')
    if identity['state'] == 'unavailable' and source['sha256'] is not None:
        raise ValueError('Source-unavailable outcomes cannot claim an observed source.')
    for name in ('last_started_node', 'error_type'):
        if identity[name] is not None and (not isinstance(identity[name], str) or not identity[name]):
            raise ValueError('Invalid run failure attribution.')
    if identity['error_type'] is None:
        raise ValueError('Terminal run evidence requires an actual exception type.')
    _validate_request(identity['request'])
    implementation = identity['implementation']
    if (not isinstance(implementation, dict) or set(implementation) != {'version', 'files'}
            or not isinstance(implementation['version'], str) or not implementation['version']
            or not isinstance(implementation['files'], dict) or set(implementation['files']) != {
                'export/run_outcomes.py', 'segmentation/baseline.py', 'export/analysis_inputs.py',
                'export/artifact_identity.py', 'timing.py'}
            or any(not _sha(value) for value in implementation['files'].values())):
        raise ValueError('Invalid run implementation identity.')
    if not _sha(payload['outcome_revision']) or payload['outcome_revision'] != sha256(_json(identity).encode()).hexdigest():
        raise ValueError('Run outcome revision does not bind its identity.')
    execution = payload['execution']
    if not isinstance(execution, dict) or set(execution) != {'source_path', 'artifact_selections'} or not isinstance(execution['source_path'], str):
        raise ValueError('Invalid run execution evidence.')
    if payload['diagnostic'] is not None and not isinstance(payload['diagnostic'], str):
        raise ValueError('Invalid run diagnostic.')
    return payload


def _validate_request(request):
    request_fields = {'contract', 'settings', 'enabled_nodes', 'supplied_masks', 'supplied_reference_transform',
        'supplied_reference_source_bindings', 'supplied_annotations', 'supplied_reference_points',
        'manual_seed_centres', 'species', 'biological_context', 'trait_vocabulary',
        'background_colour_enabled', 'species_library', 'library_shape_bank_sha256'}
    settings_names = {'settings', 'calibration_settings', 'dish_settings', 'layer_settings', 'advanced_settings',
                      'procedural_settings', 'unet_settings', 'stardist_settings', 'trait_settings'}
    # Requests recorded before the Cellpose-SAM node existed lack its settings.
    accepted_settings = (settings_names, settings_names | {'cellpose_settings'})
    mask_names = {'background_reference_mask', 'foreground_reference_mask', 'background_exclusion_mask',
                  'foreground_exclusion_mask', 'seed_instance_annotations'}
    if (not isinstance(request, dict) or set(request) != request_fields
            or request['contract'] != 'seedfiddle.supplied-analysis-request-v1'
            or not isinstance(request['settings'], dict) or set(request['settings']) not in accepted_settings
            or any(not isinstance(value, dict) for value in request['settings'].values())
            or not isinstance(request['supplied_masks'], dict) or set(request['supplied_masks']) != mask_names):
        raise ValueError('Invalid supplied run request facts.')
    enabled = request['enabled_nodes']
    if enabled is not None and (not isinstance(enabled, list) or any(not isinstance(x, str) or not x for x in enabled)
                                or enabled != sorted(set(enabled))):
        raise ValueError('Invalid supplied enabled-node inventory.')
    for value in [*request['supplied_masks'].values(), request['supplied_reference_transform']]:
        if value is None:
            continue
        if (not isinstance(value, dict) or set(value) != {'shape', 'dtype', 'sha256'}
                or not isinstance(value['shape'], list) or len(value['shape']) != 2
                or any(type(x) is not int or x <= 0 for x in value['shape'])
                or not isinstance(value['dtype'], str) or not _sha(value['sha256'])):
            raise ValueError('Invalid supplied raster/frame facts.')
        try:
            dtype = np.dtype(value['dtype'])
        except (ValueError, TypeError) as error:
            raise ValueError('Invalid supplied raster/frame dtype.') from error
        if dtype.kind not in 'biuf':
            raise ValueError('Invalid supplied raster/frame dtype.')
    bindings = request['supplied_reference_source_bindings']
    if (not isinstance(bindings, dict) or set(bindings) != {'reference_archive_source_sha256', 'reference_annotation_snapshot_source_sha256'}
            or any(value is not None and not _sha(value) for value in bindings.values())):
        raise ValueError('Invalid supplied reference source bindings.')
    points = request['supplied_reference_points']
    vocabulary = request['trait_vocabulary']
    if (not isinstance(request['species'], str) or type(request['background_colour_enabled']) is not bool
            or not isinstance(request['supplied_annotations'], list)
            or any(not isinstance(value, dict) for value in request['supplied_annotations'])
            or any(request[name] is not None and not isinstance(request[name], dict)
                   for name in ('biological_context', 'manual_seed_centres'))
            or not isinstance(vocabulary, dict) or set(vocabulary) != {'species', 'coat_patterns', 'conditions'}
            or not isinstance(vocabulary['species'], str)
            or any(not isinstance(vocabulary[name], list) or any(not isinstance(value, str) for value in vocabulary[name])
                   for name in ('coat_patterns', 'conditions'))
            or not isinstance(points, dict) or set(points) != {'background_reference_points', 'foreground_reference_points'}
            or any(not isinstance(value, list) or any(not isinstance(point, list) or len(point) != 2
                   or any(type(number) not in (int, float) for number in point) for point in value)
                   for value in points.values())):
        raise ValueError('Invalid supplied request metadata containers.')
    library = request['species_library']
    if (library is not None and (not isinstance(library, dict)
            or set(library) != {'library_id', 'version', 'species_id', 'sha256'}
            or any(not isinstance(library[name], str) or not library[name] for name in ('library_id', 'species_id'))
            or type(library['version']) not in (int, str) or not _sha(library['sha256']))
            or request['library_shape_bank_sha256'] is not None and not _sha(request['library_shape_bank_sha256'])):
        raise ValueError('Invalid supplied library facts.')


def export_run_outcome(directory, outcome):
    """Publish a separate immutable terminal record; no completed-result marker."""
    payload = outcome.payload
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    identifier = payload['outcome_revision'][:16] + '-' + uuid4().hex[:8]
    staging = root / ('.run-outcome-' + identifier + '.incomplete')
    destination = root / identifier
    staging.mkdir(exist_ok=False)
    try:
        with (staging / 'run-outcome.json').open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(_json(payload) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        staging.rename(destination)
        return destination
    finally:
        if staging.exists():
            if staging.resolve().parent != root.resolve():
                raise ValueError('Run outcome cleanup escaped its export directory.')
            shutil.rmtree(staging)


def acknowledge_run_outcome(directory, outcome, operator):
    if not isinstance(operator, str) or not operator.strip():
        raise ValueError('Failure acknowledgment requires an operator identity.')
    payload = outcome.payload
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    record = {'contract': 'seedfiddle.run-outcome-acknowledgment-v1',
        'outcome_revision': payload['outcome_revision'], 'decision': 'acknowledged',
        'operator_account': operator, 'reviewed_at_utc': datetime.now(timezone.utc).isoformat()}
    path = root / (payload['outcome_revision'] + '-' + uuid4().hex + '.json')
    temporary = path.with_suffix('.incomplete')
    try:
        with temporary.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(_json(record) + '\n'); stream.flush(); os.fsync(stream.fileno())
        temporary.rename(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path
