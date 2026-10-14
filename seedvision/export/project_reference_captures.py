"""Execute saved original PlantCV seed selections via the isolated public runner."""
from pathlib import Path
import os
import tempfile

from seedvision.benchmarking.checkpoints import CheckpointCancelled, CheckpointLimits, _cancelled
from seedvision.benchmarking.native_image_inputs import stamp
from seedvision.benchmarking.recipes import RecipeProvenanceMode, RecipeSpec
from seedvision.benchmarking.reference_capture_session import capture_reference_path, open_reference_capture
from seedvision.benchmarking.reference_mask_stage import _selected_parameters
from seedvision.benchmarking.reference_preprocessing import _chain_stages, _validate_pinned_recipe
from seedvision.benchmarking.reference_seed_chain import _parameters
from seedvision.benchmarking.reference_stage import _calculation_parameters
from seedvision.benchmarking.reference_runner import PlantCVReferenceRunner
from seedvision.export import method_qc as qc, recipe_attempt_qc as attempts
from seedvision.export.project_native_graphs import _remove_owned_directory
from seedvision.export.reference_capture_qc import qc_from_reference_session, reference_capture_attempt_qc
from seedvision.persistence.method_qc_history import retain_method_qc
from seedvision.persistence.reference_graphs import validate_reference_graphs
from seedvision.persistence.reference_regions import file_sha256

FORMAT = 'seedfiddle.project-reference-captures'
VERSION = 1
FILENAME = 'reference-results.json'
ROOT = Path(__file__).resolve().parents[2]


def _admit_recipe(recipe):
    _validate_pinned_recipe(recipe)
    for stage in _chain_stages(recipe):
        if stage.operation_id.startswith('threshold.'):
            _calculation_parameters(recipe, stage.stage_id, stage.operation_id)
        else:
            _selected_parameters(recipe, stage.stage_id, stage.operation_id)
    _parameters(recipe)


def _poll(cancel):
    if _cancelled(cancel):
        raise CheckpointCancelled('Saved public reference capture cancelled.')


def export_project_reference_captures(attachments, image_path, directory, *, expected_source_sha256,
                                      runner_factory=None, limits=CheckpointLimits(), cancel=None,
                                      project_root=None, project_path=None):
    """Publish source-bound original reports without substituting a native graph.

    Historical attachments remain immutable metadata. Unsupported or unselected
    originals produce explicit unavailable rows without requesting a runner.
    Public CPU preparation/capture remains isolated from native CUDA caches.
    """
    attachments = validate_reference_graphs(attachments)
    entries = tuple((i, item) for i, item in enumerate(attachments)
                    if item.recipe.provenance.mode is RecipeProvenanceMode.ORIGINAL_REFERENCE)
    if not entries:
        return None
    qc._digest(expected_source_sha256)
    if not isinstance(limits, CheckpointLimits):
        raise TypeError('Reference capture limits must be explicit CheckpointLimits.')
    if project_path is not None and project_root is None:
        raise ValueError('Reference QC ownership requires its explicitly selected project root.')
    image_path, destination = Path(image_path), Path(directory).absolute()
    factory = runner_factory or (lambda: PlantCVReferenceRunner(
        ROOT / 'artifacts' / 'plantcv-reference-env' / 'Scripts' / 'python.exe'))
    rows, qualities = [], []
    staging = None
    recipe = entries[0][1].recipe
    request = attempts.capture_recipe_attempt_request(recipe, {'source_image': image_path})
    phase, invoked = 'admission', False

    def check_source():
        if file_sha256(image_path) != expected_source_sha256:
            raise ValueError('Project image bytes changed before public capture execution or publication.')

    def attempt(state, detail, error=None, session=None):
        return reference_capture_attempt_qc(recipe, state=state, phase=phase, detail=detail,
            expected_source_sha256=expected_source_sha256, inputs_before=request.payload['inputs'],
            invoked=invoked, session=session, exception=error)

    try:
        if destination.exists():
            raise FileExistsError(f'Reference capture destination already exists: {destination}')
        check_source()
        _poll(cancel)
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix='.reference-captures-', suffix='.incomplete', dir=destination.parent))
        for index, (attachment_index, attachment) in enumerate(entries):
            recipe = attachment.recipe
            request = attempts.capture_recipe_attempt_request(recipe, {'source_image': image_path})
            phase, invoked = 'between_rows', False
            _poll(cancel)
            phase = 'selection'
            row = {'attachment_index': attachment_index, 'recipe_identity': recipe.identity.to_dict(),
                'status': None, 'reason': '', 'manifest': None, 'manifest_sha256': None, 'method_qc': None}
            try:
                _admit_recipe(recipe)
            except ValueError as error:
                row.update(status='unavailable', reason=str(error)[:4096])
                quality = attempt('unavailable', row['reason'], error)
            else:
                session = opened = None
                try:
                    check_source()
                    selected_stamp = stamp(image_path)
                    runner = factory()
                    if type(runner) is not PlantCVReferenceRunner:
                        raise TypeError('Reference capture requires the actual pinned public runner.')
                    phase, invoked = 'capture', True
                    session = capture_reference_path(recipe, image_path, staging / f'capture-{index:03d}',
                        runner, expected_stamp=selected_stamp, limits=limits, cancel=cancel)
                    if session.verified is None:
                        row.update(status=session.outcome.status,
                            reason='Returned public chain has no complete verified manifest.')
                        quality = attempt(row['status'], row['reason'], session=session)
                    else:
                        # Drop the first reader's full-resolution arrays before
                        # independent verification of the persisted revision.
                        manifest = Path(session.outcome.manifest_path)
                        session = None
                        phase = 'verification'
                        opened = open_reference_capture(manifest, limits=limits, cancel=cancel)
                        if (opened.recipe != recipe or opened.verified.record['source_sha256'] != expected_source_sha256):
                            raise ValueError('Reopened public capture differs from the saved recipe/source.')
                        quality = qc_from_reference_session(opened)
                        row.update(status='executed', manifest=manifest.relative_to(staging).as_posix(),
                            manifest_sha256=opened.verified.manifest_sha256)
                except CheckpointCancelled:
                    raise
                except Exception as error:
                    row.update(status='failed', reason=f'{type(error).__name__}: {error}'[:4096])
                    quality = attempt('failed', row['reason'], error)
                finally:
                    session = opened = None
            row['method_qc'] = quality.payload
            rows.append(row)
            qualities.append(quality)
            phase, invoked = 'publication', False
            qc.export_method_qc(quality, staging / f'row-{index:03d}-method-qc.json')
        phase, invoked = 'publication', False
        check_source()
        _poll(cancel)
        payload = {'format': FORMAT, 'version': VERSION, 'source_sha256': expected_source_sha256,
            'preparation': 'isolated_local_cv2_decode_not_PlantCV_readimage', 'rows': rows,
            'scientifically_validated': False}
        qc._publish(staging / FILENAME, payload, limit=32 * 1024 * 1024)
        check_source()
        _poll(cancel)
        os.rename(staging, destination)
        staging = None
    except BaseException as error:
        try:
            if staging is not None:
                _remove_owned_directory(staging, destination.parent)
        except Exception as cleanup_error:
            error.add_note(f'Owned reference capture staging cleanup failed: {type(cleanup_error).__name__}: {cleanup_error}')
        if isinstance(error, Exception):
            state = 'cancelled' if isinstance(error, CheckpointCancelled) else 'rejected' if phase == 'admission' else 'failed'
            try:
                qualities.append(attempt(state, f'{type(error).__name__}: {error}', error))
                error.reference_qualities = tuple(qualities)
                history_root = destination.with_name(destination.name + '.qc-history')
                for quality in qualities:
                    retain_method_qc(history_root, quality)
                    if project_root is not None:
                        retain_method_qc(project_root, quality, project_path=project_path)
            except Exception as recording_error:
                error.add_note(f'Reference capture QC retention failed: {type(recording_error).__name__}: {recording_error}')
        raise
    if project_root is not None:
        for quality in qualities:
            retain_method_qc(project_root, quality, project_path=project_path)
    return destination / FILENAME


def read_project_reference_qc(path):
    """Read bounded recorded row facts without probing original sources/captures."""
    payload = qc._read(path, 32 * 1024 * 1024)
    qc._exact(payload, 'format version source_sha256 preparation rows scientifically_validated', 'reference ledger')
    if (payload['format'] != FORMAT or type(payload['version']) is not int or payload['version'] != VERSION
            or payload['scientifically_validated'] is not False
            or payload['preparation'] != 'isolated_local_cv2_decode_not_PlantCV_readimage'
            or type(payload['rows']) is not list or not 1 <= len(payload['rows']) <= 16):
        raise ValueError('Invalid recorded reference ledger.')
    qc._digest(payload['source_sha256'])
    qualities, seen = [], set()
    for index, row in enumerate(payload['rows']):
        qc._exact(row, 'attachment_index recipe_identity status reason manifest manifest_sha256 method_qc', 'reference row')
        item = qc.MethodQC(qc._bytes(row['method_qc']).decode('utf-8'))
        identity = item.payload['identity']
        capture = identity.get('reference_capture')
        if (type(row['attachment_index']) is not int or not 0 <= row['attachment_index'] < 16
                or row['attachment_index'] in seen or type(row['reason']) is not str or capture is None
                or qc._bytes(row['recipe_identity']) != qc._bytes(RecipeSpec.from_json(qc._bytes(item.payload['recipe'])).identity.to_dict())):
            raise ValueError('Reference row identity differs from its recorded QC.')
        seen.add(row['attachment_index'])
        if identity['scope'] == 'whole_reference_capture':
            digest = capture['manifest_sha256']
            if (row['status'] != 'executed' or row['reason'] != '' or row['manifest_sha256'] != digest
                    or row['manifest'] != f'capture-{index:03d}/chains/{digest}.json'
                    or capture['record']['source_sha256'] != payload['source_sha256']):
                raise ValueError('Reference row manifest/source differs from its verified recorded facts.')
        elif (identity['scope'] != 'reference_capture_attempt' or row['manifest'] is not None
                or row['manifest_sha256'] is not None or capture['state'] != row['status']
                or capture['expected_source_sha256'] != payload['source_sha256']):
            raise ValueError('Incomplete reference row contradicts its captured attempt.')
        qualities.append(item)
    return tuple(qualities)
