"""Immutable CPU snapshots and saved calibrated public benchmark sessions.

The native CUDA graph does not use this path. Preparation is one local OpenCV
color decode, not a public PlantCV reader. Numeric admission excludes encoded
bytes, metadata, codec/validation/public scratch and process peak memory.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path
import re
import tempfile

import cv2
import numpy as np

from . import native_image_inputs as inputs
from .checkpoints import (
    CheckpointCancelled, CheckpointIdentity, CheckpointLimits, CoordinateFrame,
    ExecutionRecord, ExecutionStatus, MeasurementContract, RasterRole, _canonical_json, _cancelled,
    _thaw_json, publish_checkpoint,
)
from .reference_calibrated_seed import (
    ReferenceCalibratedSeedError, ReferenceCalibratedSeedOutcome,
    ReferenceCalibratedSeedSpec, _bounds, run_reference_calibrated_seed,
)
from .reference_calibrated_seed_reader import (
    VerifiedReferenceCalibratedSeed, read_reference_calibrated_seed,
)
from .reference_capture_session import reference_measurement
from .reference_runner import (
    MAX_ARRAY_BYTES, MAX_ARRAY_ELEMENTS, MAX_INPUT_ARCHIVE_BYTES,
    PlantCVReferenceRunner, _strict_json_loads,
)
from .reference_seed_chain import ReferenceRasterInput
from .reference_watershed_capture import _admit


SIDECAR_SCHEMA = 'seedfiddle.public-calibrated-capture-session.v1'
PREPARATION_SCHEMA = 'seedfiddle.prepared-calibrated-public-input.v1'
OUTPUT_MEASUREMENT = 'public_calibrated_reference_chain_output'


@dataclass(frozen=True, slots=True)
class ReferenceCalibratedCaptureSession:
    spec: ReferenceCalibratedSeedSpec
    outcome: ReferenceCalibratedSeedOutcome | None
    verified: VerifiedReferenceCalibratedSeed | None
    manifest_path: Path | None = None
    manifest_sha256: str | None = None

    def __post_init__(self):
        if (self.manifest_path is None) != (self.manifest_sha256 is None):
            raise ValueError('A saved session requires both manifest address and digest.')
        if (self.verified is None) != (self.manifest_sha256 is None):
            raise ValueError('Only a fully verified session exposes a saved revision.')


def _poll(cancel):
    if _cancelled(cancel):
        raise CheckpointCancelled('Calibrated public capture session cancelled.')


def _limits(limits):
    if not isinstance(limits, CheckpointLimits):
        raise TypeError('Explicit checkpoint limits are required.')


def _code_identity():
    return {'session_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
            'input_helper_sha256':sha256(Path(inputs.__file__).read_bytes()).hexdigest()}


def _resource_plan(shape, spec, limits):
    _bounds(spec.seed_roi_xyxy, shape)
    if spec.card_parameters['roi_bounds_xyxy'] is not None:
        _bounds(spec.card_parameters['roi_bounds_xyxy'], shape)
    parent_pixels = shape[0]*shape[1]
    x0,y0,x1,y1 = spec.seed_roi_xyxy
    crop_shape = (y1-y0,x1-x0)
    crop_pixels = crop_shape[0]*crop_shape[1]
    parent_bytes = 3*parent_pixels
    # A full RGB request has a separate transmission bound. A full parent
    # watershed snapshot is never requested; apply that envelope to the crop.
    if (parent_bytes > min(limits.max_array_bytes, MAX_ARRAY_BYTES,
                           MAX_INPUT_ARCHIVE_BYTES-1024*1024)
            or parent_bytes > MAX_ARRAY_ELEMENTS):
        raise ReferenceCalibratedSeedError('Full parent RGB exceeds its public transmission limit.')
    inputs.decoded_bound(2*parent_pixels, color=True, gray=False)
    _admit(crop_shape, min(limits.max_total_array_bytes, MAX_ARRAY_BYTES), MAX_ARRAY_ELEMENTS)
    # Reader retains parent+correction (6P), two RGB/gray crop pairs (8Q),
    # five uint8 preprocessing rasters (5Q), watershed (42Q+3), two analysis
    # RGB/label contexts (14Q) and separate scaled RGB/label context (7Q).
    retained_envelope = 6*parent_pixels+76*crop_pixels+3
    if retained_envelope > limits.max_total_array_bytes:
        raise ReferenceCalibratedSeedError('Complete retained numeric envelope exceeds its session limit.')
    return {'prepared_numeric_envelope_bytes':6*parent_pixels,
            'complete_retained_numeric_envelope_bytes':retained_envelope,
            'retained_checkpoint_entries':15,
            'watershed_shape':list(crop_shape),
            'scope':'Numeric owned preparation or retained reader arrays; excludes encoded bytes, metadata, '
                    'temporary validation/codec/loader/public scratch and process RSS.'}


def _prepared_contract(source, shape, code):
    decoder = inputs.versions()
    transformation = {'decoder':decoder, 'source_sha256':source.sha256,
        'source_size':source.size, 'shape':list(shape), 'code':code,
        'procedure':'one local OpenCV color byte decode; BGR to contiguous RGB; no parent gray'}
    frame = CoordinateFrame('prepared_calibrated_public_reference_frame', shape,
        'top_left_pixel_centres', 'x_right_y_down', 'pixel',
        sha256(_canonical_json(transformation, name='calibrated prepared frame')).hexdigest())
    contract = transformation | {'coordinate_frame':frame.to_dict(), 'actual_public_reader':False}
    digest = sha256(_canonical_json(contract, name='calibrated prepared contract')).hexdigest()
    identity = CheckpointIdentity(
        source.sha256, frame, 'prepared_calibrated_reference_rgb',
        'seedfiddle_prepared_calibrated_reference', '1', digest, digest, digest,
        'seedfiddle.calibrated_reference_input', '1',
        {'opencv':str(cv2.__version__), 'numpy':str(np.__version__)},
        {'color_order':'RGB', 'dtype':'uint8', 'value_range':[0,255],
         'channel_conversion':'BGR to RGB from one local OpenCV color byte decode',
         'geometric_transform':'decoder orientation; no rectification', 'interpolation':'decoder_defined'},
        {'foreground_polarity':'light', 'representation':'uint8_0_255',
         'connectivity':'not_applicable', 'validity_region':'full_frame', 'border_policy':'not_applicable'},
        {'label_dtype':'int32', 'background_label':0, 'label_id_mapping':'not_created',
         'split_boundary_ownership':'not_applicable', 'visibility':'full_frame', 'eligibility':'not_evaluated'},
        source.sha256)
    return frame, contract, identity


def _verify_preparation(verified, spec, limits):
    parent = verified.checkpoints['parent']
    contract = _thaw_json(parent.metadata).get('preparation', {})
    size = contract.get('source_size')
    if type(size) is not int or not 1 <= size <= inputs.MAX_ENCODED_BYTES:
        raise ReferenceCalibratedSeedError('Prepared parent lacks a bounded original byte identity.')
    source = inputs.SourceIdentity(size, verified.record['source_sha256'])
    frame, expected, identity = _prepared_contract(source, parent.identity.coordinate_frame.shape, _code_identity())
    # Stored metadata records the size envelope, not the original caller's
    # limit. The reader has already admitted actual retained allocations under
    # its supplied (possibly smaller) budget before this metadata comparison.
    plan = _resource_plan(frame.shape, spec, CheckpointLimits())
    expected_metadata = {'schema':PREPARATION_SCHEMA, 'preparation':expected,
                         'resource_plan':plan, 'actual_public_reader':False}
    measurement = reference_measurement('prepared_calibrated_reference_rgb')
    role = RasterRole('image','|u1','HWC',(0,255))
    if (parent.identity != identity or parent.measurement != measurement
            or parent.execution.status != ExecutionStatus.AVAILABLE
            or set(parent.arrays) != {'rgb'} or dict(parent.raster_roles) != {'rgb':role}
            or _canonical_json(_thaw_json(parent.metadata), name='prepared metadata')
               != _canonical_json(expected_metadata, name='expected prepared metadata')):
        raise ReferenceCalibratedSeedError('Saved calibrated preparation differs from its local snapshot contract.')


def _bounded_bytes(path, limits):
    with Path(path).open('rb') as stream:
        raw = stream.read(limits.max_metadata_bytes+1)
    if len(raw) > limits.max_metadata_bytes:
        raise ReferenceCalibratedSeedError('Saved calibrated metadata exceeds its cap.')
    return raw


def _retract_owned_manifest(outcome, store, limits):
    expected = (store/'calibrated'/'chains'/(outcome.manifest_sha256+'.json')).resolve()
    if Path(outcome.manifest_path).resolve() != expected:
        return
    try:
        if expected.is_file() and sha256(_bounded_bytes(expected, limits)).hexdigest() == outcome.manifest_sha256:
            expected.unlink()
    except (OSError, ReferenceCalibratedSeedError):
        # Cleanup must preserve a changed replacement and the original failure.
        pass


def capture_calibrated_reference_path(spec: ReferenceCalibratedSeedSpec, source_path: Path | str,
    store: Path | str, runner: PlantCVReferenceRunner, *, expected_stamp=None,
    limits: CheckpointLimits=CheckpointLimits(), cancel=None) -> ReferenceCalibratedCaptureSession:
    """Capture from one immutable snapshot and commit its verified spec sidecar.

The last caller cancellation/source check precedes a callback-free saved graph
audit and sidecar replace. Cancellation after this commit keeps the revision.
There is no transaction-wide lock on concurrent external file writers.
"""
    _limits(limits)
    if type(spec) is not ReferenceCalibratedSeedSpec or type(runner) is not PlantCVReferenceRunner:
        raise TypeError('An exact calibrated composition and public runner are required.')
    spec_sha = spec.sha256
    source_path, store = Path(source_path), Path(store)
    _poll(cancel)
    selected_stamp = inputs.stamp(source_path) if expected_stamp is None else expected_stamp
    code = _code_identity()
    def poll(**_kwargs):
        _poll(cancel)
        if inputs.stamp(source_path) != selected_stamp or selected_stamp is None:
            raise ReferenceCalibratedSeedError('Selected image changed during calibrated capture.')
        if spec.sha256 != spec_sha or _code_identity() != code:
            raise ReferenceCalibratedSeedError('Selected composition or local preparation code changed.')
        return False
    poll()
    source = inputs.source_identity(source_path, poll)
    raw = inputs.snapshot(source_path, source, poll)
    bgr = inputs.decode(raw)
    del raw
    shape = tuple(bgr.shape[:2])
    plan = _resource_plan(shape, spec, limits)
    frame, contract, identity = _prepared_contract(source, shape, code)
    rgb = np.ascontiguousarray(bgr[...,::-1])
    del bgr
    role = RasterRole('image','|u1','HWC',(0,255))
    measurement = reference_measurement('prepared_calibrated_reference_rgb')
    poll()
    reference = publish_checkpoint(store/'inputs', identity, {'rgb':rgb},
        execution=ExecutionRecord(ExecutionStatus.AVAILABLE), measurement=measurement,
        raster_roles={'rgb':role}, metadata={'schema':PREPARATION_SCHEMA, 'preparation':contract,
            'resource_plan':plan, 'actual_public_reader':False},
        allow_cpu_transfer=True, limits=limits, cancel=poll)
    del rgb
    parent = ReferenceRasterInput(store/'inputs', reference.checkpoint_sha256, identity, measurement, 'rgb', role)
    output_measurement = reference_measurement(OUTPUT_MEASUREMENT)
    outcome = run_reference_calibrated_seed(parent, spec, runner, store,
        expected_spec_sha256=spec_sha, expected_source_sha256=source.sha256,
        expected_parent_frame=frame, output_measurement=output_measurement,
        allow_cpu_transfer=True, limits=limits, cancel=poll)
    if outcome.manifest_sha256 is None:
        poll()
        return ReferenceCalibratedCaptureSession(spec, outcome, None)
    manifest = (store/'calibrated'/'chains'/(outcome.manifest_sha256+'.json')).resolve()
    sidecar = manifest.with_name(outcome.manifest_sha256+'.composition.json')
    temporary = None
    saved_bytes = None
    try:
        selected = {'schema':SIDECAR_SCHEMA, 'manifest_sha256':outcome.manifest_sha256,
                    'composition_sha256':spec_sha, 'composition':spec.to_dict()}
        saved_bytes = _canonical_json(selected, name='saved calibrated composition')
        if len(saved_bytes) > limits.max_metadata_bytes:
            raise ReferenceCalibratedSeedError('Selected composition exceeds its saved metadata cap.')
        with tempfile.NamedTemporaryFile(dir=sidecar.parent, suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(saved_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        # All arbitrary callbacks finish before this full audit. Rehash the
        # source here as well as checking its cheap stamp throughout execution.
        current_source = inputs.source_identity(source_path, poll)
        poll()
        if current_source != source:
            raise ReferenceCalibratedSeedError('Selected original bytes changed before saved capture commit.')
        verified = read_reference_calibrated_seed(store, outcome.manifest_sha256,
            parent=parent, spec=spec, expected_spec_sha256=spec_sha,
            expected_source_sha256=source.sha256, expected_parent_frame=frame,
            output_measurement=output_measurement, limits=limits, cancel=None)
        _verify_preparation(verified, spec, limits)
        if (sha256(_bounded_bytes(manifest, limits)).hexdigest() != outcome.manifest_sha256
                or _bounded_bytes(temporary, limits) != saved_bytes
                or spec.sha256 != spec_sha or _code_identity() != code
                or inputs.stamp(source_path) != selected_stamp):
            raise ReferenceCalibratedSeedError('Saved calibrated revision changed before sidecar publication.')
        if sidecar.exists() and _bounded_bytes(sidecar, limits) != saved_bytes:
            raise ReferenceCalibratedSeedError('A different saved composition occupies this revision address.')
        os.replace(temporary, sidecar)
        temporary = None
    except Exception:
        # A filesystem exception can report an unknown outcome after replacement
        # already took effect. Preserve an exact committed pair in that case.
        committed = False
        try:
            committed = (saved_bytes is not None and sidecar.is_file() and _bounded_bytes(sidecar, limits) == saved_bytes
                and sha256(_bounded_bytes(manifest, limits)).hexdigest() == outcome.manifest_sha256)
        except (OSError, ReferenceCalibratedSeedError):
            pass
        if not committed:
            _retract_owned_manifest(outcome, store, limits)
        raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return ReferenceCalibratedCaptureSession(spec, outcome, verified, manifest, outcome.manifest_sha256)


def open_calibrated_reference_capture(path: Path | str, *, limits=CheckpointLimits(), cancel=None
    ) -> ReferenceCalibratedCaptureSession:
    """Open only the selected saved revision, without reading the original image."""
    _limits(limits)
    _poll(cancel)
    path = Path(path).resolve()
    digest = path.stem
    if (path.suffix != '.json' or not re.fullmatch(r'[0-9a-f]{64}', digest)
            or path.parent.name != 'chains' or path.parent.parent.name != 'calibrated'):
        raise ReferenceCalibratedSeedError('Select a parent manifest from its calibrated chain store.')
    raw = _bounded_bytes(path, limits)
    if sha256(raw).hexdigest() != digest:
        raise ReferenceCalibratedSeedError('Selected calibrated capture differs from its addressed bytes.')
    record = _strict_json_loads(raw)
    side_raw = _bounded_bytes(path.with_name(digest+'.composition.json'), limits)
    side = _strict_json_loads(side_raw)
    if (not isinstance(side, dict) or set(side) != {'schema','manifest_sha256','composition_sha256','composition'}
            or side.get('schema') != SIDECAR_SCHEMA or side.get('manifest_sha256') != digest
            or _canonical_json(side, name='saved calibrated composition') != side_raw):
        raise ReferenceCalibratedSeedError('Saved composition sidecar schema or canonical bytes differ.')
    spec = ReferenceCalibratedSeedSpec.from_dict(side['composition'])
    if side['composition_sha256'] != spec.sha256 or not isinstance(record, dict):
        raise ReferenceCalibratedSeedError('Saved selected composition identity differs.')
    value = record.get('parent_input')
    if not isinstance(value, dict) or set(value) != {'checkpoint_sha256','identity','measurement','array_name','role'}:
        raise ReferenceCalibratedSeedError('Saved parent input binding differs.')
    store = path.parent.parent.parent
    parent = ReferenceRasterInput(store/'inputs', value['checkpoint_sha256'],
        CheckpointIdentity.from_dict(value['identity']), MeasurementContract.from_dict(value['measurement']),
        value['array_name'], RasterRole.from_dict(value['role']))
    frame = CoordinateFrame.from_dict(record['parent_frame'])
    verified = read_reference_calibrated_seed(store, digest, parent=parent, spec=spec,
        expected_spec_sha256=spec.sha256, expected_source_sha256=record['source_sha256'],
        expected_parent_frame=frame, output_measurement=reference_measurement(OUTPUT_MEASUREMENT),
        limits=limits, cancel=cancel)
    _verify_preparation(verified, spec, limits)
    _poll(cancel)
    return ReferenceCalibratedCaptureSession(spec, None, verified, path, digest)


__all__ = ['ReferenceCalibratedCaptureSession','capture_calibrated_reference_path','open_calibrated_reference_capture']
