"""Asynchronous desktop view of calibrated public seed captures."""
from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
import threading
from typing import Any, Mapping

import cv2
import numpy as np
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Qt, Signal
from PySide6.QtWidgets import QFileDialog, QFormLayout, QGroupBox, QLineEdit

from seedvision.benchmarking.checkpoint_inspection import _verify_runner_identity
from seedvision.benchmarking.checkpoints import (
    CheckpointCancelled, CheckpointLimits, CoordinateFrame, MeasurementContract,
    _canonical_json, _thaw_json, load_checkpoint,
)
from seedvision.benchmarking import native_image_inputs as inputs
from seedvision.benchmarking.native_image_inputs import stamp
from seedvision.benchmarking.reference_calibrated_capture_session import (
    ReferenceCalibratedCaptureSession, capture_calibrated_reference_path,
    open_calibrated_reference_capture,
)
from seedvision.benchmarking.reference_calibrated_seed import (
    ADAPTER_CONTRACT, ReferenceCalibratedSeedSpec, _card_parameters, _scale,
)
from seedvision.benchmarking.reference_runner import _strict_json_loads
from .public_reference_capture_dialog import (
    DisplayArtifact, MAX_PREVIEW_PIXELS, MAX_TOTAL_PREVIEW_PIXELS,
    PublicReferenceCaptureDialog, preview_rgb,
)


def _same(actual, expected, name):
    if _canonical_json(_thaw_json(actual),name=name) != _canonical_json(_thaw_json(expected),name=name):
        raise ValueError('Partial checkpoint differs from its current capture '+name+'.')


def _partial_checkpoints(session, store, cancel, limits):
    """Read successful typed references from this current outcome, never a saved partial manifest.

Each independent load verifies content, exact identity, measurement, execution,
source/frame, selected spec and public pins. Failed loads have explicit facts;
other valid phases remain available. Retained numeric bytes use a shared budget.
"""
    outcome = session.outcome
    if outcome is None or outcome.manifest_sha256 is not None:
        raise ValueError('Partial previews require the current unsaved typed outcome.')
    record = outcome.record
    _same(record['composition'],session.spec.to_dict(),'composition')
    _same(record['composition_sha256'],session.spec.sha256,'composition digest')
    if record['complete'] is not False:
        raise ValueError('An incomplete session cannot display a complete parent claim.')
    source = record['source_sha256']
    parent_frame = CoordinateFrame.from_dict(_thaw_json(record['parent_frame']))
    _same(record['parent_input']['identity']['source_sha256'],source,'parent source')
    _same(record['parent_input']['identity']['coordinate_frame'],parent_frame.to_dict(),'parent frame')
    crop = record['crop']
    crop_frame = None if crop is None else CoordinateFrame.from_dict(_thaw_json(crop['coordinate_frame']))
    output_measurement = MeasurementContract.from_dict(_thaw_json(record['output_measurement']))
    if output_measurement.eligible:
        raise ValueError('Public capture measurements remain scientifically ineligible.')
    checkpoints, facts = {}, {}
    retained = 0
    def load(name,root,reference,measurement,frame,*,call=None,parameters=None,composed=False):
        nonlocal retained
        if reference is None:
            return
        if cancel.is_set():
            raise CheckpointCancelled('Partial capture display cancelled.')
        try:
            remaining = limits.max_total_array_bytes-retained
            if remaining < 1:
                raise ValueError('Partial retained numeric budget exhausted.')
            checkpoint = load_checkpoint(root,reference.checkpoint_sha256,
                expected_identity=reference.identity,expected_measurement=measurement,
                limits=replace(limits,max_total_array_bytes=remaining,
                               max_array_bytes=min(limits.max_array_bytes,remaining)))
            if checkpoint.execution != reference.execution or measurement.eligible:
                raise ValueError('Partial execution or measurement differs.')
            if checkpoint.identity.source_sha256 != source or checkpoint.identity.coordinate_frame != frame:
                raise ValueError('Partial source or frame differs.')
            _verify_runner_identity(checkpoint.metadata,session.spec.seed_recipe)
            if composed:
                _same(checkpoint.metadata['composition_sha256'],session.spec.sha256,'selected composition')
                _same(checkpoint.metadata['composition'],session.spec.to_dict(),'composition payload')
                _same(checkpoint.metadata['composition_attempt_id'],record['composition_attempt_id'],'attempt')
                _same(checkpoint.metadata['parent_input'],record['parent_input'],'original parent binding')
            if call is not None:
                _same(checkpoint.metadata['call_id'],call.call_id,'source call')
                _same(checkpoint.metadata['operation'],call.operation,'public operation')
                _same(checkpoint.metadata['source_sha256'],source,'source bytes')
                _same(checkpoint.metadata['coordinate_frame'],frame.to_dict(),'coordinate frame')
                if not composed:
                    _same(checkpoint.metadata['recipe_identity'],session.spec.seed_recipe.identity.to_dict(),'selected child')
            if parameters is not None:
                _same(checkpoint.metadata['parameters'],parameters,'selected preprocessing parameters')
            retained += sum(array.nbytes for array in checkpoint.arrays.values())
            checkpoints[name] = checkpoint
            facts[name] = {'status':'checkpoint_verified','checkpoint_sha256':checkpoint.checkpoint_sha256,
                           'identity':checkpoint.identity.to_dict(),'measurement':checkpoint.measurement.to_dict(),
                           'execution':checkpoint.execution.to_dict()}
        except CheckpointCancelled:
            raise
        except Exception as error:
            facts[name] = {'status':'checkpoint_unavailable','checkpoint_sha256':reference.checkpoint_sha256,
                           'reason':' '.join(f'{type(error).__name__}: {error}'.split())[:512]}
    store = Path(store)
    load('correction',store,outcome.correction.checkpoint_reference,output_measurement,parent_frame,
         call=outcome.correction,composed=True)
    correction = checkpoints.get('correction')
    crop_valid = False
    if crop is not None and correction is not None:
        try:
            x0,y0,x1,y1 = session.spec.seed_roi_xyxy
            _same(crop['bounds_xyxy'],[x0,y0,x1,y1],'seed bounds')
            _same(crop['local_to_parent_xy'],[x0,y0],'crop translation')
            _same(crop['parent_frame'],parent_frame.to_dict(),'crop parent')
            _same(crop['correction_checkpoint_sha256'],correction.checkpoint_sha256,'correction lineage')
            corrected = correction.arrays['corrected_image']
            transform = {'parent_frame':parent_frame.to_dict(),'bounds_xyxy':[x0,y0,x1,y1],
                'corrected_rgb_sha256':sha256(corrected.tobytes(order='C')).hexdigest(),
                'composition_sha256':session.spec.sha256,'gray_conversion':ADAPTER_CONTRACT['gray'],
                'opencv':str(cv2.__version__),'adapter_version':'1'}
            _same(crop['transform'],transform,'crop transform')
            expected_frame = CoordinateFrame('corrected_public_seed_roi',(y1-y0,x1-x0),
                'top_left_pixel_centres','x_right_y_down','pixel',
                sha256(_canonical_json(transform,name='corrected crop frame')).hexdigest())
            if crop_frame != expected_frame:
                raise ValueError('Partial crop frame differs from selected corrected pixels.')
            # Corrected RGB is the exact context required by downstream calls.
            cropped_rgb = np.ascontiguousarray(corrected[y0:y1,x0:x1])
            crop_valid = True
        except Exception as error:
            facts['crop'] = {'status':'checkpoint_unavailable','reason':' '.join(str(error).split())[:512]}
    if outcome.child is not None:
        if crop_frame is None or outcome.child.preprocessing.recipe_identity != session.spec.seed_recipe.identity:
            raise ValueError('Partial child lacks its exact selected crop/recipe.')
        for index,stage in enumerate(outcome.child.preprocessing.stages):
            load(f'seed.preprocessing.{index}',store/'seed',stage.checkpoint_reference,
                 output_measurement,crop_frame,parameters=stage.parameters)
        for name,call in zip(('watershed','size','color'),outcome.child.calls,strict=True):
            load('seed.'+name,store/'seed',call.checkpoint_reference,output_measurement,crop_frame,call=call)
    if outcome.scaled_size.checkpoint_reference is not None:
        if crop_frame is None or record['scaled_measurement'] is None:
            raise ValueError('Partial scaled size lacks its selected crop/measurement.')
        measurement = MeasurementContract.from_dict(_thaw_json(record['scaled_measurement']))
        load('scaled_size',store,outcome.scaled_size.checkpoint_reference,measurement,crop_frame,
             call=outcome.scaled_size,composed=True)
    def drop(name,reason):
        nonlocal retained
        checkpoint = checkpoints.pop(name,None)
        if checkpoint is not None:
            retained -= sum(array.nbytes for array in checkpoint.arrays.values())
            facts[name] = {'status':'checkpoint_unavailable','checkpoint_sha256':checkpoint.checkpoint_sha256,
                           'reason':' '.join(str(reason).split())[:512]}
    watershed = checkpoints.get('seed.watershed')
    final_mask = checkpoints.get('seed.preprocessing.4')
    if watershed is not None:
        try:
            if not crop_valid or final_mask is None:
                raise ValueError('Partial watershed lacks its verified corrected crop/final mask context.')
            _same(watershed.metadata['rgb_input'],crop['inputs']['rgb'],'watershed RGB binding')
            _same(watershed.metadata['input_array_sha256'],
                {'rgb':sha256(cropped_rgb.tobytes(order='C')).hexdigest(),
                 'mask':sha256(final_mask.arrays['mask'].tobytes(order='C')).hexdigest()},'watershed input pixels')
        except Exception as error:
            drop('seed.watershed',error)
            watershed = None
    for name in ('seed.size','seed.color','scaled_size'):
        checkpoint = checkpoints.get(name)
        if checkpoint is None:
            continue
        try:
            if watershed is None or not crop_valid:
                raise ValueError('Downstream partial observation lacks its verified watershed/crop context.')
            labels = watershed.arrays['labels']
            if not np.array_equal(checkpoint.arrays['labels_context'],labels):
                raise ValueError('Partial analysis labels differ from the actual watershed labels.')
            image_key = 'rgb' if name=='seed.color' else 'image'
            _same(checkpoint.metadata['input_array_sha256'],
                {image_key:sha256(cropped_rgb.tobytes(order='C')).hexdigest(),
                 'labels':sha256(labels.tobytes(order='C')).hexdigest()},'analysis input pixels')
            if name=='scaled_size':
                lineage = checkpoint.metadata['lineage']
                _same(lineage['correction_checkpoint_sha256'],correction.checkpoint_sha256,'scaled correction')
                _same(lineage['watershed_checkpoint_sha256'],watershed.checkpoint_sha256,'scaled watershed')
                _same(lineage['corrected_crop_rgb'],crop['inputs']['rgb'],'scaled RGB binding')
                _same(lineage['calibration'],_scale(correction,session.spec),'returned scale')
            else:
                _same(checkpoint.metadata['rgb_input'],crop['inputs']['rgb'],'analysis RGB binding')
        except Exception as error:
            drop(name,error)
    return checkpoints,facts,retained


@dataclass(frozen=True,slots=True)
class CalibratedCaptureRequest:
    generation: int
    spec: ReferenceCalibratedSeedSpec | None
    source_path: Path | None
    store: Path
    source_stamp: tuple | None
    saved_manifest: Path | None = None


@dataclass(frozen=True,slots=True)
class CalibratedCaptureReply:
    session: ReferenceCalibratedCaptureSession
    artifacts: tuple[DisplayArtifact,...]
    checkpoint_facts: Mapping[str,Any]
    source_metadata: Mapping[str,Any]
    compact_arrays: Mapping[str,Any]
    retained_numeric_bytes: int


def _display_reply(session,store,cancel,*,limits=CheckpointLimits(),expected_spec=None,expected_source_sha256=None):
    if expected_spec is not None and session.spec != expected_spec:
        raise ValueError('Capture returned a different frozen selection.')
    record = session.verified.record if session.verified is not None else session.outcome.record
    if expected_source_sha256 is not None and record['source_sha256'] != expected_source_sha256:
        raise ValueError('Capture returned a different frozen original byte identity.')
    if session.verified is not None:
        checkpoints = session.verified.checkpoints
        retained = session.verified.retained_array_bytes
        if retained > limits.max_total_array_bytes:
            raise ValueError('Complete retained numeric display budget exhausted.')
        facts = {name:{'status':'complete_graph_verified','checkpoint_sha256':checkpoint.checkpoint_sha256,
            'identity':checkpoint.identity.to_dict(),'measurement':checkpoint.measurement.to_dict(),
            'execution':checkpoint.execution.to_dict()} for name,checkpoint in checkpoints.items()}
    else:
        checkpoints,facts,retained = _partial_checkpoints(session,store,cancel,limits)
    arrays = [(stage,name,checkpoint,array) for stage,checkpoint in checkpoints.items()
              for name,array in checkpoint.arrays.items()]
    per_array = min(MAX_PREVIEW_PIXELS,MAX_TOTAL_PREVIEW_PIXELS//max(1,len(arrays)))
    artifacts,compact = [],{}
    for stage,name,checkpoint,array in arrays:
        if cancel.is_set():
            raise CheckpointCancelled('Calibrated capture display preparation cancelled.')
        if name in {'coordinates','palette'}:
            display = None
            description = f'Compact actual array {list(array.shape)} {array.dtype.str}; facts include at most4096 elements.'
            if array.size <= 4096:
                compact[stage+'.'+name] = {'original_shape':list(array.shape),'original_elements':int(array.size),
                    'displayed_elements':int(array.size),'truncated':False,'values':array.tolist()}
            else:
                compact[stage+'.'+name] = {'original_shape':list(array.shape),'original_elements':int(array.size),
                    'displayed_elements':0,'truncated':True,'values':[]}
        else:
            display,description = preview_rgb(array,name,max_pixels=per_array)
        if display is not None:
            display.flags.writeable = False
        fallback = ('prepared local RGB input' if stage=='parent' else
                    'explicit crop/conversion of actual corrected image' if stage.startswith('crop.') else
                    'corrected crop input context' if stage.startswith('seed.input.') else
                    'actual public preprocessing output')
        origin = checkpoint.metadata.get('array_origins',{}).get(name,fallback)
        artifacts.append(DisplayArtifact(stage+'.'+name,sha256(array.tobytes(order='C')).hexdigest(),
                                         origin,display,description))
    metadata = {name:_thaw_json(checkpoint.metadata) for name,checkpoint in checkpoints.items()}
    if cancel.is_set():
        raise CheckpointCancelled('Calibrated capture display preparation cancelled.')
    # Successful partial arrays are released with this function's locals;
    # only bounded display copies, compact facts and metadata cross into Qt.
    return CalibratedCaptureReply(session,tuple(artifacts),facts,metadata,compact,retained)


class _CalibratedCaptureSignals(QObject):
    completed = Signal(int,object,object)


class _CalibratedCaptureWorker(QRunnable):
    def __init__(self,request,cancel,runner_factory):
        super().__init__()
        self.request,self.cancel,self.runner_factory = request,cancel,runner_factory
        self.signals = _CalibratedCaptureSignals()

    def run(self):
        try:
            request = self.request
            if request.saved_manifest is not None:
                session = open_calibrated_reference_capture(request.saved_manifest,cancel=self.cancel)
                expected_source = expected_spec = None
            else:
                def guard(**_kwargs):
                    if self.cancel.is_set():
                        raise CheckpointCancelled('Calibrated capture request cancelled.')
                    if stamp(request.source_path) != request.source_stamp:
                        raise ValueError('Source image changed before its frozen request was captured.')
                # Byte hashing occurs off the GUI thread. It adds no decode or
                # alternate raster; the session must return this exact source.
                expected_source = inputs.source_identity(request.source_path,guard).sha256
                expected_spec = request.spec
                session = capture_calibrated_reference_path(request.spec,request.source_path,request.store,
                    self.runner_factory(),expected_stamp=request.source_stamp,cancel=self.cancel)
            reply,error = _display_reply(session,request.store,self.cancel,
                expected_spec=expected_spec,expected_source_sha256=expected_source),None
        except Exception as failure:
            reply,error = None,failure
        self.signals.completed.emit(self.request.generation,reply,error)


class CalibratedPublicReferenceCaptureDialog(PublicReferenceCaptureDialog):
    """A separate public session; original/native graph records are unchanged."""
    def __init__(self,recipe,parent=None,*,source_path=None,store=None,runner_factory=None,
                 card_parameters=None,seed_roi_xyxy=None):
        self._card_values = _card_parameters({} if card_parameters is None else card_parameters)
        self._seed_roi_values = seed_roi_xyxy
        self._card_editors = {}
        super().__init__(recipe,parent,source_path=source_path,store=store,runner_factory=runner_factory)
        self.setWindowTitle('Calibrated public PlantCV seed capture')
        self.layout().itemAt(0).widget().setText(
            'Correct the full image using a color card, crop the selected seed region, and run the pinned '
            'public seed and scaled-size calls. Card bounds and seed bounds are separate. '
            'Returned scale and traits are source observations; biological validity is not certified.')
        self.store_button.setText('Capture folder…')
        self.run_button.setText('Run capture')
        self.open_button.setText('Open saved…')
        self.parameter_scroll.setMinimumWidth(360)
        self.status_label.setText('Select an image, card controls and explicit seed bounds [x0,y0,x1,y1].')

    def _build_parameters(self):
        super()._build_parameters()
        self._building = True
        body = self.parameter_scroll.widget()
        group = QGroupBox('Public color-card correction',body)
        form = QFormLayout(group)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        labels = {'color_chip_size':'Chip size (mm)',
                  'adaptive_method':'Adaptive method\n0 mean; 1 Gaussian',
                  'block_size':'Odd window size', 'radius':'Chip sampling radius',
                  'min_size':'Minimum card area', 'aspect_ratio':'Card aspect ratio',
                  'solidity':'Minimum card solidity', 'roi_bounds_xyxy':'Card detection bounds'}
        self._card_editors = {}
        for name,value in self._card_values.items():
            editor = QLineEdit(json.dumps(value,ensure_ascii=False),group)
            editor.setObjectName('publicCardParameter_'+name)
            editor.setToolTip('Selected value is passed to actual public correction. '
                'Bounds are half-open [x0,y0,x1,y1]; null uses the full image. '
                'Chip size accepts [width,height], "classic", "passport", "cameratrax", or null.')
            editor.textChanged.connect(self._invalidate)
            self._card_editors[name] = editor
            form.addRow(labels[name],editor)
        seed_group = QGroupBox('Seed region after correction',body)
        seed_form = QFormLayout(seed_group)
        self.seed_roi_editor = QLineEdit(json.dumps(self._seed_roi_values),seed_group)
        self.seed_roi_editor.setObjectName('publicSeedBounds')
        self.seed_roi_editor.setPlaceholderText('[x0,y0,x1,y1]')
        self.seed_roi_editor.setToolTip('Required integer half-open bounds inside the decoded image. '
            'Coordinates in seed results stay local to this crop.')
        self.seed_roi_editor.textChanged.connect(self._invalidate)
        seed_form.addRow('Seed crop bounds',self.seed_roi_editor)
        body.layout().insertWidget(0,group)
        body.layout().insertWidget(1,seed_group)
        self._building = False

    def selected_spec(self):
        parameters = {name:_strict_json_loads(editor.text().encode('utf-8'))
                      for name,editor in self._card_editors.items()}
        bounds = _strict_json_loads(self.seed_roi_editor.text().encode('utf-8'))
        if bounds is None:
            raise ValueError('Enter explicit seed crop bounds [x0,y0,x1,y1].')
        return ReferenceCalibratedSeedSpec(self.selected_recipe(),parameters,bounds)

    def start_capture(self):
        if self._worker is not None or self._closing:
            return
        try:
            spec = self.selected_spec()
            source_stamp = stamp(self._source_path) if self._source_path is not None else None
            if source_stamp is None:
                raise ValueError('Select an existing source image.')
        except Exception as error:
            self.status_label.setText(f'Cannot start: {error}')
            return
        self._invalidate()
        self._start(CalibratedCaptureRequest(self._generation,spec,self._source_path,self._store,source_stamp))

    def open_saved(self,path):
        if self._worker is not None or self._closing:
            return
        self._invalidate()
        self._start(CalibratedCaptureRequest(self._generation,None,None,self._store,None,Path(path)))

    def _choose_saved(self):
        path,_ = QFileDialog.getOpenFileName(self,'Open stored calibrated public capture',
            str(self._store/'calibrated'/'chains'),'Capture manifests (*.json)')
        if path:
            self.open_saved(path)

    def _start(self,request):
        self._cancel = threading.Event()
        self._worker = _CalibratedCaptureWorker(request,self._cancel,self._runner_factory)
        self._worker.signals.completed.connect(self._completed,Qt.ConnectionType.QueuedConnection)
        self._busy(True)
        self.status_label.setText('Verifying saved calibrated capture…' if request.saved_manifest
                                  else 'Executing public correction, seed and scaled-size calls…')
        QThreadPool.globalInstance().start(self._worker)

    def _completed(self,generation,reply,error):
        worker = self._worker
        if worker is None:
            return
        request = worker.request
        self._worker,self._cancel = None,None
        self._busy(False)
        if self._closing:
            self.reject()
            return
        if generation != self._generation:
            self.status_label.setText('Previous request discarded after cancellation or changed selections.')
            return
        if request.source_path is not None and stamp(request.source_path) != request.source_stamp:
            self._invalidate()
            self.status_label.setText('Source image changed; any saved capture belongs to its earlier immutable snapshot.')
            return
        if error is not None:
            self.status_label.setText(f'Capture unavailable: {type(error).__name__}: {error}')
            return
        session = reply.session
        if request.saved_manifest is not None:
            self._recipe = session.spec.seed_recipe
            self._card_values = _thaw_json(session.spec.card_parameters)
            self._seed_roi_values = list(session.spec.seed_roi_xyxy)
            # This address came from the fully verified session opener; it is
            # not inferred from an unchecked path or the current image.
            self._source_path,self._store = None,session.manifest_path.parent.parent.parent
            self._build_parameters()
            self._refresh_bindings()
        self._reply = reply
        if session.verified is None:
            self.status_label.setText(f'Public run {session.outcome.status}; successful individual checkpoints only. '
                                      'No complete capture manifest.')
            record = _thaw_json(session.outcome.record)
        else:
            self.status_label.setText(f'Verified complete calibrated capture: {session.manifest_sha256}')
            record = _thaw_json(session.verified.record)
        facts = {'run':record,'checkpoint_display_validation':dict(reply.checkpoint_facts),
            'actual_source_metadata':dict(reply.source_metadata),'compact_actual_arrays':dict(reply.compact_arrays),
            'artifact_raw_sha256':{artifact.key:artifact.raw_sha256 for artifact in reply.artifacts},
            'retained_numeric_bytes':reply.retained_numeric_bytes,
            'retained_scope':'Successful loaded checkpoint arrays; excludes preview copies, metadata and scratch.'}
        self.facts.setPlainText(json.dumps(facts,ensure_ascii=False,indent=2,allow_nan=False))
        for artifact in reply.artifacts:
            self.artifact_selector.addItem(artifact.key)
        self._show_artifact(self.artifact_selector.currentIndex())


__all__ = ['CalibratedPublicReferenceCaptureDialog']
