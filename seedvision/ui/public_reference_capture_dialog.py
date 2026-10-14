"""An isolated, asynchronous desktop view of actual public seed-chain calls."""
from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
import logging
import math
from pathlib import Path
import threading

import numpy as np
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QScrollArea, QSplitter,
    QSizePolicy, QTextBrowser, QVBoxLayout, QWidget, QInputDialog,
)

from seedvision.benchmarking.checkpoints import CheckpointCancelled, _thaw_json
from seedvision.benchmarking.native_image_inputs import stamp
from seedvision.benchmarking.recipes import RecipeSpec
from seedvision.benchmarking.reference_capture_session import (
    ReferenceCaptureSession, capture_reference_path, open_reference_capture,
)
from seedvision.benchmarking.reference_preprocessing import _chain_stages, _validate_pinned_recipe
from seedvision.benchmarking.reference_mask_stage import _selected_parameters
from seedvision.benchmarking.reference_stage import _calculation_parameters
from seedvision.benchmarking.reference_seed_chain import _parameters
from seedvision.benchmarking.reference_runner import PlantCVReferenceRunner, ReferenceCancelledError, _strict_json_loads


ROOT = Path(__file__).resolve().parents[2]
MAX_PREVIEW_PIXELS = 1_048_576
MAX_TOTAL_PREVIEW_PIXELS = 4_194_304
_LOG = logging.getLogger(__name__)


class _CapturePreview(QLabel):
    """Fit the bounded display image to the available widget, without cropping."""

    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self._source_pixmap = None
        self._fitted_size = None
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)

    def setPixmap(self, pixmap):
        self._source_pixmap = QPixmap(pixmap)
        self._fitted_size = None
        self._fit()

    def clear(self):
        self._source_pixmap = None
        self._fitted_size = None
        super().clear()

    def _fit(self):
        target = self.contentsRect().size()
        if self._source_pixmap is None or target.isEmpty() or target == self._fitted_size:
            return
        self._fitted_size = target
        super().setPixmap(self._source_pixmap.scaled(target, Qt.AspectRatioMode.KeepAspectRatio,
                                                   Qt.TransformationMode.FastTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit()


def preview_rgb(array: np.ndarray, name: str, *, max_pixels=MAX_PREVIEW_PIXELS):
    """Bound first, then transform only sampled pixels for Qt display."""
    if array.ndim not in (2, 3) or not array.shape[0] or not array.shape[1]:
        return None, "Compact metadata; no raster preview."
    step = max(1, math.ceil(math.sqrt(array.shape[0] * array.shape[1] / max_pixels)))
    while math.ceil(array.shape[0] / step) * math.ceil(array.shape[1] / step) > max_pixels:
        step += 1
    sampled = array[::step, ::step]
    description = f"Display samples every {step} pixel(s); original {list(array.shape)} {array.dtype.str}."
    if sampled.ndim == 3:
        if sampled.dtype != np.uint8 or sampled.shape[2] != 3:
            return None, "Unsupported display channels; inspect the actual array metadata."
        if name == "diagnostic_bgr":
            sampled = sampled[..., ::-1]
            description += " Actual BGR capture adapted to RGB for Qt."
        else:
            description += " RGB display."
        rgb = np.ascontiguousarray(sampled)
    elif sampled.dtype == np.bool_:
        rgb = np.repeat((sampled.astype(np.uint8) * 255)[..., None], 3, axis=2)
        description += " Boolean display: false=0, true=255."
    elif name in {"labels", "markers", "labels_context"}:
        values = sampled.astype(np.uint64)
        rgb = np.stack(((values * 67) % 256, (values * 137) % 256, (values * 211) % 256), axis=-1).astype(np.uint8)
        rgb[values == 0] = 0
        description += " Display colours encode existing IDs; zero is black."
    elif sampled.dtype == np.uint8:
        rgb = np.repeat(sampled[..., None], 3, axis=2)
        description += " Gray display."
    else:
        values = sampled.astype(np.float64)
        finite = np.isfinite(values)
        low, high = (float(values[finite].min()), float(values[finite].max())) if finite.any() else (0.0, 0.0)
        gray = np.zeros(values.shape, dtype=np.uint8)
        if high > low:
            gray[finite] = np.rint((values[finite] - low) * (255 / (high - low))).astype(np.uint8)
        rgb = np.repeat(gray[..., None], 3, axis=2)
        description += f" Sampled finite range [{low:g}, {high:g}] mapped to gray for display."
    return rgb, description


@dataclass(frozen=True, slots=True)
class CaptureRequest:
    generation: int
    recipe: RecipeSpec
    source_path: Path | None
    store: Path
    source_stamp: tuple | None
    saved_manifest: Path | None = None
    expected_source_sha256: str | None = None
    history_root: Path | None = None
    project_path: Path | None = None


@dataclass(frozen=True, slots=True)
class DisplayArtifact:
    key: str
    raw_sha256: str
    origin: str
    image: np.ndarray | None
    description: str


@dataclass(frozen=True, slots=True)
class CaptureReply:
    session: ReferenceCaptureSession
    artifacts: tuple[DisplayArtifact, ...]


def _display_artifacts(session, cancel):
    if session.verified is None:
        return ()
    arrays = [(stage, name, checkpoint, array)
              for stage, checkpoint in session.verified.checkpoints.items()
              for name, array in checkpoint.arrays.items()]
    per_array = min(MAX_PREVIEW_PIXELS, MAX_TOTAL_PREVIEW_PIXELS // max(1, len(arrays)))
    artifacts = []
    for stage, name, checkpoint, array in arrays:
        if cancel.is_set():
            raise CheckpointCancelled("Capture display preparation cancelled.")
        if name in {"coordinates", "palette"}:
            display, description = None, f"Compact actual array {list(array.shape)} {array.dtype.str}; facts show values when at most 4096 elements."
        else:
            display, description = preview_rgb(array, name, max_pixels=per_array)
        if display is not None:
            display.flags.writeable = False
        origin = checkpoint.metadata.get("array_origins", {}).get(name,
            "prepared caller input" if stage.startswith("input.") else "actual public preprocessing output")
        artifacts.append(DisplayArtifact(f"{stage}.{name}", sha256(array.tobytes(order="C")).hexdigest(),
                                         origin, display, description))
    return tuple(artifacts)


class _CaptureSignals(QObject):
    completed = Signal(int, object, object)


class _CaptureWorker(QRunnable):
    def __init__(self, request, cancel, runner_factory):
        super().__init__()
        self.request, self.cancel, self.runner_factory = request, cancel, runner_factory
        self.signals = _CaptureSignals()
        self.attempt = None
        self.execution_quality = None
        self.qualities = []
        self.retention_errors = []
        self.invoked = False
        self.source_unchanged = None

    def retain(self, quality, original=None):
        """Retain immutable history even when its UI delivery is discarded."""
        from seedvision.persistence.method_qc_history import retain_method_qc
        self.qualities.append(quality)
        try:
            retain_method_qc(self.request.history_root or ROOT, quality,
                             project_path=self.request.project_path)
        except Exception as error:
            message = f'QC history retention failed: {type(error).__name__}: {error}'
            self.retention_errors.append(message)
            _LOG.warning('%s', message)
            if original is not None:
                original.add_note(message)

    def record_delivery(self, state, detail, original=None):
        from seedvision.export.public_reference_attempt import delivery_quality
        if self.execution_quality is None:
            return
        recipe = RecipeSpec.from_json(json.dumps(self.execution_quality.payload['recipe']))
        quality = delivery_quality(recipe, self.execution_quality, self.attempt,
            state=state, detail=detail, invoked=self.invoked)
        self.retain(quality, original)

    def run(self):
        from seedvision.export.public_reference_attempt import (capture_public_reference_attempt,
            source_matches_attempt, execution_quality, terminal_quality)
        phase = 'admission'
        try:
            request = self.request
            if request.saved_manifest is not None:
                session = open_reference_capture(request.saved_manifest, cancel=self.cancel)
            else:
                self.attempt = capture_public_reference_attempt(request.recipe, request.source_path,
                    expected_source_sha256=request.expected_source_sha256)
                if self.cancel.is_set():
                    raise CheckpointCancelled('Public capture cancelled before wrapper invocation.')
                runner = self.runner_factory()
                phase = 'capture'
                self.invoked = True
                session = capture_reference_path(request.recipe, request.source_path, request.store,
                    runner, expected_stamp=request.source_stamp, cancel=self.cancel)
                phase = 'verification'
                if not isinstance(session, ReferenceCaptureSession):
                    self.source_unchanged = source_matches_attempt(self.attempt)
                    if not self.source_unchanged:
                        raise ValueError('Source image changed or became unavailable after capture.')
                    raise ValueError('Returned capture is not a verified session.')
                if session.recipe != request.recipe:
                    raise ValueError('Returned capture differs from the selected immutable recipe.')
                record = session.verified.record if session.verified is not None else session.outcome.record
                if (self.attempt.expected_source_sha256 is not None
                        and record['source_sha256'] != self.attempt.expected_source_sha256):
                    raise ValueError('Returned capture differs from the selected source bytes.')
            self.execution_quality = execution_quality(self.attempt, session)
            self.retain(self.execution_quality)
            phase = 'verification'
            if self.attempt is not None:
                self.source_unchanged = source_matches_attempt(self.attempt)
                if not self.source_unchanged:
                    raise ValueError('Source image changed or became unavailable after capture.')
            artifacts = _display_artifacts(session, self.cancel)
            if self.attempt is not None:
                self.source_unchanged = source_matches_attempt(self.attempt)
                if not self.source_unchanged:
                    raise ValueError('Source image changed or became unavailable during display preparation.')
            reply, error = CaptureReply(session, artifacts), None
        except Exception as failure:
            reply, error = None, failure
            state = 'cancelled' if isinstance(failure, (CheckpointCancelled, ReferenceCancelledError)) or self.cancel.is_set() else 'failed'
            try:
                if self.execution_quality is not None:
                    self.record_delivery(state, f'{type(failure).__name__}: {failure}', failure)
                elif self.attempt is not None:
                    self.retain(terminal_quality(self.attempt, state=state, phase=phase,
                        detail=str(failure), invoked=self.invoked, exception=failure), failure)
            except Exception as secondary:
                message = f'QC recording failed: {type(secondary).__name__}: {secondary}'
                self.retention_errors.append(message)
                _LOG.warning('%s', message)
                failure.add_note(message)
        self.signals.completed.emit(self.request.generation, reply, error)


class PublicReferenceCaptureDialog(QDialog):
    """Separate source-call session; it never writes historical or native graphs."""

    def __init__(self, recipe: RecipeSpec, parent=None, *, source_path=None, store=None, runner_factory=None,
                 expected_source_sha256=None, project_root=None, project_path=None):
        _validate_pinned_recipe(recipe)
        super().__init__(parent)
        self.setWindowTitle("Public PlantCV seed capture")
        self.resize(1100, 780)
        self._recipe = recipe
        self._source_path = Path(source_path) if source_path is not None else None
        self._expected_source_sha256 = expected_source_sha256
        self._project_root = project_root or getattr(parent, '_project_root', None) or ROOT
        self._project_path = project_path or getattr(parent, '_project_path', None)
        self._quality = None
        self._quality_history = []
        self._store = Path(store) if store is not None else ROOT / "artifacts" / "public-reference-captures"
        self._runner_factory = runner_factory or (lambda: PlantCVReferenceRunner(
            ROOT / "artifacts" / "plantcv-reference-env" / "Scripts" / "python.exe"))
        self._generation = 0
        self._worker = None
        self._cancel = None
        self._closing = False
        self._building = False
        self._reply = None
        self._editors: dict[tuple[str, str], QWidget] = {}
        layout = QVBoxLayout(self)
        intro = QLabel("Execute the pinned public preprocessing, watershed, size and color calls. "
            "Watershed locals belong to one call. Local decoding prepares RGB and gray inputs. "
            "Public execution does not certify biological measurements.", self)
        intro.setWordWrap(True)
        layout.addWidget(intro)
        row = QHBoxLayout()
        self.image_button = QPushButton("Select image…", self)
        self.store_button = QPushButton("Select capture folder…", self)
        self.run_button = QPushButton("Run public capture", self)
        self.stop_button = QPushButton("Stop", self)
        self.open_button = QPushButton("Open captured run…", self)
        self.stop_button.setEnabled(False)
        for button in (self.image_button, self.store_button, self.run_button, self.stop_button, self.open_button):
            row.addWidget(button)
        layout.addLayout(row)
        self.binding_label = QLabel(self)
        self.binding_label.setWordWrap(True)
        self.binding_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.binding_label.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.binding_label)
        self.status_label = QLabel("Select calculation values and an image.", self)
        self.status_label.setWordWrap(True)
        self.status_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.status_label.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.status_label)
        splitter = QSplitter(self)
        self.parameter_scroll = QScrollArea(self)
        self.parameter_scroll.setWidgetResizable(True)
        self.parameter_scroll.setMinimumWidth(300)
        splitter.addWidget(self.parameter_scroll)
        view = QWidget(self)
        view_layout = QVBoxLayout(view)
        self.artifact_selector = QComboBox(view)
        view_layout.addWidget(self.artifact_selector)
        self.preview = _CapturePreview("No verified capture", view)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(240, 180)
        view_layout.addWidget(self.preview, 2)
        self.preview_description = QLabel(view)
        self.preview_description.setWordWrap(True)
        self.preview_description.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.preview_description.setTextFormat(Qt.TextFormat.PlainText)
        view_layout.addWidget(self.preview_description)
        self.facts = QTextBrowser(view)
        view_layout.addWidget(self.facts, 1)
        splitter.addWidget(view)
        layout.addWidget(splitter, 1)
        close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        self.qc_button = QPushButton('Review recorded capture QC…', self)
        self.qc_button.setEnabled(False)
        self.qc_button.clicked.connect(self._show_quality)
        close.addButton(self.qc_button, QDialogButtonBox.ButtonRole.ActionRole)
        close.rejected.connect(self.reject)
        layout.addWidget(close)
        self.image_button.clicked.connect(self._choose_image)
        self.store_button.clicked.connect(self._choose_store)
        self.run_button.clicked.connect(self.start_capture)
        self.stop_button.clicked.connect(self.stop_capture)
        self.open_button.clicked.connect(self._choose_saved)
        self.artifact_selector.currentIndexChanged.connect(self._show_artifact)
        self._build_parameters()
        self._refresh_bindings()

    def _build_parameters(self):
        self._building = True
        body, self._editors = QWidget(self), {}
        layout = QVBoxLayout(body)
        for stage in self._recipe.stages:
            parameters = [p for p in stage.parameters if p.affects_calculation]
            if not parameters:
                continue
            group = QGroupBox(stage.operation_id, body)
            form = QFormLayout(group)
            for parameter in parameters:
                if parameter.choices:
                    editor = QComboBox(group)
                    for choice in parameter.choices:
                        editor.addItem(json.dumps(_thaw_json(choice), ensure_ascii=False), _thaw_json(choice))
                    for index in range(editor.count()):
                        if editor.itemData(index) == _thaw_json(parameter.value):
                            editor.setCurrentIndex(index)
                            break
                    editor.currentIndexChanged.connect(self._invalidate)
                else:
                    editor = QLineEdit(json.dumps(_thaw_json(parameter.value), ensure_ascii=False), group)
                    editor.textChanged.connect(self._invalidate)
                editor.setObjectName(f"publicParameter_{stage.stage_id}_{parameter.parameter_id}")
                editor.setToolTip(parameter.description + (" Null uses int(public_labels.max())." if parameter.parameter_id == "n_labels" else ""))
                self._editors[stage.stage_id, parameter.parameter_id] = editor
                form.addRow(parameter.parameter_id, editor)
            layout.addWidget(group)
        layout.addStretch(1)
        old = self.parameter_scroll.takeWidget()
        if old is not None:
            old.deleteLater()
        self.parameter_scroll.setWidget(body)
        self._building = False

    def selected_recipe(self):
        stages = []
        for stage in self._recipe.stages:
            parameters = []
            for parameter in stage.parameters:
                editor = self._editors.get((stage.stage_id, parameter.parameter_id))
                if isinstance(editor, QComboBox):
                    value = editor.currentData()
                elif isinstance(editor, QLineEdit):
                    value = _strict_json_loads(editor.text().encode("utf-8"))
                else:
                    parameters.append(parameter)
                    continue
                parameters.append(replace(parameter, value=value))
            stages.append(replace(stage, parameters=tuple(parameters)))
        recipe = replace(self._recipe, stages=tuple(stages))
        _validate_pinned_recipe(recipe)
        for stage in _chain_stages(recipe):
            if stage.operation_id.startswith("threshold."):
                _calculation_parameters(recipe, stage.stage_id, stage.operation_id)
            else:
                _selected_parameters(recipe, stage.stage_id, stage.operation_id)
        _parameters(recipe)
        return recipe

    def _refresh_bindings(self):
        self.binding_label.setText(f"Image: {self._source_path or 'not selected'}\nCapture folder: {self._store}")

    def _invalidate(self, *_):
        if self._building:
            return
        self._generation += 1
        if self._cancel is not None:
            self._cancel.set()
        self._reply = None
        self.qc_button.setEnabled(self._quality is not None)
        self.artifact_selector.clear()
        self.preview.clear()
        self.preview.setText("No current verified capture")
        self.preview_description.clear()
        self.facts.clear()

    def set_source_path(self, path):
        self._invalidate()
        self._source_path = Path(path) if path is not None else None
        self._expected_source_sha256 = None
        self._refresh_bindings()

    def _choose_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select reference input image", str(self._source_path or ROOT / "images"))
        if path:
            self.set_source_path(path)

    def _choose_store(self):
        path = QFileDialog.getExistingDirectory(self, "Select public capture folder", str(self._store))
        if path:
            self._invalidate()
            self._store = Path(path)
            self._refresh_bindings()

    def _choose_saved(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open stored public capture", str(self._store / "chains"), "Capture manifests (*.json)")
        if path:
            self.open_saved(path)

    def start_capture(self):
        if self._worker is not None or self._closing:
            return
        try:
            recipe = self.selected_recipe()
            if self._source_path is None:
                raise ValueError("Select a source image.")
        except Exception as error:
            self.status_label.setText(f"Cannot start: {error}")
            return
        self._invalidate()
        request = CaptureRequest(self._generation, recipe, self._source_path, self._store, stamp(self._source_path),
            expected_source_sha256=self._expected_source_sha256, history_root=self._project_root,
            project_path=self._project_path)
        self._start(request)

    def open_saved(self, path):
        if self._worker is not None or self._closing:
            return
        self._invalidate()
        self._start(CaptureRequest(self._generation, self._recipe, None, self._store, None, Path(path),
            history_root=self._project_root, project_path=self._project_path))

    def _busy(self, busy):
        for control in (self.image_button, self.store_button, self.run_button, self.open_button, self.parameter_scroll):
            control.setEnabled(not busy)
        self.stop_button.setEnabled(busy)

    def _start(self, request):
        self._cancel = threading.Event()
        self._worker = _CaptureWorker(request, self._cancel, self._runner_factory)
        owned = self._worker
        self._worker.signals.completed.connect(
            lambda generation, reply, error: self._completed(generation, reply, error, worker=owned),
            Qt.ConnectionType.QueuedConnection)
        self._busy(True)
        self.status_label.setText("Verifying saved capture…" if request.saved_manifest else "Executing public seed calls…")
        QThreadPool.globalInstance().start(self._worker)

    def stop_capture(self):
        self._invalidate()
        self.status_label.setText("Cancellation requested; waiting for the current worker to drain.")

    def _completed(self, generation, reply, error, *, worker=None):
        worker = self._worker if worker is None else worker
        if worker is None:
            return
        self._quality_history.extend(worker.qualities)
        if worker is not self._worker:
            if reply is not None:
                try:
                    worker.record_delivery('cancelled', 'Stale worker delivery; replacement worker preserved')
                    self._quality_history.extend(worker.qualities[-1:])
                except Exception as secondary:
                    worker.retention_errors.append(f'QC recording failed: {type(secondary).__name__}: {secondary}')
            return
        request = worker.request
        if worker.qualities:
            self._quality = worker.qualities[-1]
            self.qc_button.setEnabled(True)
        self._worker, self._cancel = None, None
        self._busy(False)
        if self._closing or generation != self._generation or worker.cancel.is_set():
            if reply is not None:
                try:
                    worker.record_delivery('cancelled', 'Closing, Stop, or changed selection before Qt installation')
                    self._quality_history.extend(worker.qualities[-1:])
                    self._quality = worker.qualities[-1]
                    self.qc_button.setEnabled(True)
                except Exception as secondary:
                    worker.retention_errors.append(f'QC recording failed: {type(secondary).__name__}: {secondary}')
            if self._closing:
                super().reject()
                return
            notes = '\n'.join(worker.retention_errors)
            self.status_label.setText("Previous request discarded after cancellation or changed selections."
                                     + ('\n'+notes if notes else ''))
            return
        if error is not None:
            notes = '\n'.join(worker.retention_errors)
            self.status_label.setText(f"Capture unavailable: {type(error).__name__}: {error}" + ('\n'+notes if notes else ''))
            return
        self._reply = reply
        self.qc_button.setEnabled(True)
        session = reply.session
        if request.saved_manifest is not None:
            self._recipe = session.recipe
            self._source_path, self._store = None, request.saved_manifest.resolve().parent.parent
            self._build_parameters()
            self._refresh_bindings()
        if session.verified is None:
            self.status_label.setText(f"Public run {session.outcome.status}; no complete capture manifest.")
            record = _thaw_json(session.outcome.record)
        else:
            self.status_label.setText(f"Verified complete capture: {session.verified.manifest_sha256}")
            record = _thaw_json(session.verified.record)
        if worker.retention_errors:
            self.status_label.setText(self.status_label.text()+'\n'+'\n'.join(worker.retention_errors))
        facts = {"run": record, "artifact_raw_sha256": {a.key: a.raw_sha256 for a in reply.artifacts}}
        if session.verified is not None:
            facts["actual_source_metadata"] = {name: _thaw_json(checkpoint.metadata)
                for name, checkpoint in session.verified.checkpoints.items() if not name.startswith("input.")}
            facts["compact_actual_arrays"] = {f"{name}.{key}": array.tolist()
                for name, checkpoint in session.verified.checkpoints.items()
                for key, array in checkpoint.arrays.items() if key in {"coordinates", "palette"} and array.size <= 4096}
        self.facts.setPlainText(json.dumps(facts, ensure_ascii=False, indent=2, allow_nan=False))
        for artifact in reply.artifacts:
            self.artifact_selector.addItem(artifact.key)
        self._show_artifact(self.artifact_selector.currentIndex())

    def _show_quality(self):
        if self._quality is None:
            return
        from seedvision.ui.method_qc_dialog import MethodQCDialog
        records = list({quality.payload_json: quality for quality in self._quality_history}.values())
        quality = self._quality
        if len(records) > 1:
            labels = [f'{index}: {item.payload["identity"]["scope"]} · {item.revision}'
                      for index, item in enumerate(records)]
            choice, accepted = QInputDialog.getItem(self, 'Recorded capture and delivery QC',
                'Select exact historical record:', labels, len(labels)-1, False)
            if not accepted:
                return
            quality = records[labels.index(choice)]
        MethodQCDialog(quality, self, project_root=self._project_root,
                       project_path=self._project_path).exec()

    def _show_artifact(self, index):
        self.preview.clear()
        if self._reply is None or index < 0 or index >= len(self._reply.artifacts):
            return
        artifact = self._reply.artifacts[index]
        self.preview_description.setText(f"{artifact.origin}\nRaw pixels SHA-256: {artifact.raw_sha256}\n{artifact.description}")
        if artifact.image is None:
            self.preview.setText("Compact actual array; inspect facts.")
        else:
            rgb = artifact.image
            image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888).copy()
            self.preview.setPixmap(QPixmap.fromImage(image))

    def reject(self):
        if self._worker is not None:
            self._closing = True
            self.stop_capture()
        else:
            super().reject()

    def closeEvent(self, event):
        if self._worker is not None:
            self.reject()
            event.ignore()
        else:
            super().closeEvent(event)


__all__ = ["PublicReferenceCaptureDialog"]
