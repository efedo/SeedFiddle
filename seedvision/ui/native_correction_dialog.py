"""Four-file native Qt runner for PlantCV's isolated colour-card correction."""
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
import threading
from types import MappingProxyType
from typing import Mapping

import cv2
import numpy as np
import torch
from PySide6.QtCore import QThreadPool, Qt
from PySide6.QtWidgets import QFileDialog, QGridLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from seedvision.benchmarking.checkpoints import CoordinateFrame
from seedvision.benchmarking.native_correction_recipe import (
    MAX_COMBINED_PIXELS, NativeCorrectionCancelled, NativeCorrectionError, NativeCorrectionResult,
    RECIPE_ID, STAGE_ID, build_native_correction_recipe,
)
from seedvision.benchmarking.native_recipe import _tensor_signature, NativeRecipeCancelled
from seedvision.benchmarking.project_correction import run_project_correction
from seedvision.persistence import correction_inputs as assets
from seedvision.persistence.correction_session import (
    CorrectionProjectContext, CorrectionKeepRequest, validate_correction_keep_result,
)
from seedvision.benchmarking.recipes import RecipeSpec
from seedvision.segmentation.baseline import PipelineAnalysisCache
from seedvision.ui.native_recipe_dialog import NativeRecipeDialog, _RunWorker
from seedvision.export.recipe_attempt_qc import (
    capture_recipe_attempt_request, CORRECTION_FILE_EXECUTOR, PROJECT_CORRECTION_EXECUTOR,
)


_PORTS = ("target_rgb", "target_mask", "source_rgb", "source_mask")
_TITLES = {"target_rgb": "Target RGB image", "target_mask": "Target chip-label mask",
    "source_rgb": "Source RGB image", "source_mask": "Source chip-label mask"}
_CORRECTION_CACHE_KEY = f"reference.recipe:{RECIPE_ID}:{STAGE_ID}"
_HASH_CHUNK_BYTES = 1024*1024
# One explicitly capped prepared bundle; path changes release it.
MAX_PREPARED_CUDA_BYTES = 4*MAX_COMBINED_PIXELS


# Retain the desktop preparation API and its explicit test-time bounds.
from seedvision.benchmarking import native_correction_inputs as _inputs
from seedvision.benchmarking.native_correction_inputs import (
    NativeCorrectionInputs, _cancel, _file_hashes, _signatures,
)


def prepare_native_correction_inputs(paths, *, previous=None, device="cuda", cancel=None):
    return _inputs.prepare_native_correction_inputs(paths, previous=previous,
        device=device, cancel=cancel, max_combined_pixels=MAX_COMBINED_PIXELS,
        max_prepared_cuda_bytes=MAX_PREPARED_CUDA_BYTES)


@dataclass(frozen=True, slots=True)
class _Completion:
    result: object
    prepared: NativeCorrectionInputs | None
    generation: int | None = None


class _CorrectionRunWorker(_RunWorker):
    def __init__(self, recipe, paths, cache, previous, cancel, *, generation=None, context=None, binding=None):
        super().__init__(recipe, None, cache, cancel)
        self.paths = dict(paths)
        self.previous, self.cancel = previous, cancel
        self.generation, self.context, self.binding = generation, context, binding
        self.invoked_executor = None

    def _invalidate_correction(self):
        self.cache.values.pop(_CORRECTION_CACHE_KEY, None)

    def run(self):
        phase = 'request_capture'
        returned_result = None
        try:
            self.request = capture_recipe_attempt_request(self.recipe, self.paths)
            if self.binding is not None:
                context = (self.context.root, self.context.image, self.context.source_path)
                self.invoked_executor = PROJECT_CORRECTION_EXECUTOR
                phase = 'correction_path'
                result, prepared = run_project_correction(self.recipe, context[0], self.binding,
                    context[1], context[2], cache=self.cache,
                    previous=self.previous, cancel=self.cancel.is_set)
            else:
                self.invoked_executor = CORRECTION_FILE_EXECUTOR
                phase = 'correction_path'
                result, prepared = _inputs.run_native_correction_files(self.recipe, self.paths,
                    previous=self.previous, device='cuda', cache=self.cache, cancel=self.cancel.is_set,
                    max_combined_pixels=MAX_COMBINED_PIXELS, max_prepared_cuda_bytes=MAX_PREPARED_CUDA_BYTES)
            self.result_returned = True
            returned_result = result
            phase = 'delivery'
            if self.cancel.is_set():
                raise NativeCorrectionCancelled('Correction cancelled after the runner returned.')
            self.signals.completed.emit(_Completion(result, prepared, self.generation))
        except Exception as error:
            self.record_attempt(error, phase=phase, returned_result=returned_result,
                                invoked_executor=self.invoked_executor)
            self.signals.completed.emit(_Completion(error, None, self.generation))


def _validate_result_outputs(recipe, result):
    """Check preview/Keep products without copying the image or rerunning it."""
    from seedvision.export.native_results import _correction_spec, _immutable_correction_matrix
    if (result.recipe_identity.analytical_dict() != recipe.identity.analytical_dict()
            or tuple(node.stage_id for node in result.nodes) != tuple(stage.stage_id for stage in recipe.stages)):
        raise ValueError('Correction calculation returned a different recipe or incomplete stage inventory.')
    for stage, node in zip(recipe.stages, result.nodes):
        if node.operation_id != stage.operation_id or dict(node.parameters) != recipe.parameters_for_runner(stage.stage_id):
            raise ValueError('Correction calculation differs from its selected operation or controls.')
        if node.status != 'executed':
            continue
        if set(node.outputs) != {port.name for port in stage.ports if port.direction == 'output'}:
            raise ValueError('Correction calculation returned incomplete output ports.')
        image = node.outputs['corrected_image']
        if (not torch.is_tensor(image) or image.device.type != 'cuda' or image.layout != torch.strided
                or image.dtype != torch.uint8):
            raise ValueError('Corrected image must be a strided CUDA uint8 tensor.')
        _correction_spec('corrected_image', image.shape, 'uint8', result.coordinate_frame)
        for name in ('target_matrix', 'source_matrix', 'transformation_matrix'):
            value = node.outputs[name]
            _immutable_correction_matrix(value)
            _correction_spec(name, value.shape, value.dtype, result.coordinate_frame)


class NativeCorrectionDialog(NativeRecipeDialog):
    """Shared recipe canvas/inspector and worker lifecycle with four explicit files."""
    def __init__(self, recipe, parent=None, *, project_context=None, project_binding=None, keep_in_project=None):
        if not isinstance(recipe, RecipeSpec) or recipe.recipe_id != RECIPE_ID:
            raise NativeCorrectionError(f"Select the canonical {RECIPE_ID} RecipeSpec.")
        self._project_context, self._project_binding = project_context, project_binding
        self._generation, self._result_generation, self._keeping = 0, None, False
        if project_context is not None and not isinstance(project_context, CorrectionProjectContext):
            raise ValueError('Correction requires its explicit project image context.')
        if project_binding is not None:
            if project_context is None:
                raise ValueError('Reopened correction requires its project image context.')
            assets.validate_correction_bindings((project_binding,), (project_context.image,),
                                                native_recipes=(build_native_correction_recipe(recipe),))
        super().__init__(build_native_correction_recipe(recipe), parent, keep_in_project=keep_in_project)
        self.setWindowTitle("Native PlantCV colour-card correction")
        self._paths, self._prepared = {}, None
        self.choose_button.hide()
        self.source_label.hide()
        self._input_buttons, self._input_labels = {}, {}
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        for index, name in enumerate(_PORTS):
            group = QWidget(self)
            column = QVBoxLayout(group)
            column.setContentsMargins(2, 2, 2, 2)
            button, label = QPushButton("Choose file…", group), QLabel("Not selected", group)
            button.setObjectName("nativeCorrectionChoose"+"".join(part.title() for part in name.split("_")))
            label.setTextFormat(Qt.TextFormat.PlainText)
            label.setWordWrap(True)
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            column.addWidget(QLabel(_TITLES[name], group))
            column.addWidget(button)
            column.addWidget(label)
            grid.addWidget(group, index//2, index%2)
            self._input_buttons[name], self._input_labels[name] = button, label
            button.clicked.connect(lambda checked=False, port=name: self._choose_file(port))
        self.layout().insertLayout(1, grid)
        self.run_button.setEnabled(False)
        self.status_label.setText("Choose four files: target RGB, target chip-label mask, source RGB, and source chip-label mask.")
        self.project_button.setVisible(project_context is not None and keep_in_project is not None)
        self.project_button.setEnabled(False)
        self.canvas.layout_changed.connect(self._keep_layout_changed)
        if project_binding is not None:
            for asset in project_binding.inputs:
                self._paths[asset.role] = assets.resolve_correction_asset_path(project_context.root, asset.path)
                self._input_labels[asset.role].setText(str(self._paths[asset.role]))
            self.source_path = self._paths['source_rgb']
            self.run_button.setEnabled(True)
            self.status_label.setText('Saved correction inputs selected. Run to verify the project copies and compute.')
        elif project_context is not None:
            self.set_input_path('source_rgb', project_context.source_path)

    def set_input_path(self, port, path):
        if self._worker is not None or self._keeping:
            raise RuntimeError("Correction inputs cannot change while the worker is running.")
        if port not in _PORTS:
            raise ValueError(f"Unknown correction file input {port!r}.")
        resolved = Path(path).expanduser().resolve()
        was_pinned = self._project_binding is not None
        self._paths[port] = resolved
        self._project_binding = None
        self._input_labels[port].setText(str(resolved))
        if was_pinned and port != 'source_rgb':
            # A new mask/target starts a fresh selection for this exact image.
            # An explicitly chosen source_rgb always remains authoritative.
            self._paths['source_rgb'] = self._project_context.source_path
            self._input_labels['source_rgb'].setText(str(self._paths['source_rgb']))
        self._prepared = None
        self.cache.values.pop(_CORRECTION_CACHE_KEY, None)
        self.source_path = self._paths.get("source_rgb") if set(self._paths) == set(_PORTS) else None
        self._invalidate_view(STAGE_ID)
        missing = [name for name in _PORTS if name not in self._paths]
        self.run_button.setEnabled(not missing)
        self.status_label.setText("Correction inputs ready. Run to compute." if not missing
            else "Select the remaining files: "+", ".join(_TITLES[name] for name in missing)+".")

    def _choose_file(self, port):
        path, _ = QFileDialog.getOpenFileName(self, f"Choose {_TITLES[port]}", "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;All files (*)")
        if path:
            self.set_input_path(port, path)

    def start_run(self):
        if self._worker is not None or self._keeping or set(self._paths) != set(_PORTS):
            return
        self._generation += 1
        self._result_generation = None
        self._cancel = threading.Event()
        worker = _CorrectionRunWorker(self.recipe, self._paths, self.cache, self._prepared, self._cancel,
            generation=self._generation, context=self._project_context, binding=self._project_binding)
        self._worker = worker
        worker.signals.completed.connect(lambda completion, worker=worker: self._correction_completed(worker, completion))
        self.choose_button.setEnabled(False)
        for button in self._input_buttons.values():
            button.setEnabled(False)
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.export_button.setEnabled(False)
        self.project_button.setEnabled(False)
        self.inspector.setEnabled(False)
        self.status_label.setText("Preparing four explicit files and computing colour correction…")
        self.result = None
        self._select_stage(self._selected_stage)
        QThreadPool.globalInstance().start(worker)

    def _correction_completed(self, worker, completion):
        result = completion.result if isinstance(completion, _Completion) else completion
        generation = completion.generation if isinstance(completion, _Completion) else None
        stale = self._worker is not worker or generation is not None and generation != self._generation
        phase = None
        returned_result = result
        if not isinstance(result, Exception):
            if stale or worker.cancel.is_set() or self._close_requested:
                result = NativeCorrectionCancelled('Correction cancelled or superseded before result delivery.')
                phase = 'delivery'
            elif isinstance(result, NativeCorrectionResult):
                try:
                    _validate_result_outputs(self.recipe, result)
                except (ValueError, TypeError, AttributeError) as error:
                    result, phase = error, 'result_validation'
        if phase is not None:
            if hasattr(worker, 'record_attempt'):
                worker.record_attempt(result, phase=phase, returned_result=returned_result,
                    capture_completion=False, invoked_executor=worker.invoked_executor)
            completion = _Completion(result, None, generation)
        quality = getattr(worker, 'quality', None)
        if quality is not None and quality not in self._attempt_history:
            self._attempt_history.append(quality)
        if self._worker is not worker:
            return
        self._attempt_quality = quality
        self._completed(completion)
        if getattr(worker, 'quality_error', None):
            self.status_label.setText(self.status_label.text()+' Recipe QC recording failed: '+worker.quality_error)

    def _completed(self, completion):
        generation = completion.generation if isinstance(completion, _Completion) else None
        if generation is not None and generation != self._generation:
            if self._worker is None or getattr(self._worker, 'generation', None) != generation:
                return  # A completion from an older worker cannot release a newer worker.
            completion = _Completion(NativeRecipeCancelled('Stale correction calculation discarded.'), None, generation)
        if isinstance(completion, _Completion):
            self._prepared, result = (None if self._close_requested else completion.prepared), completion.result
        else:
            self._prepared, result = None, completion
        if isinstance(result, NativeCorrectionResult):
            try:
                _validate_result_outputs(self.recipe, result)
            except (ValueError, TypeError, AttributeError) as error:
                result, self._prepared = error, None
        if isinstance(result, Exception):
            self.result = None
            self.export_button.setEnabled(False)
        super()._completed(result)
        self._result_generation = self._generation if self.result is not None and not self._close_requested else None
        if isinstance(result, NativeCorrectionCancelled) and not self._close_requested:
            self.status_label.setText("Calculation cancelled.")
        ready = set(self._paths) == set(_PORTS)
        self.source_path = self._paths.get("source_rgb") if ready else None
        for button in self._input_buttons.values():
            button.setEnabled(self._worker is None)
        if self._worker is None and not self._close_requested:
            self.run_button.setEnabled(ready)
        self._refresh_keep_enabled()

    def _invalidate_view(self, stage_id=None):
        self._generation += 1
        self._result_generation = None
        if self._keeping and self._cancel is not None:
            self._cancel.set()
        super()._invalidate_view(stage_id)
        self.project_button.setEnabled(False)

    def _keep_layout_changed(self):
        if self._keeping and self._cancel is not None:
            self._cancel.set()

    def _current_keep_binding(self):
        result, context = self.result, self._project_context
        if (context is None or self._keep_in_project is None or self._close_requested
                or self._result_generation != self._generation
                or not isinstance(result, NativeCorrectionResult) or result.status != 'executed'
                or tuple(node.stage_id for node in result.nodes) != tuple(stage.stage_id for stage in self.recipe.stages)
                or any(node.status != 'executed' for node in result.nodes)):
            raise ValueError('Run the current correction successfully for the selected project image before keeping it.')
        from seedvision.export.native_results import _identity
        _validate_result_outputs(self.recipe, result)
        _identity(self.recipe, result)
        if result.input_origins is None:
            if Path(result.input_sources['source_rgb']['path']).resolve() != context.source_path:
                raise ValueError('The correction source must be the exact selected project image.')
            binding = assets.binding_for_result(self.recipe, context.image.identifier, result)
        else:
            binding = result.input_origins.binding
            if binding != self._project_binding:
                raise ValueError('The correction result is stale for its saved input selection.')
        assets.validate_correction_bindings((binding,), (context.image,), native_recipes=(self.recipe,))
        return binding

    def _refresh_keep_enabled(self):
        try:
            self._current_keep_binding()
            enabled = self._worker is None and not self._keeping
        except (ValueError, TypeError, AttributeError):
            enabled = False
        self.project_button.setEnabled(enabled)

    def _keep_recipe_in_project(self):
        if self._worker is not None or self._keeping:
            return
        try:
            request = CorrectionKeepRequest(self._project_context, self._selected_recipe_with_layout(),
                self._current_keep_binding(), self._generation, self.result.input_origins is not None)
        except (ValueError, TypeError, AttributeError) as error:
            self.status_label.setText(f'Could not keep correction: {error}')
            self._refresh_keep_enabled()
            return
        self._keeping, self._cancel = True, threading.Event()
        states = [(button, button.isEnabled()) for button in self._export_conflicting_buttons()]
        for button, _ in states:
            button.setEnabled(False)
        self.inspector.setEnabled(False)
        def current():
            return (not self._close_requested and not self._cancel.is_set()
                    and self._generation == request.generation
                    and self._selected_recipe_with_layout() == request.recipe)
        try:
            captured = self._keep_in_project(request, self._cancel, current)
            if not current():
                raise NativeRecipeCancelled('Correction Keep cancelled or changed before completion.')
            validate_correction_keep_result(request, captured)
            if request.pinned and captured != self._project_binding:
                self._project_binding = captured
                for asset in captured.inputs:
                    self._paths[asset.role] = assets.resolve_correction_asset_path(request.context.root, asset.path)
                    self._input_labels[asset.role].setText(str(self._paths[asset.role]))
                self.source_path = self._paths['source_rgb']
                self._prepared = None
                self.cache.values.pop(_CORRECTION_CACHE_KEY, None)
                self._invalidate_view(STAGE_ID)
            self.status_label.setText('Correction recipe and verified inputs kept in the project. Use Save Project to save the master.')
        except (ValueError, TypeError, OSError, RuntimeError) as error:
            self.status_label.setText(f'Could not keep correction: {error}')
        finally:
            self._keeping, self._cancel = False, None
            self.inspector.setEnabled(True)
            for button, enabled in states:
                button.setEnabled(enabled)
            self._refresh_keep_enabled()
            if self._close_requested:
                super().reject()

    def _request_close(self):
        if self._keeping:
            self._close_requested = True
            self._cancel.set()
            return True
        return super()._request_close()

    def _export_completed(self, result):
        super()._export_completed(result)
        self._refresh_keep_enabled()

    def _select_stage(self, stage_id):
        super()._select_stage(stage_id)
        if self.result is not None:
            details = [f"{_TITLES[name]}: {self.result.preparation.get(name, 'unavailable')}" for name in _PORTS]
            self.facts.setPlainText(self.facts.toPlainText()+"\n"+"\n".join(details))

    def _export_conflicting_buttons(self):
        return (*super()._export_conflicting_buttons(), *self._input_buttons.values())
