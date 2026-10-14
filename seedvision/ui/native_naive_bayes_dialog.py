"""Qt model selection and CUDA mask previews for a frozen PlantCV classifier."""
from collections.abc import Mapping
from dataclasses import replace
import math

import numpy as np
import torch
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QComboBox, QDialog, QFileDialog, QHBoxLayout, QLabel, QPushButton, QWidget

from seedvision.benchmarking.native_naive_bayes_recipe import (
    CLASSIFIER_STAGE_ID, MODEL_PARAMETER_ID, MODEL_STAGE_ID, TRAINING_FIELD,
    load_selected_model, model_source_for_path,
)
from seedvision.ui.native_naive_bayes_training_dialog import NativeNaiveBayesTrainingDialog
from seedvision.ui.native_recipe_dialog import NativeRecipeDialog


class NativeNaiveBayesDialog(NativeRecipeDialog):
    """Reuse the native canvas, asynchronous worker, cache and recipe saving."""

    model_stage_id = MODEL_STAGE_ID
    model_parameter_id = MODEL_PARAMETER_ID
    classifier_stage_id = CLASSIFIER_STAGE_ID
    model_button_caption = 'Choose PDF model…'
    model_button_object_name = 'nativeNaiveBayesChooseModel'
    model_dialog_title = 'Choose frozen PlantCV probability model'
    model_file_filter = 'PlantCV probability tables (*.tsv *.txt);;All files (*)'
    class_selector_object_name = 'nativeNaiveBayesClassSelector'
    class_selector_accessible_name = 'Class mask'
    no_model_text = 'No model selected. Choose a PlantCV TSV probability table.'
    model_required_text = 'Choose a frozen PlantCV model before running this recipe.'
    model_selected_text = 'Model selected and byte-bound. Run to compute its class masks.'

    def __init__(self, recipe, parent=None, *, model_asset_paths=None, keep_in_project=None):
        self._preferred_class = ''
        self._last_class_preview = None
        self._replay_class_masks = None
        self._replay_class_shape = None
        self._training_dialog = None
        super().__init__(recipe, parent, model_asset_paths=model_asset_paths, keep_in_project=keep_in_project)
        row = QWidget(self)
        controls = QHBoxLayout(row)
        controls.setContentsMargins(0, 0, 0, 0)
        self.choose_model_button = QPushButton(self.model_button_caption, row)
        self.choose_model_button.setObjectName(self.model_button_object_name)
        self.model_label = QLabel(row)
        self.model_label.setWordWrap(True)
        self.model_label.setTextFormat(Qt.TextFormat.PlainText)
        self.class_selector = QComboBox(row)
        self.class_selector.setObjectName(self.class_selector_object_name)
        self.class_selector.setAccessibleName(self.class_selector_accessible_name)
        self.class_selector.hide()
        controls.addWidget(self.choose_model_button)
        if self.model_stage_id == MODEL_STAGE_ID:
            self.train_model_button = QPushButton('Train model…', row)
            self.train_model_button.setObjectName('nativeNaiveBayesTrainModel')
            controls.addWidget(self.train_model_button)
            self.train_model_button.clicked.connect(self._train_model)
        controls.addWidget(self.model_label, 1)
        controls.addWidget(self.class_selector)
        self.layout().insertWidget(0, row)
        self.choose_model_button.clicked.connect(self._choose_model)
        self.class_selector.currentIndexChanged.connect(self._show_selected_class_mask)
        self._update_model_label()
        self._sync_run_enabled()

    def _configure_parameters(self):
        super()._configure_parameters()
        node = self.graph.nodes.get(self.model_stage_id)
        if node is not None:
            # The picker changes one effective asset selection. Hashes and
            # class order are derived metadata, never independent controls.
            node.parameters = {}
            node.parameter_specs = ()
            node.parameter_sections = ()

    def _selected_model_source(self):
        return self.recipe.parameters_for_runner(self.model_stage_id).get(self.model_parameter_id)

    def _model_source_for_path(self, path):
        return model_source_for_path(path)

    def _format_model_label(self, selected):
        classes = tuple(selected.get('class_names', ()))
        training = ' · training provenance retained in this recipe' if TRAINING_FIELD in selected else ''
        return (
            f"{selected.get('path', '')} · SHA-256 {selected.get('source_sha256', '')[:12]}… · "
            f"{len(classes)} ordered classes{training} · saved recipes reference this local file")

    def _train_model(self):
        if self._worker is not None or self._training_dialog is not None:
            return
        dialog = NativeNaiveBayesTrainingDialog(self)
        self._training_dialog = dialog
        selected = None
        try:
            accepted = dialog.exec() == QDialog.DialogCode.Accepted
            if accepted:
                selected = dialog.saved_model_source
        finally:
            self._training_dialog = None
            # Keep only compact selection metadata after the modal worker has
            # drained; repeated training must not retain old CUDA fits in children.
            dialog._completion = None
            dialog.deleteLater()
        if self._close_requested:
            super().reject()
            return
        if not accepted or selected is None:
            return
        try:
            load_selected_model(selected)
            self._set_model_source(selected)
        except (ValueError, TypeError, OSError) as error:
            self.status_label.setText(f'Trained model selection failed: {error}')
            return
        self.status_label.setText('Trained model selected. Run to compute its class masks; '
            'Keep in project retains the training provenance.')

    def _request_close(self):
        if self._training_dialog is not None:
            self._close_requested = True
            self._training_dialog.reject()
            return True
        return super()._request_close()

    def _update_model_label(self):
        selected = self._selected_model_source()
        if not isinstance(selected, Mapping):
            self.model_label.setText(self.no_model_text)
            return
        self.model_label.setText(self._format_model_label(selected))

    def _sync_run_enabled(self):
        if self._worker is None:
            self.run_button.setEnabled(
                self.source_path is not None and self._selected_model_source() is not None)

    def set_source_path(self, path):
        super().set_source_path(path)
        self._sync_run_enabled()

    def _set_model_source(self, selected):
        self.model_asset_paths.pop(self.model_stage_id, None)
        stage = self.recipe.stage(self.model_stage_id)
        changed = replace(stage, parameters=tuple(
            replace(item, value=selected) if item.parameter_id == self.model_parameter_id else item
            for item in stage.parameters))
        self.recipe = replace(self.recipe, stages=tuple(
            changed if item.stage_id == self.model_stage_id else item for item in self.recipe.stages))
        self._preferred_class = ''
        self._configure_parameters()
        self._update_model_label()
        self._invalidate_view(self.model_stage_id)
        self._sync_run_enabled()

    def _choose_model(self):
        if self._worker is not None:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, self.model_dialog_title, '', self.model_file_filter)
        if not path:
            return
        try:
            self._set_model_source(self._model_source_for_path(path))
            self.status_label.setText(self.model_selected_text)
        except (ValueError, OSError, TypeError) as error:
            self._set_model_source(None)
            self.status_label.setText(f'Model selection failed: {type(error).__name__}: {error}')

    def start_run(self):
        if self._worker is not None or self.source_path is None:
            return
        if self._selected_model_source() is None:
            self.status_label.setText(self.model_required_text)
            return
        self._clear_class_selector()
        super().start_run()
        self.choose_model_button.setEnabled(self._worker is None)
        if hasattr(self, 'train_model_button'):
            self.train_model_button.setEnabled(self._worker is None)

    def _completed(self, result):
        super()._completed(result)
        self.choose_model_button.setEnabled(not self._close_requested)
        if hasattr(self, 'train_model_button'):
            self.train_model_button.setEnabled(not self._close_requested)
        self._sync_run_enabled()
        if isinstance(result, Exception) or self.result is None:
            self._clear_class_selector()
            self.preview.clear()

    def _clear_class_selector(self):
        self._last_class_preview = None
        self._replay_class_masks = None
        self._replay_class_shape = None
        if not hasattr(self, 'class_selector'):
            return
        self.class_selector.blockSignals(True)
        self.class_selector.clear()
        self.class_selector.blockSignals(False)
        self.class_selector.hide()

    def _select_stage(self, stage_id):
        self._replay_class_masks = None
        self._replay_class_shape = None
        super()._select_stage(stage_id)
        if not hasattr(self, 'class_selector'):
            return
        if stage_id != self.classifier_stage_id or self.result is None:
            self._clear_class_selector()
            return
        outputs = self.result.node(self.classifier_stage_id).outputs
        names = tuple(outputs.get('class_names', ()))
        masks = outputs.get('class_masks')
        if not names or not isinstance(masks, Mapping):
            self._clear_class_selector()
            return
        self.class_selector.blockSignals(True)
        self.class_selector.clear()
        self.class_selector.addItems(names)
        index = self.class_selector.findText(self._preferred_class)
        self.class_selector.setCurrentIndex(index if index >= 0 else 0)
        self.class_selector.blockSignals(False)
        self.class_selector.show()
        self._show_selected_class_mask()

    def _replay_completed(self, result):
        super()._replay_completed(result)
        masks=None if isinstance(result,Exception) else result.outputs.get('class_masks')
        if (self._close_requested or not isinstance(masks,Mapping)
                or result.stage_id!=self.classifier_stage_id):
            self._clear_class_selector()
            return
        self._replay_class_masks=masks
        self._replay_class_shape=tuple(result.metadata['coordinate_frame']['shape'])
        self.class_selector.blockSignals(True)
        self.class_selector.clear()
        self.class_selector.addItems(tuple(result.outputs['class_names']))
        index=self.class_selector.findText(self._preferred_class)
        self.class_selector.setCurrentIndex(index if index>=0 else 0)
        self.class_selector.blockSignals(False)
        self.class_selector.show()
        self._show_selected_class_mask()

    def _show_selected_class_mask(self, *_):
        if (self.result is None and self._replay_class_masks is None or self._selected_stage != self.classifier_stage_id
                or self.class_selector.currentIndex() < 0):
            self._last_class_preview = None
            self.preview.clear()
            return
        masks=(self._replay_class_masks if self._replay_class_masks is not None
               else self.result.node(self.classifier_stage_id).outputs.get('class_masks',{}))
        frame_shape=self._replay_class_shape if self._replay_class_masks is not None else self.result.coordinate_frame.shape
        class_name = self.class_selector.currentText()
        mask = masks.get(class_name)
        if (not torch.is_tensor(mask) or mask.device.type != 'cuda' or mask.dtype != torch.uint8
                or mask.ndim != 2 or tuple(mask.shape) != frame_shape):
            self._last_class_preview = None
            self.preview.setText('The selected class mask is unavailable.')
            return
        self._preferred_class = class_name
        stride = max(1, math.ceil(mask.shape[0]/256), math.ceil(mask.shape[1]/256))
        array = np.ascontiguousarray(mask[::stride, ::stride].contiguous().cpu().numpy())
        self._last_class_preview = array.copy()
        image = QImage(array.data, array.shape[1], array.shape[0], array.strides[0],
            QImage.Format.Format_Grayscale8).copy()
        self.preview.setPixmap(QPixmap.fromImage(image).scaled(
            400, 300, Qt.AspectRatioMode.KeepAspectRatio))
        self.preview.show()

    def _parameters_reset(self, stage_id):
        if self._worker is not None:
            return
        super()._parameters_reset(stage_id)
        if stage_id == self.model_stage_id:
            self._preferred_class = ''
            self._clear_class_selector()
            self._update_model_label()
            self._sync_run_enabled()
