"""Native Qt calculation view for selected PlantCV seed and threshold recipes."""
from dataclasses import replace
import json
import math
from pathlib import Path
import threading
from uuid import uuid4
from collections.abc import Mapping

import numpy as np
import torch
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Qt, QSaveFile, QIODevice
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFileDialog, QHBoxLayout,
    QLabel, QPushButton, QScrollArea, QSplitter, QTextBrowser, QVBoxLayout, QWidget)

from seedvision.benchmarking.native_recipe import run_native_recipe_path, NativeRecipeCancelled
from seedvision.benchmarking.native_stage_replay import replay_native_stage, SUPPORTED_REPLAY_OPERATIONS
from seedvision.benchmarking.recipes import to_pipeline_graph, LayoutPosition, RecipeSpec
from seedvision.cuda.plantcv_color import ColorAnalysisResult
from seedvision.benchmarking.color_histogram import HueHistogramChart
from seedvision.pipeline.model import ParameterSpec, ParameterSection, NodeStatus
from seedvision.segmentation.baseline import PipelineAnalysisCache
from seedvision.ui.pipeline_canvas import PipelineCanvas
from seedvision.ui.pipeline_inspector import PipelineInspector
from seedvision.ui.color_histogram import HueHistogramView
from seedvision.export.native_results import export_native_recipe_result, read_native_recipe_inventory, _json_bytes
from seedvision.export.recipe_attempt_qc import capture_recipe_attempt_request, recipe_attempt_qc


class _Signals(QObject):
    completed = Signal(object)


class _RunWorker(QRunnable):
    def __init__(self, recipe, path, cache, cancel, model_asset_paths=None):
        super().__init__()
        self.recipe, self.path, self.cache, self.cancel = recipe, path, cache, cancel
        self.model_asset_paths = dict(model_asset_paths or {})
        self.signals = _Signals()
        self.request = None
        self.quality = None
        self.quality_error = None
        self.result_returned = False

    def record_attempt(self, error, *, phase, returned_result=None, capture_completion=True,
                       invoked_executor=None):
        if self.request is None:
            return
        try:
            from seedvision.benchmarking.native_correction_recipe import NativeCorrectionCancelled
            self.quality = recipe_attempt_qc(self.request,
                state='cancelled' if isinstance(error, (NativeRecipeCancelled, NativeCorrectionCancelled)) else 'failed',
                phase=phase, detail=str(error) or type(error).__name__, exception=error,
                invoked=phase != 'request_capture', result_returned=self.result_returned,
                returned_result=returned_result, capture_completion=capture_completion,
                invoked_executor=invoked_executor)
        except Exception as recording_error:
            self.quality_error = f'{type(recording_error).__name__}: {recording_error}'
            error.add_note('Recipe attempt QC could not be captured: '+self.quality_error)

    def run(self):
        phase = 'request_capture'
        returned_result = None
        try:
            inputs = {'source_image': self.path}
            inputs.update({'model.'+key: path for key, path in self.model_asset_paths.items()})
            self.request = capture_recipe_attempt_request(self.recipe, inputs)
            options = {'model_asset_paths': self.model_asset_paths} if self.model_asset_paths else {}
            phase = 'native_path'
            result = run_native_recipe_path(self.recipe, self.path, cache=self.cache, cancel=self.cancel.is_set, **options)
            self.result_returned = True
            returned_result = result
            phase = 'delivery'
            if self.cancel.is_set():
                raise NativeRecipeCancelled('Recipe calculation cancelled after the runner returned.')
        except Exception as error:
            self.record_attempt(error, phase=phase, returned_result=returned_result)
            result = error
        self.signals.completed.emit(result)


class _ExportWorker(QRunnable):
    def __init__(self, recipe, result, directory, cancel):
        super().__init__()
        self.recipe, self.result, self.directory, self.cancel = recipe, result, directory, cancel
        self.signals = _Signals()

    def run(self):
        try:
            result = export_native_recipe_result(self.directory, self.recipe, self.result,
                cancel=self.cancel.is_set)
            from seedvision.export.method_qc import load_method_qc, FILENAME
            self.quality = load_method_qc(result.parent/FILENAME)
        except Exception as error:
            result = error
        self.signals.completed.emit(result)


class _ReplayWorker(QRunnable):
    def __init__(self, recipe, stage_id, directory, cancel):
        super().__init__()
        self.recipe, self.stage_id, self.directory, self.cancel = recipe, stage_id, directory, cancel
        self.signals = _Signals()

    def run(self):
        try:
            report, _ = read_native_recipe_inventory(self.directory, cancel=self.cancel.is_set)
            saved = RecipeSpec.from_json(_json_bytes(report['recipe']))
            if saved.identity.analytical_dict() != self.recipe.identity.analytical_dict():
                raise ValueError('Saved run belongs to different recipe settings or topology.')
            result = replay_native_stage(self.directory, self.stage_id, cancel=self.cancel.is_set,
                expected_revision=report['result_revision'], expected_recipe_identity=self.recipe.identity)
        except Exception as error:
            result = error
        self.signals.completed.emit(result)


class NativeRecipeDialog(QDialog):
    """Use the existing canvas/inspector; retain CUDA products for this session."""
    def __init__(self, recipe, parent=None, *, model_asset_paths=None, keep_in_project=None):
        super().__init__(parent)
        self.recipe = recipe
        self._initial_recipe = recipe
        self.model_asset_paths = dict(model_asset_paths or {})
        self._initial_model_asset_paths = dict(self.model_asset_paths)
        self._keep_in_project = keep_in_project
        self.result = None
        self.replay_result = None
        self._exported_quality = None
        self._attempt_quality = None
        self._attempt_history = []
        self.cache = PipelineAnalysisCache()
        self.source_path = None
        self._worker = None
        self._cancel = None
        self._close_requested = False
        self._selected_stage = recipe.stages[0].stage_id
        self.setWindowTitle(f'Native calculation — {recipe.title}')
        self.resize(1440, 900)
        self.setMinimumSize(1050, 650)
        layout = QVBoxLayout(self)
        toolbar = QHBoxLayout()
        self.choose_button = QPushButton('Choose image…')
        self.choose_button.setObjectName('nativeRecipeChooseImage')
        self.run_button = QPushButton('Run selected recipe')
        self.run_button.setObjectName('nativeRecipeRun')
        self.run_button.setEnabled(False)
        self.stop_button = QPushButton('Stop')
        self.stop_button.setEnabled(False)
        self.save_button = QPushButton('Save selected recipe…')
        self.export_button = QPushButton('Export run…')
        self.export_button.setObjectName('nativeRecipeExportRun')
        self.export_button.setEnabled(False)
        self.qc_button = QPushButton('Recorded recipe QC…', self)
        self.qc_button.setObjectName('nativeRecipeMethodQC')
        self.qc_button.clicked.connect(self._review_method_qc)
        self.source_label = QLabel('No image selected')
        self.source_label.setTextFormat(Qt.TextFormat.PlainText)
        toolbar.addWidget(self.choose_button)
        toolbar.addWidget(self.source_label, 1)
        toolbar.addWidget(self.run_button)
        toolbar.addWidget(self.stop_button)
        toolbar.addWidget(self.save_button)
        toolbar.addWidget(self.export_button)
        toolbar.addWidget(self.qc_button)
        layout.addLayout(toolbar)
        self.status_label = QLabel('Set required node parameters, choose an image and run. '
            'Measurements remain in pixels; reference agreement is separate from scientific validation.')
        from seedvision.benchmarking.native_smpts_recipe import is_executable_paper_recipe
        if is_executable_paper_recipe(recipe):
            self.status_label.setText('Provisional SMPTS paper reconstruction with declared assumptions. '
                'Inspect threshold branches, contact selection, targeted erosion and original RGB crops. '
                'Original-tool conformance and scientific validation remain unavailable.')
        self.status_label.setWordWrap(True)
        self.status_label.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.status_label)
        self.graph = to_pipeline_graph(recipe)
        self._configure_parameters()
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.canvas = PipelineCanvas(self.graph, read_only=True, reference_view=True,
            initial_focus_node_id=self._selected_stage, preserve_initial_layout=bool(recipe.layout))
        splitter.addWidget(self.canvas)
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        self.inspector = PipelineInspector(panel)
        self.inspector_scroll = QScrollArea(panel)
        self.inspector_scroll.setWidgetResizable(True)
        self.inspector_scroll.setMinimumHeight(140)
        self.inspector_scroll.setWidget(self.inspector)
        panel_layout.addWidget(self.inspector_scroll)
        self.facts = QTextBrowser()
        self.facts.setMinimumHeight(140)
        panel_layout.addWidget(self.facts)
        self.crop_selector = QComboBox(panel)
        self.crop_selector.setObjectName('nativePaperCropSelector')
        self.crop_selector.setToolTip('Inspect each original masked RGB crop and its source-parent lineage.')
        self.crop_selector.hide()
        self.crop_selector.currentIndexChanged.connect(self._select_paper_crop)
        panel_layout.addWidget(self.crop_selector)
        self.preview = QLabel('Select a computed image, mask or label node to preview its pixels.')
        self.preview.setWordWrap(True)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(280, 200)
        self.preview.setMaximumHeight(340)
        panel_layout.addWidget(self.preview)
        self.histogram = HueHistogramView(panel)
        self.histogram.setMaximumHeight(340)
        panel_layout.addWidget(self.histogram)
        self.histogram.hide()
        splitter.addWidget(panel)
        splitter.setSizes([1000, 420])
        layout.addWidget(splitter, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.project_button = QPushButton('Keep recipe in project', self)
        self.project_button.setObjectName('nativeRecipeKeepInProject')
        buttons.addButton(self.project_button, QDialogButtonBox.ButtonRole.ActionRole)
        self.replay_button = QPushButton('Replay saved stage…', self)
        self.replay_button.setObjectName('nativeRecipeReplaySavedStage')
        buttons.addButton(self.replay_button, QDialogButtonBox.ButtonRole.ActionRole)
        self.replay_button.clicked.connect(self._choose_stage_replay)
        self.project_button.setVisible(keep_in_project is not None)
        self.project_button.clicked.connect(self._keep_recipe_in_project)
        from seedvision.benchmarking.native_watershed_diagnostic_recipe import BASE_IDS
        self.diagnostic_button = QPushButton('Add watershed debug image', self)
        self.diagnostic_button.setObjectName('nativeRecipeWatershedDiagnostic')
        buttons.addButton(self.diagnostic_button, QDialogButtonBox.ButtonRole.ActionRole)
        self.diagnostic_button.setVisible(recipe.recipe_id in BASE_IDS)
        self.diagnostic_button.clicked.connect(self._add_watershed_diagnostic)
        from seedvision.benchmarking.native_morphology_roi_recipe import BASE_IDS as ROI_BASE_IDS
        self.morphology_roi_button = QPushButton('Add morphology ROI controls', self)
        self.morphology_roi_button.setObjectName('nativeRecipeMorphologyROI')
        buttons.addButton(self.morphology_roi_button, QDialogButtonBox.ButtonRole.ActionRole)
        self.morphology_roi_button.setVisible(recipe.recipe_id in ROI_BASE_IDS)
        self.morphology_roi_button.clicked.connect(self._add_morphology_roi)
        from seedvision.benchmarking.native_fill_roi_recipe import (
            BASE_IDS as FILL_BASE_IDS, RECIPE_IDS as FILL_IDS, base_recipe_id as fill_base)
        self.morphology_roi_button.setVisible(recipe.recipe_id in ROI_BASE_IDS or
            recipe.recipe_id in FILL_IDS and fill_base(recipe.recipe_id) in ROI_BASE_IDS)
        self.fill_roi_button = QPushButton('Add small-object ROI controls', self)
        self.fill_roi_button.setObjectName('nativeRecipeFillROI')
        buttons.addButton(self.fill_roi_button, QDialogButtonBox.ButtonRole.ActionRole)
        self.fill_roi_button.setVisible(recipe.recipe_id in FILL_BASE_IDS)
        self.fill_roi_button.clicked.connect(self._add_fill_roi)
        self.median_roi_button = QPushButton('Add median ROI controls', self)
        self.median_roi_button.setObjectName('nativeRecipeMedianROI')
        buttons.addButton(self.median_roi_button, QDialogButtonBox.ButtonRole.ActionRole)
        self.median_roi_button.clicked.connect(self._add_median_roi)
        self.reader_button = QPushButton('Add grayscale reader control', self)
        self.reader_button.setObjectName('nativeRecipeGrayReader')
        buttons.addButton(self.reader_button, QDialogButtonBox.ButtonRole.ActionRole)
        self.reader_button.clicked.connect(self._add_reader)
        self._refresh_roi_buttons()
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.choose_button.clicked.connect(self._choose_image)
        self.run_button.clicked.connect(self.start_run)
        self.stop_button.clicked.connect(self.stop)
        self.save_button.clicked.connect(self._save_recipe)
        self.export_button.clicked.connect(self._export_run)
        self.canvas.node_selected.connect(self._select_stage)
        self.inspector.parameter_changed.connect(self._parameter_changed)
        self.inspector.parameters_reset.connect(self._parameters_reset)
        self._select_stage(self._selected_stage)

    def _add_watershed_diagnostic(self):
        from seedvision.benchmarking.native_watershed_diagnostic_recipe import (
            BASE_IDS, STAGE, build_native_watershed_diagnostic_recipe)
        from seedvision.benchmarking.native_reader_recipe import is_reader_recipe
        if self._worker is not None or self.recipe.recipe_id not in BASE_IDS or is_reader_recipe(self.recipe):
            return
        self.recipe = build_native_watershed_diagnostic_recipe(self._selected_recipe_with_layout())
        self._initial_recipe = self.recipe
        self.result = self.replay_result = None
        self.graph = to_pipeline_graph(self.recipe)
        self._configure_parameters()
        self.canvas.graph = self.graph
        self.canvas.rebuild_from_graph(STAGE)
        self.canvas.select_node(STAGE)
        self.diagnostic_button.hide()
        self.export_button.setEnabled(False)
        self.setWindowTitle(f'Native calculation — {self.recipe.title}')
        self._select_stage(STAGE)
        self.status_label.setText('Watershed debug branch added. Run the selected recipe to calculate its image.')

    def _add_morphology_roi(self):
        from seedvision.benchmarking.native_morphology_roi_recipe import (
            BASE_IDS, OPERATIONS, build_native_morphology_roi_recipe)
        from seedvision.benchmarking.native_fill_roi_recipe import (
            RECIPE_IDS as FILL_IDS, base_recipe_id as fill_base, add_morphology_roi)
        reverse = self.recipe.recipe_id in FILL_IDS and fill_base(self.recipe.recipe_id) in BASE_IDS
        if self._worker is not None or not (self.recipe.recipe_id in BASE_IDS or reverse):
            return
        selected = self._selected_recipe_with_layout()
        from seedvision.benchmarking import native_median_roi_recipe as median_roi
        self.recipe = (median_roi.add_morphology_roi(selected) if median_roi.is_median_roi_recipe(selected)
            else add_morphology_roi(selected) if reverse else build_native_morphology_roi_recipe(selected))
        self._initial_recipe = self.recipe
        self.result = self.replay_result = None
        self.graph = to_pipeline_graph(self.recipe)
        self._configure_parameters()
        stage = next(s.stage_id for s in self.recipe.stages if s.operation_id in OPERATIONS)
        self.canvas.graph = self.graph
        self.canvas.rebuild_from_graph(stage)
        self.canvas.select_node(stage)
        self.morphology_roi_button.hide()
        self._refresh_roi_buttons()
        self.diagnostic_button.hide()
        self.export_button.setEnabled(False)
        self.setWindowTitle(f'Native calculation — {self.recipe.title}')
        self._select_stage(stage)
        self.status_label.setText('Morphology ROI controls added. Use null for the whole image or '
            '[x0,y0,x1,y1] in each node input frame; exterior pixels stay unchanged.')

    def _add_fill_roi(self):
        from seedvision.benchmarking.native_fill_roi_recipe import (
            BASE_IDS, OPERATION, build_native_fill_roi_recipe)
        if self._worker is not None or self.recipe.recipe_id not in BASE_IDS:
            return
        from seedvision.benchmarking import native_median_roi_recipe as median_roi
        selected = self._selected_recipe_with_layout()
        self.recipe = (median_roi.add_fill_roi(selected) if median_roi.is_median_roi_recipe(selected)
            else build_native_fill_roi_recipe(selected))
        self._initial_recipe = self.recipe
        self.result = self.replay_result = None
        self.graph = to_pipeline_graph(self.recipe)
        self._configure_parameters()
        stage = next(s.stage_id for s in self.recipe.stages if s.operation_id == OPERATION)
        self.canvas.graph = self.graph
        self.canvas.rebuild_from_graph(stage)
        self.canvas.select_node(stage)
        self.fill_roi_button.hide()
        from seedvision.benchmarking.native_morphology_roi_recipe import BASE_IDS as MORPH_BASE_IDS
        from seedvision.benchmarking.native_fill_roi_recipe import base_recipe_id
        self.morphology_roi_button.setVisible(base_recipe_id(self.recipe.recipe_id) in MORPH_BASE_IDS)
        self._refresh_roi_buttons()
        self.diagnostic_button.hide()
        self.export_button.setEnabled(False)
        self.setWindowTitle(f'Native calculation — {self.recipe.title}')
        self._select_stage(stage)
        self.status_label.setText('Small-object ROI controls added. Use null for the whole image or '
            '[x0,y0,x1,y1] in this node input frame; objects are counted within the crop.')

    def _refresh_roi_buttons(self):
        from seedvision.benchmarking import native_median_roi_recipe as median_roi
        from seedvision.benchmarking import native_morphology_roi_recipe as morphology
        from seedvision.benchmarking import native_fill_roi_recipe as fill
        from seedvision.benchmarking import native_reader_recipe as reader
        self.reader_button.setVisible(self.recipe.recipe_id in reader.BASE_IDS
            and not reader.is_reader_recipe(self.recipe))
        self.median_roi_button.setVisible(self.recipe.recipe_id in median_roi.BASE_IDS
            and not median_roi.is_median_roi_recipe(self.recipe))
        self.fill_roi_button.setVisible(self.recipe.recipe_id in fill.BASE_IDS)
        self.morphology_roi_button.setVisible(self.recipe.recipe_id in morphology.BASE_IDS or
            self.recipe.recipe_id in fill.RECIPE_IDS and fill.base_recipe_id(self.recipe.recipe_id) in morphology.BASE_IDS)
        if median_roi.is_median_roi_recipe(self.recipe) or reader.is_reader_recipe(self.recipe):
            self.diagnostic_button.hide()

    def _add_reader(self):
        from seedvision.benchmarking import native_reader_recipe as reader
        if (self._worker is not None or self.recipe.recipe_id not in reader.BASE_IDS
                or reader.is_reader_recipe(self.recipe)):
            return
        self.recipe = reader.build_native_reader_recipe(self._selected_recipe_with_layout())
        self._initial_recipe = self.recipe
        self.result = self.replay_result = None
        self.graph = to_pipeline_graph(self.recipe)
        self._configure_parameters()
        stage = next(s.stage_id for s in self.recipe.stages if s.operation_id == reader.OPERATION)
        self.canvas.graph = self.graph
        self.canvas.rebuild_from_graph(stage)
        self.canvas.select_node(stage)
        self._refresh_roi_buttons()
        self.export_button.setEnabled(False)
        self.setWindowTitle(f'Native calculation – {self.recipe.title}')
        self._select_stage(stage)
        self.status_label.setText('Grayscale reader control added. Select color-derived or codec-gray '
            'on the image input node; both use the full decoded frame.')

    def _add_median_roi(self):
        from seedvision.benchmarking import native_median_roi_recipe as median_roi
        if (self._worker is not None or self.recipe.recipe_id not in median_roi.BASE_IDS
                or median_roi.is_median_roi_recipe(self.recipe)):
            return
        self.recipe = median_roi.build_native_median_roi_recipe(self._selected_recipe_with_layout())
        self._initial_recipe = self.recipe
        self.result = self.replay_result = None
        self.graph = to_pipeline_graph(self.recipe)
        self._configure_parameters()
        stage = next(s.stage_id for s in self.recipe.stages if s.operation_id == median_roi.OPERATION)
        self.canvas.graph = self.graph
        self.canvas.rebuild_from_graph(stage)
        self.canvas.select_node(stage)
        self._refresh_roi_buttons()
        self.export_button.setEnabled(False)
        self.setWindowTitle(f'Native calculation — {self.recipe.title}')
        self._select_stage(stage)
        self.status_label.setText('Median ROI controls added. Use null for the whole image or '
            '[x0,y0,x1,y1] in this node input frame; exterior pixels stay unchanged.')

    def _configure_parameters(self):
        self.graph.reference_recipe_identity = self.recipe.identity
        descriptions = {
            'external.image_inputs': 'Decode the selected image and prepare full-resolution RGB and grayscale pixels.',
            'filter.median_blur': 'Filter local grayscale noise before foreground selection.',
            'threshold.binary': 'Select foreground using the cutoff and light/dark polarity.',
            'threshold.mean': 'Select foreground using local mean intensity, neighborhood size and offset. Even sizes round up to odd.',
            'threshold.otsu': 'Select foreground using the supported Otsu partition and light/dark polarity.',
            'threshold.gaussian': 'Select foreground with the supported Gaussian kernel, offset and polarity. Even sizes round up to odd.',
            'threshold.triangle': 'Select foreground with the source histogram preamble and supported Otsu decision.',
            'threshold.texture': 'Select foreground from the GLCM property, kernel, offset, cutoff and border mode.',
            'threshold.dual_channels': 'Select foreground using the chosen channels and straight-line points.',
            'threshold.custom_range': 'Apply inclusive channel bounds and retain selected original RGB pixels. RGB bounds use B,G,R storage order; CMYK preserves the pinned C/Y swap.',
            'morphology.fill': 'Remove foreground components smaller than the selected pixel area.',
            'morphology.erode': 'Shrink the foreground with the selected square kernel and number of passes.',
            'morphology.dilate': 'Expand the foreground with the selected square kernel and number of passes.',
            'reference.internal.watershed.distance_map': 'Measure distance inside the prepared foreground.',
            'reference.internal.watershed.local_maxima': 'Select local distance peaks with the chosen minimum spacing.',
            'reference.internal.watershed.connected_markers': 'Group adjacent peak pixels into eight-connected markers.',
            'reference.internal.watershed.negate_distance': 'Negate the distance map to form the watershed elevation.',
            'reference.internal.watershed.masked_flood': 'Partition the original foreground using the markers and distance elevation.',
            'reference.internal.watershed.marker_count_observation': 'Report the reference object-count observation derived from connected markers.',
            'native.watershed_debug_diagnostic': 'Draw a separate watershed debug image with the source contours, colormap and an explicit reproducible shuffle seed. Palette and stroke edits reuse segmentation.',
            'analysis.color': 'Measure per-object channel distributions, circular hue and positive saturation/value summaries.',
            'analysis.size': 'Measure supported objects in pixels and draw their size diagnostic, including disconnected regions, holes and nested islands. Unsupported numerical geometry is unavailable; diagnostic resource limits preserve available measurements.',
        }
        for stage in self.recipe.stages:
            node = self.graph.nodes[stage.stage_id]
            node.description = descriptions.get(stage.operation_id, stage.description)
            specs, values = [], {}
            for parameter in stage.parameters:
                if not parameter.affects_calculation:
                    continue
                if parameter.choices:
                    kind, value = 'choice', parameter.value
                else:
                    kind = 'text'
                    value = (str(parameter.value) if parameter.value_type == 'string'
                        else json.dumps(self.recipe.parameters_for_runner(stage.stage_id)[parameter.parameter_id]))
                values[parameter.parameter_id] = value
                description = parameter.description
                if parameter.parameter_id == 'ksize' and stage.operation_id in ('threshold.mean', 'threshold.gaussian'):
                    maximum = 255 if stage.operation_id == 'threshold.mean' else 15
                    description = f'Integer requested kernel from 3 to {maximum}; even sizes round up to odd. Uncertain rounding is unavailable.'
                specs.append(ParameterSpec(parameter.parameter_id, parameter.parameter_id.replace('_', ' '),
                    kind, description=description, choices=tuple(parameter.choices)))
            node.parameters = values
            node.parameter_specs = tuple(specs)
            node.parameter_sections = (ParameterSection('Selected calculation parameters', tuple(values)),) if values else ()
            node.bypassable = False
            node.implemented = True

    def _invalidate_view(self, stage_id=None):
        dirty = set(self.graph.nodes) if stage_id is None else {stage_id}
        while True:
            expanded = dirty | {edge.target_stage for edge in self.recipe.dependencies if edge.source_stage in dirty}
            if expanded == dirty:
                break
            dirty = expanded
        for identifier in dirty:
            node = self.graph.nodes[identifier]
            node.status = NodeStatus.IDLE
            node.status_detail = 'Changed; run to update.'
            node.calculation_seconds = None
        self.canvas.refresh(tuple(dirty))
        self.result = None
        self.export_button.setEnabled(False)
        self.preview.clear()
        self._select_stage(self._selected_stage)

    def set_source_path(self, path):
        if self._worker is not None:
            return
        self.source_path = Path(path).resolve()
        self.source_label.setText(str(self.source_path))
        self.run_button.setEnabled(True)
        self._invalidate_view()
        self.status_label.setText('Image selected. Run to compute this recipe.')

    def _choose_image(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Choose seed image', '',
            'Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;All files (*)')
        if path:
            self.set_source_path(path)

    def _parameter_changed(self, stage_id, key, value):
        if self._worker is not None:
            return
        stage = self.recipe.stage(stage_id)
        parameter = next(item for item in stage.parameters if item.parameter_id == key)
        try:
            parsed = str(value) if parameter.value_type == 'string' or parameter.choices else json.loads(str(value))
            changed = replace(stage, parameters=tuple(replace(item, value=parsed) if item.parameter_id == key
                else item for item in stage.parameters))
            self.recipe = replace(self.recipe, stages=tuple(changed if item.stage_id == stage_id else item
                for item in self.recipe.stages))
        except (ValueError, TypeError) as error:
            self.status_label.setText(f'Invalid parameter value: {error}')
            self._configure_parameters()
            self.inspector.set_node(self.graph.nodes[stage_id])
            return
        self._configure_parameters()
        self.inspector.set_node(self.graph.nodes[stage_id])
        self._invalidate_view(stage_id)
        self.status_label.setText('Parameters changed. Run to update this node and its dependents.')

    def _save_recipe(self):
        path, _ = QFileDialog.getSaveFileName(self, 'Save selected recipe',
            self.recipe.recipe_id+'.seedfiddle-recipe.json', 'SeedFiddle recipes (*.seedfiddle-recipe.json)')
        if not path:
            return
        output = QSaveFile(path)
        encoded = self._selected_recipe_with_layout().to_json().encode('utf-8')
        if not output.open(QIODevice.OpenModeFlag.WriteOnly) or output.write(encoded) != len(encoded):
            output.cancelWriting()
            self.status_label.setText('Recipe save failed: '+output.errorString())
        elif not output.commit():
            self.status_label.setText('Recipe save failed: '+output.errorString())
        else:
            self.status_label.setText('Selected recipe saved. Its required values and exact source/topology identity are retained.')

    def _selected_recipe_with_layout(self):
        positions = tuple(LayoutPosition(stage.stage_id,
            float(self.canvas.node_items[stage.stage_id].pos().x()),
            float(self.canvas.node_items[stage.stage_id].pos().y())) for stage in self.recipe.stages)
        return replace(self.recipe, layout=positions)

    def _keep_recipe_in_project(self):
        if self._worker is not None or self._keep_in_project is None:
            return
        try:
            self._keep_in_project(self._selected_recipe_with_layout())
        except (ValueError, TypeError, OSError) as error:
            self.status_label.setText(f'Could not keep recipe in project: {error}')
            return
        self.status_label.setText('Working recipe kept in the project. Use Save Project to save its settings and model files.')

    def _export_conflicting_buttons(self):
        return tuple(button for name in ('choose_button', 'run_button', 'save_button',
            'export_button', 'replay_button', 'project_button', 'diagnostic_button', 'morphology_roi_button', 'fill_roi_button',
            'median_roi_button', 'reader_button',
            'choose_model_button', 'train_model_button')
            if (button := getattr(self, name, None)) is not None)

    def _export_run(self):
        if self._worker is not None or self.result is None:
            return
        parent = QFileDialog.getExistingDirectory(self, 'Export native run')
        if not parent:
            return
        destination = Path(parent)/(self.recipe.recipe_id+'-'+uuid4().hex[:12])
        self._cancel = threading.Event()
        worker = _ExportWorker(self.recipe, self.result, destination, self._cancel)
        self._worker = worker
        self._export_button_states = []
        for button in self._export_conflicting_buttons():
            self._export_button_states.append((button, button.isEnabled()))
            button.setEnabled(False)
        self.inspector.setEnabled(False)
        self.stop_button.setEnabled(True)
        worker.signals.completed.connect(self._export_completed)
        self.status_label.setText('Saving native run products and observations…')
        QThreadPool.globalInstance().start(worker)

    def _export_completed(self, result):
        worker = self._worker
        self._worker = None
        self._cancel = None
        self.stop_button.setEnabled(False)
        self.inspector.setEnabled(True)
        if self._close_requested:
            super().reject()
            return
        for button, enabled in self._export_button_states:
            button.setEnabled(enabled)
        if isinstance(result, Exception):
            self.status_label.setText('Run export cancelled.' if isinstance(result, NativeRecipeCancelled)
                else f'Run export failed: {type(result).__name__}: {result}')
        else:
            self._exported_quality = worker.quality
            self.status_label.setText(f'Native run exported: {result}')

    def _review_method_qc(self):
        if self._worker is not None:
            return
        from seedvision.ui.method_qc_dialog import MethodQCDialog
        context = self.parent()
        MethodQCDialog(self._attempt_quality or self._exported_quality, self,
            project_root=getattr(context, '_project_root', None),
            project_path=getattr(context, '_project_path', None)).exec()

    def _parameters_reset(self, stage_id):
        if self._worker is not None:
            return
        original = self._initial_recipe.stage(stage_id)
        if stage_id in self._initial_model_asset_paths:
            self.model_asset_paths[stage_id] = self._initial_model_asset_paths[stage_id]
        else:
            self.model_asset_paths.pop(stage_id, None)
        self.recipe = replace(self.recipe, stages=tuple(replace(stage, parameters=original.parameters)
            if stage.stage_id == stage_id else stage for stage in self.recipe.stages))
        self._configure_parameters()
        self.inspector.set_node(self.graph.nodes[stage_id])
        self._invalidate_view(stage_id)
        self.status_label.setText('Node parameters reset. Run to update its calculations.')

    def _choose_stage_replay(self):
        if self._worker is not None:
            return
        directory = QFileDialog.getExistingDirectory(self, 'Select exported native run for this recipe')
        if directory:
            self.start_stage_replay(directory)

    def start_stage_replay(self, directory):
        if self._worker is not None or self.recipe.stage(self._selected_stage).operation_id not in SUPPORTED_REPLAY_OPERATIONS:
            return
        self.replay_result = None
        self._cancel = threading.Event()
        worker = _ReplayWorker(self.recipe, self._selected_stage, directory, self._cancel)
        self._worker = worker
        self._replay_button_states = [(button,button.isEnabled()) for button in self._export_conflicting_buttons()]
        for button,_ in self._replay_button_states:
            button.setEnabled(False)
        self.inspector.setEnabled(False)
        self.canvas.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.preview.clear()
        self.histogram.hide()
        self.facts.setPlainText('Saved-input replay is in progress. The active run is retained.')
        self.status_label.setText('Verifying saved inputs and replaying the selected stage…')
        worker.signals.completed.connect(self._replay_completed)
        QThreadPool.globalInstance().start(worker)

    def _replay_completed(self, result):
        self._paper_crop_replay = False
        self.crop_selector.hide()
        self._worker = None
        self._cancel = None
        self.stop_button.setEnabled(False)
        self.inspector.setEnabled(True)
        self.canvas.setEnabled(True)
        if self._close_requested:
            super().reject()
            return
        for button,enabled in self._replay_button_states:
            button.setEnabled(enabled)
        if isinstance(result, Exception):
            self.status_label.setText('Saved stage replay cancelled.' if isinstance(result, NativeRecipeCancelled)
                else f'Saved stage replay failed: {type(result).__name__}: {result}')
            self.facts.setPlainText(self.status_label.text()+'\nThe active run is retained.')
            self.preview.setText('No replay output is available.')
            return
        self.replay_result = result
        self.status_label.setText(f'Saved stage replay: {result.status}. {result.reason} '
            'Agreement does not establish scientific validation.')
        compact = {name:value for name,value in result.outputs.items() if not torch.is_tensor(value)}
        summaries = []
        card_operation=str(result.metadata.get('operation_id','')).startswith('color_card.')
        recovery_operation=str(result.metadata.get('operation_id','')).startswith('recovery.')
        paper_operation=str(result.metadata.get('operation_id','')).startswith('smpts_paper.')
        if result.metadata.get('operation_id')=='recovery.clip_cores' and result.outputs:
            summaries.append('Cores clipped to valid source support; surviving source IDs are unchanged.')
        for name,value in compact.items():
            if paper_operation and name=='components':
                summaries.append(f'Eight-connected components: {len(value)}; IDs are consecutive in first-pixel order.')
            elif paper_operation and name=='selected_parent_ids':
                summaries.append(f'Selected contact parents: {len(value)}; strict bounding-box area cutoff.')
                summaries.append('IDs: '+str(tuple(value[:40]))+(' …' if len(value)>40 else ''))
            elif paper_operation and name=='crops':
                summaries.append(f'Packed original RGB crops: {len(value)}; half-open source bounds and parent lineage.')
                for row in value[:10]:
                    summaries.append(f'Component {row["label_id"]}, parent {row["parent_id"]}: '
                        f'bounds={row["bounds_xyxy"]}; shape={row["crop_shape"]}; offset={row["offset_pixels"]}.')
                if len(value)>10:summaries.append('Showing the first 10 crops; all rows and packed bytes are compared.')
            elif paper_operation and name=='disappeared_parent_ids':
                summaries.append(f'Disappeared parents: {len(value)}; IDs={tuple(value[:40])}.')
            elif paper_operation and name=='metadata':
                summaries.append('Provisional paper reconstruction; original execution and scientific validation are not established.')
                summaries.append('Crops use original RGB masked by eroded components; recovered measurement outlines are unavailable.')
            elif result.metadata.get('operation_id')=='calibrated_seed.crop' and name=='metadata':
                summaries.append(f'Calibrated seed crop (half-open parent xyxy): {value["bounds_xyxy"]}.')
                summaries.append(f'Local frame: {value["coordinate_frame"]["frame_id"]}; '
                    f'parent xy = local xy + {value["local_to_parent_xy"]}.')
                summaries.append(f'Card bounding overlap: {value["card_bounding_overlap_pixels"]} pixels; '
                    'card detection and correction are not recomputed by this replay.')
            elif recovery_operation and name=='inventory' and isinstance(value,Mapping):
                summaries.append(f'Four-connected cores: {len(value["objects"])}; '
                    f'{value["core_pixels"]} core pixels. IDs retain their first source-pixel index plus one.')
            elif recovery_operation and name=='metadata' and isinstance(value,Mapping):
                if 'bounds_xyxy' in value:
                    summaries.append(f'Valid source support (half-open xyxy): {value["bounds_xyxy"]}.')
                elif value.get('schema')=='seedfiddle.core-recovery-partition.v1':
                    summaries.append(f'Recovered partition: {value["status"]}; {len(value["objects"])} active cores; '
                        f'{value["unresolved_pixels"]} unresolved pixels; {value["interface_pixels"]} interface pixels.')
                elif value.get('schema')=='seedfiddle.core-recovery-annotation.v1':
                    permitted=sum(row['measurement_permitted'] for row in value['objects'])
                    summaries.append(f'Boundary uncertainty: radius {value["uncertainty_radius"]}; '
                        f'{value["measurement_policy"]}; {permitted}/{len(value["objects"])} cores eligible for measurements.')
                summaries.append(f'Source SHA-256: {value["binding"]["source_sha256"]}.')
            elif recovery_operation and name in ('partition','annotation'):
                continue  # The separate metadata summary describes this nested CUDA product.
            elif recovery_operation and name=='observations':
                summaries.append(f'Recovery measurements: {len(value)} cores; assigned area and bounding width/height in pixels.')
                for sample,rows in list(value.items())[:10]:
                    area=rows['assigned_area'];width=rows['bounding_width'];height=rows['bounding_height']
                    summaries.append(f'{sample}: area={area["value"]}, width={width["value"]}, height={height["value"]}; '
                        f'available={area["available"]}; reasons={area["reasons"]}.')
                if len(value)>10:summaries.append('Showing the first 10 cores; all records are compared and available in the saved result.')
            elif card_operation and name=='geometry' and isinstance(value,Mapping):
                summaries.append(f'Card geometry: {value["detected_chips"]} detected contours; '
                    f'{len(value["centers_xy"])} sampling centres; median chip '
                    f'{value["median_color_chip_width"]:.6g} × {value["median_color_chip_height"]:.6g} pixels.')
                summaries.append(f'Card bounding rectangle (ROI xywh): {value["bounding_rect"]}.')
            elif card_operation and name=='bounds':
                summaries.append(f'Card ROI (source xyxy): {tuple(value)}.')
            elif card_operation and name in ('source_matrix','target_matrix'):
                summaries.append(f'{name.replace("_"," ").capitalize()}: {len(value)} chip rows [ID,R,G,B].')
                summaries.extend(str(tuple(row)) for row in value)
            elif result.metadata.get('operation_id')=='color_card.scale' and name=='metadata':
                summaries.append('No physical chip size selected.' if value['unit'] is None else
                    f'Chip scale: {value["px_width"]:.6g} × {value["px_height"]:.6g} mm per pixel.')
            elif name=='class_masks' and isinstance(value,Mapping):
                summaries.append(f'Class masks: {len(value)} ordered classes.')
                summaries.extend(f'{str(key)[:120]}: {int(torch.count_nonzero(mask).item())} selected pixels.'
                    for key,mask in value.items() if torch.is_tensor(mask))
            elif name=='class_names' and isinstance(value,(tuple,list)):
                summaries.append('Class order: '+', '.join(str(item)[:120] for item in value))
            elif name=='model_identity' and isinstance(value,Mapping):
                summaries.append(f'Model SHA-256: {value.get("source_sha256", "")}; '
                    f'{value.get("source_bytes",0)} archived bytes; reference {value.get("reference_version","")}.')
            elif isinstance(value,ColorAnalysisResult):
                summaries.append(f'Colour observations: {value.status}; {value.colorspace}; {len(value.labels)} labels.')
                summaries.extend(f'{item.sample}: {item.pixel_count} pixels, {len(item.histograms)} histograms, '
                    f'{len(item.statistics)} statistics.' for item in value.labels)
            elif isinstance(value,HueHistogramChart):
                summaries.append(f'Hue chart: {value.status}; {len(value.series)} series.')
            elif name=='marker_count' and isinstance(value,Mapping):
                summaries.append(f'Estimated object count ({value["sample"]}): {value["value"]}')
            elif name=='observations' and isinstance(value,Mapping) and result.metadata.get('operation_id') in ('analysis.size','calibrated_seed.size_units'):
                summaries.append(f'Size observations: {len(value)} samples;17 variables per sample.')
                for sample,rows in value.items():
                    summaries.append(f'{sample}: area={rows["area"]["value"]:.6g} {rows["area"]["scale"]}; '
                        f'width={rows["width"]["value"]:.6g}, height={rows["height"]["value"]:.6g} '
                        f'{rows["width"]["scale"]}; centre={rows["center_of_mass"]["value"]}.')
            else:
                summaries.append(f'{name}: {value}')
        rendered = 'Saved-input replay; active run and historical records are unchanged.\n'
        if summaries:
            rendered += '\n'.join(summaries)+'\n'
            if not card_operation and not recovery_operation and not paper_operation and result.metadata.get('operation_id') not in ('analysis.color','analysis.size','calibrated_seed.size_units','calibrated_seed.crop',
                    'analysis.naive_bayes_classifier','analysis.kmeans_predict','analysis.kmeans_masks'):
                rendered += 'Compact outputs: '+str(compact)+'\n'
        self.facts.setPlainText(rendered+'Replay details:\n'+str(dict(result.metadata)))
        if paper_operation and result.metadata.get('operation_id') == 'smpts_paper.crops' and result.outputs:
            self._paper_crop_replay = True
            self._paper_crop_facts = self.facts.toPlainText()
            self.crop_selector.blockSignals(True)
            self.crop_selector.clear()
            self.crop_selector.addItem('Replayed full-frame final labels')
            for row in result.outputs['crops']:
                self.crop_selector.addItem(f'Replayed component {row["label_id"]} (parent {row["parent_id"]})')
            self.crop_selector.blockSignals(False)
            self.crop_selector.show()
            self._select_paper_crop(0)
            return
        chart=result.outputs.get('histogram')
        if isinstance(chart,HueHistogramChart):
            self.histogram.set_chart(chart)
            self.histogram.show()
            self.preview.hide()
            return
        self.preview.show()
        preview_port = result.metadata.get('preview_port')
        raster = result.outputs.get(preview_port)
        if preview_port=='class_masks' and isinstance(raster,Mapping):
            raster=next(iter(raster.values()),None)
        if torch.is_tensor(raster):
            semantics = next(port.semantics for port in self.recipe.stage(result.stage_id).ports
                if port.direction=='output' and port.name==preview_port)
            self._render_raster(raster,semantics)
        else:
            self.preview.setText('No image output for this replay. See the observations above.')

    def start_run(self):
        if self._worker is not None or self.source_path is None:
            return
        self._cancel = threading.Event()
        worker = _RunWorker(self.recipe, self.source_path, self.cache, self._cancel, self.model_asset_paths)
        self._worker = worker
        worker.signals.completed.connect(lambda result, worker=worker: self._run_completed(worker, result))
        self.choose_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.export_button.setEnabled(False)
        self.replay_button.setEnabled(False)
        self.project_button.setEnabled(False)
        self.diagnostic_button.setEnabled(False)
        self.morphology_roi_button.setEnabled(False)
        self.fill_roi_button.setEnabled(False)
        self.median_roi_button.setEnabled(False)
        self.reader_button.setEnabled(False)
        self.inspector.setEnabled(False)
        self.status_label.setText('Computing the selected native recipe…')
        self.result = None
        self._select_stage(self._selected_stage)
        QThreadPool.globalInstance().start(worker)

    def stop(self):
        if self._cancel is not None:
            self._cancel.set()
            self.status_label.setText('Stopping after the current calculation batch…')

    def _completed(self, result):
        self._worker = None
        self._cancel = None
        self.choose_button.setEnabled(True)
        self.run_button.setEnabled(self.source_path is not None)
        self.stop_button.setEnabled(False)
        self.inspector.setEnabled(True)
        self.replay_button.setEnabled(self.recipe.stage(self._selected_stage).operation_id in SUPPORTED_REPLAY_OPERATIONS)
        self.project_button.setEnabled(self._keep_in_project is not None)
        self.diagnostic_button.setEnabled(True)
        self.morphology_roi_button.setEnabled(True)
        self.fill_roi_button.setEnabled(True)
        self.median_roi_button.setEnabled(True)
        self.reader_button.setEnabled(True)
        if self._close_requested:
            super().reject()
            return
        if isinstance(result, Exception):
            self.result = None
            self.export_button.setEnabled(False)
            self._select_stage(self._selected_stage)
            self.status_label.setText('Calculation cancelled.' if isinstance(result, NativeRecipeCancelled)
                else f'Calculation failed: {type(result).__name__}: {result}')
            return
        self.result = result
        self.export_button.setEnabled(True)
        for item in result.nodes:
            node = self.graph.nodes[item.stage_id]
            node.status = (NodeStatus.COMPLETE if item.status in ('provided', 'executed', 'empty')
                else NodeStatus.WARNING if item.status == 'partial' else NodeStatus.FAILED
                if item.status == 'failed' else NodeStatus.BLOCKED)
            node.status_detail = ('Reused · ' if item.reused else '')+item.status
            if item.reason:
                node.status_detail += ': '+item.reason
            node.calculation_seconds = self.cache.node_timings_seconds.get(item.stage_id)
        self.canvas.refresh(tuple(self.graph.nodes))
        self.status_label.setText(f'Native run {result.status}. Computed {len(self.cache.last_computed_nodes)} nodes; '
            f'reused {len(self.cache.last_reused_nodes)}. Select nodes for actual outputs and availability.')
        self._select_stage(self._selected_stage)

    def _run_completed(self, worker, result):
        # A queued signal from an older task cannot release or install a newer
        # task, even when both recipes have the same analytical identity.
        if not isinstance(result, Exception) and worker.cancel.is_set():
            error = NativeRecipeCancelled('Recipe calculation cancelled before result delivery.')
            worker.record_attempt(error, phase='delivery', returned_result=result, capture_completion=False)
            result = error
        if worker.quality is not None:
            self._attempt_history.append(worker.quality)
        if self._worker is not worker:
            return
        self._attempt_quality = worker.quality
        self._completed(result)
        if worker.quality_error:
            self.status_label.setText(self.status_label.text()+' Recipe QC recording failed: '+worker.quality_error)

    def _select_stage(self, stage_id):
        self._paper_crop_replay = False
        self._selected_stage = stage_id
        self.replay_button.setEnabled(self._worker is None and
            self.recipe.stage(stage_id).operation_id in SUPPORTED_REPLAY_OPERATIONS)
        self.inspector.set_node(self.graph.nodes[stage_id])
        self.histogram.set_chart(None)
        self.histogram.hide()
        self.preview.show()
        self.crop_selector.blockSignals(True)
        self.crop_selector.clear()
        self.crop_selector.blockSignals(False)
        self.crop_selector.hide()
        if self.result is None:
            self.facts.setPlainText(self.recipe.stage(stage_id).description)
            self.preview.clear()
            return
        node = self.result.node(stage_id)
        self.facts.setPlainText(f'{node.operation_id}\nStatus: {node.status}\n{node.reason}\n'
            f'Resolved parameters: {dict(node.parameters)}\nAvailable outputs: {", ".join(node.outputs)}\n'
            f'Source: {self.result.source_sha256}\nFrame: {self.result.coordinate_frame.frame_id} '
            f'{self.result.coordinate_frame.shape}')
        if node.operation_id in ('threshold.mean', 'threshold.gaussian') and node.status == 'executed':
            requested = node.parameters['ksize']
            effective = requested + int(requested % 2 == 0)
            self.facts.setPlainText(self.facts.toPlainText()+f'\nKernel: requested {requested}, effective {effective} pixels '
                '(PlantCV rounds even sizes up to odd).')
        if node.operation_id == 'threshold.otsu' and node.calculation is not None:
            self.facts.setPlainText(self.facts.toPlainText()
                +f"\nNative selected cutoff: {node.calculation['native_selected_cutoff']}"
                +f"\nComparison: {node.calculation['comparison_rule']}"
                +"\nPlantCV's public call returns only the mask; OpenCV scalar parity is not established.")
        observation = node.outputs.get('observations')
        if observation is not None and hasattr(observation, 'labels'):
            summaries = []
            for item in observation.labels:
                summaries.append(f'{item.sample}: {item.pixel_count} pixels')
                summaries.extend(f'  {value.variable}: {value.value:.6g} {value.units}'
                    + ('' if value.available else ' (unavailable)') for value in item.statistics)
                summaries.extend(f'  {value.variable}: {len(value.values)} histogram bins'
                    for value in item.histograms)
            self.facts.setPlainText(self.facts.toPlainText()+'\n'+'\n'.join(summaries))
        elif isinstance(observation, Mapping):
            summaries = []
            for sample, values in observation.items():
                summaries.append(sample)
                for variable, item in values.items():
                    value = item['value']
                    rendered = f'{value:.6g}' if type(value) is float else str(value)
                    units = '' if item['scale'] == 'none' else ' '+item['scale']
                    if node.operation_id == 'calibrated_seed.size_units' and variable in ('area','convex_hull_area'):
                        units += '²'
                    summaries.append(f'  {variable}: {rendered}{units}')
            self.facts.setPlainText(self.facts.toPlainText()+'\n'+'\n'.join(summaries))
        elif 'marker_count' in node.outputs:
            self.facts.setPlainText(self.facts.toPlainText()+'\n'+str(dict(node.outputs['marker_count'])))
        if isinstance(node.outputs.get('metadata'), Mapping):
            self.facts.setPlainText(self.facts.toPlainText()+'\n'+str(dict(node.outputs['metadata'])))
        if node.operation_id == 'smpts_paper.contacts' and 'selected_parent_ids' in node.outputs:
            self.facts.setPlainText(self.facts.toPlainText()+'\nSelected parent IDs: '
                +str(node.outputs['selected_parent_ids']))
        if node.operation_id == 'smpts_paper.crops' and 'packed_rgb' in node.outputs:
            self.crop_selector.blockSignals(True)
            self.crop_selector.addItem('Full-frame final labels')
            for row in node.outputs['crops']:
                self.crop_selector.addItem(f"Crop {row['label_id']} · parent {row['parent_id']} · {row['crop_shape'][1]}×{row['crop_shape'][0]}")
            self.crop_selector.blockSignals(False)
            self.crop_selector.show()
            self._paper_crop_facts = self.facts.toPlainText()
            self._select_paper_crop(0)
            return
        local_ports = any(port.semantics.get('frame') == 'seed_roi_xyxy'
            for port in self.recipe.stage(stage_id).ports)
        if local_ports:
            crop = next((item for item in self.result.nodes if item.operation_id == 'calibrated_seed.crop'),None)
            if crop is not None and 'metadata' in crop.outputs:
                metadata = crop.outputs['metadata']
                self.facts.setPlainText(self.facts.toPlainText()+'\nSeed frame: '
                    +str(dict(metadata['coordinate_frame']))+'\nParent xy = local xy + '
                    +str(metadata['local_to_parent_xy']))
        chart = node.outputs.get('histogram')
        if chart is not None:
            self.histogram.set_chart(chart)
            self.histogram.show()
            self.preview.hide()
            return
        ports = {port.name: port for port in self.recipe.stage(stage_id).ports if port.direction == 'output'}
        selected = next(((name, value) for name, value in node.outputs.items()
            if torch.is_tensor(value) and value.ndim in (2, 3)
            and (ports.get(name) is None or ports[name].data_type!='PeakCoordinates')
            and (tuple(value.shape[:2]) == self.result.coordinate_frame.shape
                 or ports.get(name) is not None and ports[name].semantics.get('frame') in
                    ('roi_bounds_xyxy','seed_roi_xyxy'))), None)
        raster = None if selected is None else selected[1]
        if raster is None:
            self.preview.setText('No image output for this node. See the available observations above.')
            return
        self._render_raster(raster, ports[selected[0]].semantics)

    def _select_paper_crop(self, index):
        if self.result is None or self._worker is not None or index < 0:
            return
        node = self.result.node(self._selected_stage)
        products = (self.replay_result.outputs if self._paper_crop_replay and self.replay_result is not None
            else node.outputs)
        if node.operation_id != 'smpts_paper.crops' or 'packed_rgb' not in products:
            return
        self.facts.setPlainText(self._paper_crop_facts)
        if index == 0:
            self._render_raster(products['labels'], {})
            return
        row = products['crops'][index-1]
        height, width = row['crop_shape']
        offset, pixels = row['offset_pixels'], height*width
        raster = products['packed_rgb'][3*offset:3*(offset+pixels)].reshape(height, width, 3)
        self.facts.setPlainText(self._paper_crop_facts+'\nSelected crop: '+str(dict(row)))
        self._render_raster(raster, {'color_order': 'RGB'})

    def _render_raster(self, raster, semantics):
        stride = max(1, math.ceil(raster.shape[0]/256), math.ceil(raster.shape[1]/256))
        sampled = raster[::stride, ::stride]
        if semantics.get('color_order') == 'BGR':
            sampled = sampled.flip(-1)
        if semantics.get('positive_values') == 'chip IDs10..240':
            ids = sampled.to(torch.int64)
            sampled = torch.stack(((ids*53)%255, (ids*97)%255, (ids*193)%255), -1).to(torch.uint8)
        elif sampled.dtype in (torch.float32, torch.float64):
            lower, upper = sampled.min(), sampled.max()
            sampled = ((sampled-lower)/(upper-lower).clamp_min(1e-12)*255).to(torch.uint8)
        elif sampled.dtype == torch.int32 or semantics.get('range') == '0..cluster_count-1':
            ids = sampled.to(torch.int64)
            # Distinct colours for sparse IDs throughout the recovery graph's
            # supported range; zero remains black (joint period16,193,265).
            sampled = torch.stack(((ids*53)%251, (ids*97)%253, (ids*193)%255), -1).to(torch.uint8)
        elif sampled.dtype == torch.bool:
            sampled = sampled.to(torch.uint8)*255
        elif sampled.dtype == torch.uint8 and semantics.get('positive_value') == 1:
            sampled = sampled*255
        array = np.ascontiguousarray(sampled.cpu().numpy())
        format = QImage.Format.Format_RGB888 if array.ndim == 3 else QImage.Format.Format_Grayscale8
        image = QImage(array.data, array.shape[1], array.shape[0], array.strides[0], format).copy()
        preview_width = min(400, max(1, self.preview.width()))
        preview_height = min(300, max(1, self.preview.height()))
        self.preview.setPixmap(QPixmap.fromImage(image).scaled(
            preview_width, preview_height, Qt.AspectRatioMode.KeepAspectRatio))

    def _request_close(self):
        if self._worker is None:
            return False
        self._close_requested = True
        self.stop()
        return True

    def reject(self):
        if not self._request_close():
            super().reject()

    def accept(self):
        if not self._request_close():
            super().accept()

    def closeEvent(self, event):
        if self._request_close():
            event.ignore()
        else:
            super().closeEvent(event)
