"""Seed Fiddle desktop interface for pipeline control and image review."""

from __future__ import annotations

import json
import weakref
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
from threading import Event

import numpy as np
from PySide6.QtCore import (
    QObject,
    QRunnable,
    QSignalBlocker,
    QSize,
    Qt,
    QThreadPool,
    Signal,
    Slot,
)
from PySide6.QtGui import QAction, QKeySequence, QPalette
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QCheckBox,
    QDoubleSpinBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressDialog,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from seedvision.pipeline import NodeStatus, build_default_pipeline
from seedvision.persistence import (
    ImageFingerprintMismatch,
    ImportedInstanceMask,
    InvalidReferenceArchive,
    InstanceMaskImportError,
    ReferenceRegionBundle,
    ReferenceRegionError,
    ReferenceRegionStore,
    load_bundled_instance_mask,
    load_corrected_instance_mask,
    read_source_raster_shape,
)
from seedvision.resources import release_host_caches, resident_bytes
from seedvision.learning.pipeline import StarDistPipelineSettings, UNetPipelineSettings
from seedvision.annotation import (
    EdgeTraceOptions,
    InstanceContinuitySummary,
    ShapeGuidedFillOptions,
    SmartFillOptions,
    summarize_instance_continuity,
)
from seedvision.segmentation import (
    AdvancedAnalysisSettings,
    AnalysisLayerSettings,
    BaselineSettings,
    CalibrationSettings,
    DishDetectionSettings,
    PipelineAnalysisCache,
    ProceduralFitOptions,
    ProceduralInstanceSettings,
    fit_procedural_settings,
    prepare_procedural_instance_inputs,
    procedural_seed_instances_from_prepared,
)
from seedvision.visualization import ADVANCED_NODE_MODES, ADVANCED_OVERLAY_LABELS
from seedvision.ui.image_view import ImageView, SUPPORTED_SUFFIXES
from seedvision.ui.reference_history import RasterUndoHistory
from seedvision.ui.learning_workflow import (
    LearningExportDialog,
    LearningTrainingDialog,
    LearningTrainingRequest,
    format_learning_audit,
)
from seedvision.ui.pipeline_canvas import PipelineCanvas
from seedvision.ui.pipeline_inspector import PipelineInspector


FALLBACK_SPECIES = (
    "Soybean",
    "Lupinus mutabilis",
    "Lupinus polyphyllus",
    "Lupinus mexicanus",
)
OVERLAY_NODE_IDS = (
    "instance_masks",
    "background_likelihood",
    "refined_background_likelihood",
    "foreground_noise_likelihood",
    "edge_gradients",
    "surface_darkness_gradients",
    "lightening_gradient_ceiling",
    "darkening_gradient_ceiling",
    "frequency_noise_masks",
    "undirected_edges",
    "directed_edges",
    "edge_ridges",
    "reference_texture_prototypes",
    "reference_edge_probability",
    "reference_edge_ridges",
    "edge_traces",
    "seed_edge_curves",
    "procedural_instances",
    "unet_instances",
    "stardist_instances",
    *ADVANCED_NODE_MODES.keys(),
)
CALIBRATION_NODE_IDS = (
    "colour_reference",
    "ruler_detection",
    "deskew_colour",
    "scale_calibration",
)
IDENTIFICATION_STAGE_NODE_IDS = (
    "seed_scale_estimation",
    "perimeter_background_reference",
    "foreground_segmentation",
)
VIEWER_NODE_MODES = {
    "raw_images": "raw_image",
    "metadata": "raw_image",
    "colour_reference": "colour_reference",
    "ruler_detection": "ruler_detection",
    "deskew_colour": "deskew_colour",
    "layout_detection": "layout_detection",
    "scale_calibration": "calibrated_image",
    "seed_scale_estimation": "seed_scale_estimation",
    "perimeter_background_reference": "perimeter_background_reference",
    "foreground_segmentation": "foreground_mask",
    "distance_candidates": "distance_candidates",
    "circle_candidates": "circle_candidates",
    "identification": "proposals",
    "output": "proposals",
    **{
        node_id: ADVANCED_NODE_MODES.get(node_id, node_id)
        for node_id in OVERLAY_NODE_IDS
    },
}
VIEWER_NODE_MODES.update(
    {
        "surface_darkness_gradients": "surface_lightening_gradient",
        "lightening_gradient_ceiling": "weak_lightening_gradient",
        "darkening_gradient_ceiling": "weak_darkening_gradient",
        "frequency_noise_masks": "darkness_noise_fine",
        "reference_texture_prototypes": "reference_texture_prototypes",
        "reference_edge_probability": "physical_edge_probability",
    }
)

OVERLAY_NODE_OWNERS = {
    "raw_image": "raw_images",
    "calibrated_image": "scale_calibration",
    "deskew_colour": "deskew_colour",
    "colour_reference": "colour_reference",
    "ruler_detection": "ruler_detection",
    "layout_detection": "layout_detection",
    "perimeter_background_reference": "perimeter_background_reference",
    "seed_scale_estimation": "seed_scale_estimation",
    "foreground_feature": "foreground_segmentation",
    "foreground_mask": "foreground_segmentation",
    "foreground_colour_gamut": "foreground_segmentation",
    "foreground_binary_mask": "foreground_segmentation",
    "distance_transform": "distance_candidates",
    "distance_candidates": "distance_candidates",
    "circle_candidates": "circle_candidates",
    "proposals": "identification",
    "instance_masks": "instance_masks",
    "background_likelihood": "background_likelihood",
    "other_colour_probability": "background_likelihood",
    "background_colour_gamut": "background_likelihood",
    "refined_background_likelihood": "refined_background_likelihood",
    "other_noise_probability": "refined_background_likelihood",
    "foreground_noise_likelihood": "foreground_noise_likelihood",
    "edge_gradients": "edge_gradients",
    "surface_lightening_gradient": "surface_darkness_gradients",
    "surface_lightening_magnitude": "surface_darkness_gradients",
    "surface_darkening_gradient": "surface_darkness_gradients",
    "surface_darkening_magnitude": "surface_darkness_gradients",
    "weak_lightening_gradient": "lightening_gradient_ceiling",
    "weak_lightening_magnitude": "lightening_gradient_ceiling",
    "weak_darkening_gradient": "darkening_gradient_ceiling",
    "weak_darkening_magnitude": "darkening_gradient_ceiling",
    "darkness_noise_fine": "frequency_noise_masks",
    "darkness_noise_medium": "frequency_noise_masks",
    "darkness_noise_coarse": "frequency_noise_masks",
    "colour_noise_fine": "frequency_noise_masks",
    "colour_noise_medium": "frequency_noise_masks",
    "colour_noise_coarse": "frequency_noise_masks",
    "undirected_edges": "undirected_edges",
    "directed_edges": "directed_edges",
    "edge_ridges": "edge_ridges",
    "reference_texture_prototypes": "reference_texture_prototypes",
    "reference_seed_surface_probability": "reference_texture_prototypes",
    "reference_background_texture_probability": "reference_texture_prototypes",
    "reference_other_texture_probability": "reference_texture_prototypes",
    "physical_edge_probability": "reference_edge_probability",
    "non_edge_probability": "reference_edge_probability",
    "reference_edge_comparison": "reference_edge_probability",
    "net_physical_edge_probability": "reference_edge_probability",
    "reference_edge_ridges": "reference_edge_ridges",
    "edge_traces": "edge_traces",
    "edge_trace_continuity": "edge_traces",
    "edge_trace_gap_confidence": "edge_traces",
    "edge_radius_confirmation": "seed_edge_curves",
    "edge_circle_fit": "seed_edge_curves",
    "edge_ellipse_fit": "seed_edge_curves",
    "edge_fit_residual": "seed_edge_curves",
    "edge_centre_votes": "seed_edge_curves",
    "edge_semantic_sides": "seed_edge_curves",
    "edge_rejections": "seed_edge_curves",
    "edge_fit_geometry": "seed_edge_curves",
    "seed_edge_curves": "seed_edge_curves",
    "procedural_seed_material": "procedural_instances",
    "procedural_seed_mask": "procedural_instances",
    "procedural_boundary_cost": "procedural_instances",
    "procedural_centres": "procedural_instances",
    "procedural_instances": "procedural_instances",
    "procedural_confidence": "procedural_instances",
    "unet_interior": "unet_instances",
    "unet_physical_boundary": "unet_instances",
    "unet_pattern_boundary": "unet_instances",
    "unet_centres": "unet_instances",
    "unet_distance": "unet_instances",
    "unet_uncertainty": "unet_instances",
    "unet_instances": "unet_instances",
    "unet_confidence": "unet_instances",
    "stardist_object_probability": "stardist_instances",
    "stardist_radial_uncertainty": "stardist_instances",
    "stardist_instances": "stardist_instances",
    "stardist_confidence": "stardist_instances",
}
for _node_id, _mode in ADVANCED_NODE_MODES.items():
    OVERLAY_NODE_OWNERS[_mode] = _node_id
OVERLAY_NODE_OWNERS.update(
    {
        "boundary_magnitude": "boundary_normals",
        "contested_pixels": "assignment_confidence",
        "flattened_grayscale": "illumination_decomposition",
        "shadow_likelihood": "illumination_decomposition",
        "highlight_likelihood": "illumination_decomposition",
        "reflectance_image": "illumination_decomposition",
        "glare_likelihood": "illumination_decomposition",
        "focus_quality": "image_quality",
        "clipped_highlights": "image_quality",
        "underexposure": "image_quality",
        "sensor_noise": "image_quality",
        "radial_coordinate": "radial_profile",
        "pattern_confidence": "pattern_decomposition",
        "colour_uncertainty": "colour_probabilities",
    }
)


def _overlay_node_owner(mode: str) -> str | None:
    if mode.startswith("colour_probability:"):
        return "colour_probabilities"
    if mode.startswith("pattern_probability:"):
        return "pattern_decomposition"
    return OVERLAY_NODE_OWNERS.get(mode)


class _AnalysisSignals(QObject):
    completed = Signal(object, int)
    failed = Signal(str, str, int)
    node_progress = Signal(str, str, str, int)


class _AnalysisTask(QRunnable):
    """Run the PyTorch tensor analysis away from the Qt GUI thread."""

    def __init__(
        self,
        path: Path,
        settings: BaselineSettings,
        calibration_settings: CalibrationSettings,
        dish_settings: DishDetectionSettings,
        layer_settings: AnalysisLayerSettings,
        advanced_settings: AdvancedAnalysisSettings,
        procedural_settings: ProceduralInstanceSettings,
        unet_settings: UNetPipelineSettings,
        stardist_settings: StarDistPipelineSettings,
        learning_root: Path,
        species: str,
        background_reference_points: tuple[tuple[float, float], ...],
        foreground_reference_points: tuple[tuple[float, float], ...],
        background_reference_mask: np.ndarray | None,
        foreground_reference_mask: np.ndarray | None,
        background_exclusion_mask: np.ndarray | None,
        foreground_exclusion_mask: np.ndarray | None,
        physical_edge_reference_mask: np.ndarray | None,
        non_edge_reference_mask: np.ndarray | None,
        seed_instance_annotations: np.ndarray | None,
        background_colour_enabled: bool,
        enabled_nodes: frozenset[str],
        pipeline_revision: int,
        node_cache: PipelineAnalysisCache,
        dirty_nodes: frozenset[str],
    ) -> None:
        super().__init__()
        self.path = path
        self.settings = settings
        self.calibration_settings = calibration_settings
        self.dish_settings = dish_settings
        self.layer_settings = layer_settings
        self.advanced_settings = advanced_settings
        self.procedural_settings = procedural_settings
        self.unet_settings = unet_settings
        self.stardist_settings = stardist_settings
        self.learning_root = Path(learning_root)
        self.species = str(species)
        self.background_reference_points = background_reference_points
        self.foreground_reference_points = foreground_reference_points
        self.background_reference_mask = (
            None
            if background_reference_mask is None
            else np.asarray(background_reference_mask, dtype=bool)
        )
        self.foreground_reference_mask = (
            None
            if foreground_reference_mask is None
            else np.asarray(foreground_reference_mask, dtype=bool)
        )
        self.background_exclusion_mask = (
            None
            if background_exclusion_mask is None
            else np.asarray(background_exclusion_mask, dtype=bool)
        )
        self.foreground_exclusion_mask = (
            None
            if foreground_exclusion_mask is None
            else np.asarray(foreground_exclusion_mask, dtype=bool)
        )
        self.physical_edge_reference_mask = (
            None
            if physical_edge_reference_mask is None
            else np.asarray(physical_edge_reference_mask, dtype=bool)
        )
        self.non_edge_reference_mask = (
            None
            if non_edge_reference_mask is None
            else np.asarray(non_edge_reference_mask, dtype=bool)
        )
        self.seed_instance_annotations = (
            None
            if seed_instance_annotations is None
            else np.asarray(seed_instance_annotations, dtype=np.uint16)
        )
        self.background_colour_enabled = background_colour_enabled
        self.enabled_nodes = enabled_nodes
        self.pipeline_revision = pipeline_revision
        self.node_cache = node_cache
        self.dirty_nodes = dirty_nodes
        self.signals = _AnalysisSignals()

    @Slot()
    def run(self) -> None:
        try:
            from seedvision.segmentation.baseline import analyze_path

            result = analyze_path(
                self.path,
                self.settings,
                self.calibration_settings,
                dish_settings=self.dish_settings,
                layer_settings=self.layer_settings,
                advanced_settings=self.advanced_settings,
                procedural_settings=self.procedural_settings,
                unet_settings=self.unet_settings,
                stardist_settings=self.stardist_settings,
                background_reference_points=self.background_reference_points,
                foreground_reference_points=self.foreground_reference_points,
                background_reference_mask=self.background_reference_mask,
                foreground_reference_mask=self.foreground_reference_mask,
                background_exclusion_mask=self.background_exclusion_mask,
                foreground_exclusion_mask=self.foreground_exclusion_mask,
                physical_edge_reference_mask=self.physical_edge_reference_mask,
                non_edge_reference_mask=self.non_edge_reference_mask,
                seed_instance_annotations=self.seed_instance_annotations,
                background_colour_enabled=self.background_colour_enabled,
                enabled_nodes=self.enabled_nodes,
                node_cache=self.node_cache,
                dirty_nodes=self.dirty_nodes,
                progress_callback=lambda node_id, state: self.signals.node_progress.emit(
                    str(self.path), node_id, state, self.pipeline_revision
                ),
                learning_root=self.learning_root,
                species=self.species,
            )
        except Exception as error:  # noqa: BLE001 - cross-thread error boundary
            self.signals.failed.emit(
                str(self.path), str(error), self.pipeline_revision
            )
            return
        self.signals.completed.emit(result, self.pipeline_revision)


class _LearningTrainingSignals(QObject):
    progress = Signal(int, int, float, float)
    completed = Signal(object)
    failed = Signal(str)
    cancelled = Signal()


class _LearningTrainingTask(QRunnable):
    """Train one learned model on the existing serial CUDA worker."""

    def __init__(self, request: LearningTrainingRequest) -> None:
        super().__init__()
        self.request = request
        self.signals = _LearningTrainingSignals()
        self._cancel_requested = Event()

    def cancel(self) -> None:
        self._cancel_requested.set()

    @Slot()
    def run(self) -> None:
        try:
            from seedvision.learning.training import TrainingCancelled, train_from_manifest

            try:
                report = train_from_manifest(
                    self.request.manifest_path,
                    self.request.output_path,
                    self.request.configuration,
                    progress_callback=lambda epoch, maximum, record: self.signals.progress.emit(
                        epoch,
                        maximum,
                        float(record["train"]["total"]),
                        float(record["validation"]["total"]),
                    ),
                    cancellation_requested=self._cancel_requested.is_set,
                )
            except TrainingCancelled:
                self.signals.cancelled.emit()
                return
        except Exception as error:  # noqa: BLE001 - cross-thread error boundary
            self.signals.failed.emit(str(error))
            return
        self.signals.completed.emit(report)


class _ProceduralFitSignals(QObject):
    progress = Signal(int, int, float)
    completed = Signal(object)
    failed = Signal(str)
    cancelled = Signal()


class _ProceduralFitTask(QRunnable):
    """Fit procedural settings against applied masks withheld as markers."""

    def __init__(
        self,
        result,
        annotations: np.ndarray,
        settings: ProceduralInstanceSettings,
        *,
        false_positive_weight: float,
        annotations_are_complete: bool,
        overreach_distance_scale_fraction: float = 0.50,
        image_key: str,
        pipeline_revision: int,
        annotation_source: np.ndarray,
    ) -> None:
        super().__init__()
        self.result = result
        self.annotations = np.asarray(annotations, dtype=np.uint16)
        self.settings = settings
        self.false_positive_weight = float(false_positive_weight)
        self.overreach_distance_scale_fraction = float(
            overreach_distance_scale_fraction
        )
        self.annotations_are_complete = bool(annotations_are_complete)
        self.image_key = str(image_key)
        self.pipeline_revision = int(pipeline_revision)
        self.annotation_source = annotation_source
        self.signals = _ProceduralFitSignals()
        self._cancel_requested = Event()

    def cancel(self) -> None:
        self._cancel_requested.set()

    @Slot()
    def run(self) -> None:
        try:
            result = self.result
            prepared = prepare_procedural_instance_inputs(
                result.layers.valid_mask,
                result.estimated_seed_diameter_px,
                foreground_probability=result.foreground_probability,
                foreground_noise_probability=(
                    result.layers.foreground_noise_likelihood
                ),
                background_probability=result.layers.background_likelihood,
                refined_background_probability=(
                    result.layers.refined_background_likelihood
                ),
                edge_magnitude=result.layers.edge_likelihood,
                edge_ridges=result.layers.edge_ridges,
                physical_edge_probability=result.layers.physical_edge_probability,
                non_edge_probability=result.layers.non_edge_probability,
                thinned_reference_edge_ridges=result.layers.reference_edge_ridges,
                oriented_edge_trace_labels=result.layers.edge_trace_labels,
                oriented_edge_trace_continuity=(
                    result.layers.edge_trace_continuity
                ),
                reference_surface_probability=(
                    result.layers.reference_seed_surface_probability
                ),
                working_maximum_dimension=(
                    self.settings.working_maximum_dimension
                ),
            )
            fit_annotations = prepared.annotation_targets(
                self.annotations,
                mask_to_valid=True,
            )
            if not np.any(fit_annotations):
                raise ValueError(
                    "No applied seed annotations overlap the valid dish region."
                )

            def evaluate(settings: ProceduralInstanceSettings):
                # The annotations are intentionally withheld here. They are the
                # target for fitting, not authoritative watershed markers.
                return procedural_seed_instances_from_prepared(
                    prepared,
                    seed_instance_annotations=None,
                    settings=settings,
                )

            report = fit_procedural_settings(
                fit_annotations,
                evaluate,
                initial_settings=self.settings,
                options=ProceduralFitOptions(
                    false_positive_weight=self.false_positive_weight,
                    false_negative_weight=1.0,
                    overreach_distance_scale_fraction=(
                        self.overreach_distance_scale_fraction
                    ),
                    annotations_are_complete=self.annotations_are_complete,
                ),
                seed_diameter_px=prepared.seed_diameter_px,
                progress=lambda trial, maximum: self.signals.progress.emit(
                    trial.evaluation_number,
                    maximum,
                    float(trial.score.loss),
                ),
                cancelled=self._cancel_requested.is_set,
            )
        except Exception as error:  # noqa: BLE001 - worker error boundary
            self.signals.failed.emit(str(error))
            return
        if report.cancelled or self._cancel_requested.is_set():
            self.signals.cancelled.emit()
            return
        self.signals.completed.emit(report)


class MainWindow(QMainWindow):
    """Main desktop window for visual pipeline control and seed review."""

    ANALYSIS_CACHE_CUDA_BUDGET_BYTES = 2 * 1024**3
    ANALYSIS_CACHE_MAX_IMAGES = 3
    REFERENCE_UNDO_LIMIT = 20

    def __init__(self, root: Path, parent=None) -> None:
        super().__init__(parent)
        self._root = root
        self._reference_region_store = ReferenceRegionStore(root)
        self._reference_region_autoload_attempted: set[str] = set()
        self._image_paths: dict[str, Path] = {}
        self._analyses: dict[str, object] = {}
        self._analysis_caches: OrderedDict[str, PipelineAnalysisCache] = OrderedDict()
        self._cache_dirty_nodes: dict[str, set[str]] = {}
        # Draft masks are edited by the viewer. Applied masks are immutable
        # analysis inputs until the user explicitly confirms the draft.
        self._draft_background_reference_masks: dict[str, np.ndarray] = {}
        self._draft_foreground_reference_masks: dict[str, np.ndarray] = {}
        self._applied_background_reference_masks: dict[str, np.ndarray] = {}
        self._applied_foreground_reference_masks: dict[str, np.ndarray] = {}
        self._draft_background_exclusion_masks: dict[str, np.ndarray] = {}
        self._draft_foreground_exclusion_masks: dict[str, np.ndarray] = {}
        self._applied_background_exclusion_masks: dict[str, np.ndarray] = {}
        self._applied_foreground_exclusion_masks: dict[str, np.ndarray] = {}
        self._draft_physical_edge_reference_masks: dict[str, np.ndarray] = {}
        self._draft_non_edge_reference_masks: dict[str, np.ndarray] = {}
        self._applied_physical_edge_reference_masks: dict[str, np.ndarray] = {}
        self._applied_non_edge_reference_masks: dict[str, np.ndarray] = {}
        self._reference_masks_dirty: set[str] = set()
        self._reference_dirty_classes: dict[str, set[str]] = {}
        self._draft_instance_annotations: dict[str, np.ndarray] = {}
        self._applied_instance_annotations: dict[str, np.ndarray] = {}
        self._draft_instance_annotation_origins: dict[str, str] = {}
        self._applied_instance_annotation_origins: dict[str, str] = {}
        self._instance_annotations_dirty: set[str] = set()
        self._instance_continuity_cache: dict[
            str,
            tuple[weakref.ReferenceType[np.ndarray], InstanceContinuitySummary],
        ] = {}
        self._reference_undo_histories: dict[str, RasterUndoHistory] = {}
        self._instance_undo_histories: dict[str, RasterUndoHistory] = {}
        self._active_tasks: dict[str, _AnalysisTask] = {}
        self._learning_training_task: _LearningTrainingTask | None = None
        self._learning_training_progress: QProgressDialog | None = None
        self._procedural_fit_task: _ProceduralFitTask | None = None
        self._procedural_fit_progress: QProgressDialog | None = None
        self._pending_analysis_key: str | None = None
        self._thread_pool = QThreadPool(self)
        # A single CUDA context gains no useful throughput from full-image jobs
        # competing in parallel, while their peak allocations readily add up to
        # an out-of-memory failure on an 8 GiB device.
        self._thread_pool.setMaxThreadCount(1)
        self._selected_pipeline_node = "seed_scale_estimation"
        self._selecting_node_from_overlay = False
        self._selecting_overlay_from_node = False
        self.pipeline = build_default_pipeline()

        self.setWindowTitle("Seed Fiddle")
        self.setMinimumSize(1100, 700)
        self.resize(1540, 920)

        self.image_view = ImageView(self)
        self.image_view.image_dropped.connect(self._add_and_open_image)
        self.image_view.reference_mask_edited.connect(
            self._reference_mask_edited
        )
        self.image_view.instance_annotations_edited.connect(
            self._instance_annotations_edited
        )
        self.image_view.instance_tool_status.connect(self.statusBar().showMessage)
        self.image_view.shape_fill_size_preference_changed.connect(
            self._shape_fill_size_preference_changed
        )
        self.pipeline_canvas = PipelineCanvas(self.pipeline, self)
        self.pipeline_canvas.node_selected.connect(self._pipeline_node_selected)
        self.pipeline_canvas.unused_node_restored.connect(
            self._pipeline_unused_node_restored
        )
        self.pipeline_canvas.unused_node_shelved.connect(
            self._pipeline_unused_node_shelved
        )
        self.pipeline_canvas.parameter_changed.connect(
            self._pipeline_parameter_changed
        )
        self.pipeline_canvas.connections_changed.connect(
            self._pipeline_connections_changed
        )
        self.pipeline_canvas.connection_error.connect(
            self.statusBar().showMessage
        )
        self.pipeline_inspector = PipelineInspector(self)
        self.pipeline_inspector.parameter_changed.connect(
            self._pipeline_parameter_changed
        )
        self.pipeline_inspector.enabled_changed.connect(self._pipeline_enabled_changed)
        self.pipeline_inspector.parameters_reset.connect(
            self._pipeline_parameters_reset
        )
        self.pipeline_inspector.overlay_selected.connect(
            self._inspector_overlay_selected
        )
        self.pipeline_inspector.node_action_requested.connect(
            self._pipeline_node_action_requested
        )

        self.image_list = QListWidget(self)
        self.image_list.setAlternatingRowColors(True)
        self.image_list.itemActivated.connect(self._open_list_item)
        self.species_combo = QComboBox(self)
        self.species_combo.addItems(self._load_species_names())
        self.species_combo.currentTextChanged.connect(self._species_changed)

        self._build_actions()
        self._build_overlay_controls()
        self._build_reference_panel()
        self._build_menu()
        self._build_toolbar()
        self._build_layout()
        self.pipeline_canvas.select_node(self._selected_pipeline_node)
        self._load_workspace_images()
        if self.image_view.image_path is None:
            self.statusBar().showMessage("Ready — add a laboratory image to begin.")

    def _load_species_names(self) -> tuple[str, ...]:
        try:
            payload = json.loads(
                (self._root / "config" / "traits.json").read_text(encoding="utf-8")
            )
            names = tuple(item["display_name"] for item in payload["species"])
            return names or FALLBACK_SPECIES
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            return FALLBACK_SPECIES

    def _build_actions(self) -> None:
        self.open_action = QAction("Open images…", self)
        self.open_action.setShortcut(QKeySequence.StandardKey.Open)
        self.open_action.triggered.connect(self._choose_images)

        self.save_reference_regions_action = QAction(
            "Save applied reference regions", self
        )
        self.save_reference_regions_action.setShortcut(QKeySequence.StandardKey.Save)
        self.save_reference_regions_action.setEnabled(False)
        self.save_reference_regions_action.setToolTip(
            "Save all applied material and seed-instance reference "
            "layers for automatic restoration with this unchanged image."
        )
        self.save_reference_regions_action.triggered.connect(
            self._save_reference_regions
        )

        self.undo_reference_edit_action = QAction("Undo reference edit", self)
        self.undo_reference_edit_action.setShortcut(
            QKeySequence.StandardKey.Undo
        )
        self.undo_reference_edit_action.setShortcutContext(
            Qt.ShortcutContext.WindowShortcut
        )
        self.undo_reference_edit_action.setEnabled(False)
        self.undo_reference_edit_action.triggered.connect(
            self._undo_active_reference_edit
        )

        self.analyze_action = QAction("Run active pipeline", self)
        self.analyze_action.setShortcut("Ctrl+R")
        self.analyze_action.setEnabled(False)
        self.analyze_action.triggered.connect(self._analyze_current_image)

        self.image_workspace_action = QAction("Image review", self)
        self.image_workspace_action.setShortcut("Ctrl+1")
        self.image_workspace_action.triggered.connect(self._show_image_workspace)

        self.pipeline_workspace_action = QAction("Pipeline", self)
        self.pipeline_workspace_action.setShortcut("Ctrl+2")
        self.pipeline_workspace_action.triggered.connect(self._show_pipeline_workspace)

        self.split_workspace_action = QAction("Image + pipeline", self)
        self.split_workspace_action.setShortcut("Ctrl+3")
        self.split_workspace_action.triggered.connect(self._show_split_workspace)

        self.exit_action = QAction("Exit", self)
        self.exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        self.exit_action.triggered.connect(self.close)

        self.fit_action = QAction("Fit workspace", self)
        self.fit_action.setShortcut("F")
        self.fit_action.triggered.connect(self._fit_active_workspace)

        self.actual_size_action = QAction("Actual image size", self)
        self.actual_size_action.setShortcut("1")
        self.actual_size_action.triggered.connect(self.image_view.actual_size)

        self.paint_background_action = QAction("Material references", self)
        self.paint_background_action.setToolTip(
            "Paint mutually exclusive Background, Foreground, or Other material classes."
        )
        self.paint_background_action.setCheckable(True)
        self.paint_background_action.setEnabled(False)
        self.paint_background_action.toggled.connect(
            self._background_toolbar_editing_changed
        )

        self.annotate_instances_action = QAction("Annotate seed instances", self)
        self.annotate_instances_action.setCheckable(True)
        self.annotate_instances_action.setEnabled(False)
        self.annotate_instances_action.toggled.connect(
            self._instance_annotation_editing_changed
        )

        self.export_learning_sample_action = QAction(
            "Export applied labels to learning dataset…", self
        )
        self.export_learning_sample_action.setEnabled(False)
        self.export_learning_sample_action.triggered.connect(
            self._export_learning_sample
        )

        self.load_instance_labels_action = QAction(
            "Load matching bundled reference as draft…", self
        )
        self.load_instance_labels_action.setEnabled(False)
        self.load_instance_labels_action.setToolTip(
            "Load the source-bound repository reference matching this image. If "
            "none exists, Seed Fiddle offers a corrected-coordinate mask chooser. "
            "Imported IDs remain an unapplied, undoable draft for review."
        )
        self.load_instance_labels_action.triggered.connect(
            self._load_instance_reference_mask
        )
        self.choose_instance_labels_action = QAction(
            "Choose seed-instance mask file as draft…", self
        )
        self.choose_instance_labels_action.setEnabled(False)
        self.choose_instance_labels_action.setToolTip(
            "Explicitly choose a full-resolution corrected-coordinate PNG, TIFF, "
            "or NPZ, even when a bundled reference exists for the current image."
        )
        self.choose_instance_labels_action.triggered.connect(
            self._choose_instance_mask_file
        )
        self.save_instance_labels_action = QAction(
            "Save applied seed-label mask…", self
        )
        self.save_instance_labels_action.setEnabled(False)
        self.save_instance_labels_action.triggered.connect(
            self._save_instance_labels
        )
        self.audit_learning_dataset_action = QAction(
            "Audit learning dataset…", self
        )
        self.audit_learning_dataset_action.triggered.connect(
            self._audit_learning_dataset
        )
        self.train_learning_model_action = QAction(
            "Train or refine U-Net / StarDist…", self
        )
        self.train_learning_model_action.triggered.connect(
            self._start_learning_training
        )

        self.diagnostics_action = QAction("Runtime summary", self)
        self.diagnostics_action.triggered.connect(self._show_runtime_summary)

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(self.open_action)
        file_menu.addAction(self.save_reference_regions_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)

        edit_menu = self.menuBar().addMenu("&Edit")
        edit_menu.addAction(self.undo_reference_edit_action)

        analysis_menu = self.menuBar().addMenu("&Analysis")
        analysis_menu.addAction(self.analyze_action)

        learning_menu = self.menuBar().addMenu("&Learning")
        learning_menu.addAction(self.load_instance_labels_action)
        learning_menu.addAction(self.choose_instance_labels_action)
        learning_menu.addAction(self.save_instance_labels_action)
        learning_menu.addAction(self.export_learning_sample_action)
        learning_menu.addSeparator()
        learning_menu.addAction(self.audit_learning_dataset_action)
        learning_menu.addAction(self.train_learning_model_action)

        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(self.image_workspace_action)
        view_menu.addAction(self.pipeline_workspace_action)
        view_menu.addAction(self.split_workspace_action)
        view_menu.addSeparator()
        view_menu.addAction(self.fit_action)
        view_menu.addAction(self.actual_size_action)

        help_menu = self.menuBar().addMenu("&Help")
        help_menu.addAction(self.diagnostics_action)

    def _build_overlay_controls(self) -> None:
        """Create the compact viewer controls shared by the top toolbar."""

        self.overlay_combo = QComboBox(self)
        self._overlay_entries: list[tuple[str, str]] = [
            ("Raw image", "raw_image"),
            ("Detected colour swatches", "colour_reference"),
            ("Deskewed colour image + swatches", "deskew_colour"),
            ("Detected ruler", "ruler_detection"),
            ("Calibrated image & 5 cm scale", "calibrated_image"),
            ("Detected vessel layout", "layout_detection"),
            ("Perimeter background reference", "perimeter_background_reference"),
            ("Reference seed scale", "seed_scale_estimation"),
            ("Foreground strength", "foreground_feature"),
            ("Foreground colour probability", "foreground_mask"),
            ("Accepted foreground colours (HSV)", "foreground_colour_gamut"),
            ("Foreground binary proposal mask", "foreground_binary_mask"),
            ("Distance transform", "distance_transform"),
            ("Distance-peak candidates", "distance_candidates"),
            ("Circle candidates", "circle_candidates"),
            ("Seed proposals", "proposals"),
            ("Instance colour masks", "instance_masks"),
            ("Background colour probability", "background_likelihood"),
            ("Other colour probability", "other_colour_probability"),
            ("Accepted background colours (HSV)", "background_colour_gamut"),
            ("Background noise probability", "refined_background_likelihood"),
            ("Other noise probability", "other_noise_probability"),
            ("Foreground noise probability", "foreground_noise_likelihood"),
            ("Shared edge magnitude", "edge_gradients"),
            ("Surface lightening direction", "surface_lightening_gradient"),
            ("Surface lightening magnitude", "surface_lightening_magnitude"),
            ("Surface darkening direction", "surface_darkening_gradient"),
            ("Surface darkening magnitude", "surface_darkening_magnitude"),
            ("Weak lightening direction", "weak_lightening_gradient"),
            ("Weak lightening magnitude", "weak_lightening_magnitude"),
            ("Weak darkening direction", "weak_darkening_gradient"),
            ("Weak darkening magnitude", "weak_darkening_magnitude"),
            ("Fine darkness noise", "darkness_noise_fine"),
            ("Medium darkness noise", "darkness_noise_medium"),
            ("Coarse darkness noise", "darkness_noise_coarse"),
            ("Fine colour noise", "colour_noise_fine"),
            ("Medium colour noise", "colour_noise_medium"),
            ("Coarse colour noise", "colour_noise_coarse"),
            ("Reference texture prototype collage", "reference_texture_prototypes"),
            ("Reference seed-surface probability", "reference_seed_surface_probability"),
            ("Reference background-texture probability", "reference_background_texture_probability"),
            ("Reference Other-material probability", "reference_other_texture_probability"),
            ("Edge tangent (undirected)", "undirected_edges"),
            ("Edge tangent (directed)", "directed_edges"),
            ("Thinned edge ridges", "edge_ridges"),
            ("Physical-edge probability", "physical_edge_probability"),
            ("Non-physical edge probability", "non_edge_probability"),
            (
                "Physical blue / non-physical red",
                "reference_edge_comparison",
            ),
            (
                "Net physical-edge probability",
                "net_physical_edge_probability",
            ),
            ("Thinned reference edge ridge", "reference_edge_ridges"),
            ("Oriented edge traces", "edge_traces"),
            ("Trace continuity", "edge_trace_continuity"),
            ("Trace gap confidence", "edge_trace_gap_confidence"),
            ("Radius confirmation", "edge_radius_confirmation"),
            ("Circle-fit confidence", "edge_circle_fit"),
            ("Ellipse-fit confidence", "edge_ellipse_fit"),
            ("Circle/ellipse fit residual", "edge_fit_residual"),
            ("Seed-centre votes", "edge_centre_votes"),
            ("Semantic boundary side", "edge_semantic_sides"),
            ("Rejected edge reasons", "edge_rejections"),
            ("Fitted centres and ellipses", "edge_fit_geometry"),
            ("Final seed-boundary confidence", "seed_edge_curves"),
            ("Seed-material likelihood", "procedural_seed_material"),
            ("Seed-material mask", "procedural_seed_mask"),
            ("Physical boundary cost", "procedural_boundary_cost"),
            ("Procedural centre likelihood", "procedural_centres"),
            ("Procedural seed instances", "procedural_instances"),
            ("Procedural instance confidence", "procedural_confidence"),
            ("U-Net seed interior", "unet_interior"),
            ("U-Net physical boundary", "unet_physical_boundary"),
            ("U-Net coat-pattern boundary", "unet_pattern_boundary"),
            ("U-Net centre likelihood", "unet_centres"),
            ("U-Net interior distance", "unet_distance"),
            ("U-Net uncertainty", "unet_uncertainty"),
            ("U-Net watershed instances", "unet_instances"),
            ("U-Net instance confidence", "unet_confidence"),
            ("StarDist object probability", "stardist_object_probability"),
            ("StarDist radial uncertainty", "stardist_radial_uncertainty"),
            ("StarDist seed instances", "stardist_instances"),
            ("StarDist instance confidence", "stardist_confidence"),
            *ADVANCED_OVERLAY_LABELS,
            ("None", "none"),
        ]
        self._rebuild_overlay_combo("raw_image")
        self.overlay_combo.setEnabled(False)
        self.overlay_combo.currentIndexChanged.connect(self._overlay_changed)

        self.overlay_opacity_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.overlay_opacity_slider.setRange(0, 100)
        self.overlay_opacity_slider.setValue(68)
        self.overlay_opacity_slider.setEnabled(False)
        self.overlay_opacity_slider.valueChanged.connect(
            self._overlay_opacity_changed
        )
        self.overlay_opacity_label = QLabel("68%", self)
        self.overlay_opacity_label.setMinimumWidth(36)

        self.hsv_value_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.hsv_value_slider.setRange(0, 100)
        self.hsv_value_slider.setValue(75)
        self.hsv_value_slider.setToolTip(
            "Brightness of the exact HSV hue/saturation slice. Lower values "
            "scan darker colours; higher values scan brighter colours."
        )
        self.hsv_value_slider.valueChanged.connect(self._hsv_value_changed)
        self.hsv_value_label = QLabel("75%", self)
        self.hsv_value_label.setMinimumWidth(36)
        self.hsv_peak_button = QToolButton(self)
        self.hsv_peak_button.setText("Peak")
        self.hsv_peak_button.setToolTip(
            "Jump to the HSV Value of the most frequent visible fitted colour mode."
        )
        self.hsv_peak_button.clicked.connect(self._hsv_peak_requested)
        self._hsv_gamut_values: dict[str, int | None] = {
            "background": None,
            "foreground": None,
        }

        # The owning node remains available to tests and accessibility tools,
        # but is no longer repeated as a visible field in the detail panel.
        self.overlay_owner_label = self._muted_label("No graph node")
        self.overlay_owner_label.hide()
        self.overlay_legend_label = self._muted_label(
            "Run the analysis to inspect its intermediate raster layers."
        )
        self.overlay_legend_label.setObjectName("overlayLegend")

    def _active_overlay_entries(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (label, mode)
            for label, mode in self._overlay_entries
            if (
                (owner := _overlay_node_owner(mode)) is None
                or self.pipeline.is_active(owner)
            )
        )

    def _rebuild_overlay_combo(self, selected_mode: str | None = None) -> None:
        """Group selectable overlays beneath disabled owning-node headings."""

        selected_mode = selected_mode or str(self.overlay_combo.currentData() or "")
        grouped: dict[str | None, list[tuple[str, str]]] = {}
        for label, mode in self._active_overlay_entries():
            grouped.setdefault(_overlay_node_owner(mode), []).append((label, mode))
        with QSignalBlocker(self.overlay_combo):
            self.overlay_combo.clear()
            for owner, entries in grouped.items():
                heading = (
                    "Viewer"
                    if owner is None
                    else self.pipeline.node(owner).title
                )
                self.overlay_combo.addItem(heading, f"__overlay_group__:{owner}")
                heading_item = self.overlay_combo.model().item(
                    self.overlay_combo.count() - 1
                )
                heading_item.setEnabled(False)
                heading_font = heading_item.font()
                heading_font.setBold(True)
                heading_item.setFont(heading_font)
                for label, mode in entries:
                    self.overlay_combo.addItem(f"    {label}", mode)
            index = self.overlay_combo.findData(selected_mode)
            if index < 0:
                index = self.overlay_combo.findData("raw_image")
            if index >= 0:
                self.overlay_combo.setCurrentIndex(index)
        if hasattr(self, "pipeline_inspector"):
            self._sync_inspector_overlay_options()

    def _overlay_options_for_node(self, node_id: str) -> tuple[tuple[str, str], ...]:
        return tuple(
            (label, mode)
            for label, mode in self._active_overlay_entries()
            if _overlay_node_owner(mode) == node_id
        )

    def _sync_inspector_overlay_options(self) -> None:
        if not hasattr(self, "pipeline_inspector"):
            return
        self.pipeline_inspector.set_overlay_options(
            self._overlay_options_for_node(self._selected_pipeline_node),
            str(self.overlay_combo.currentData() or ""),
        )

    def _build_reference_panel(self) -> None:
        """Build reference-painting and instance-annotation controls."""

        self.reference_panel = QFrame(self.image_view)
        self.reference_panel.setObjectName("referencePaintPanel")
        self.reference_panel.setMaximumWidth(390)
        self.reference_panel.setStyleSheet(
            "QFrame#referencePaintPanel { background: palette(window); "
            "border: 1px solid palette(mid); border-radius: 3px; }"
        )
        layout = QVBoxLayout(self.reference_panel)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)

        reference_panel_header = QWidget(self.reference_panel)
        reference_panel_header_layout = QHBoxLayout(reference_panel_header)
        reference_panel_header_layout.setContentsMargins(0, 0, 0, 0)
        reference_panel_header_layout.setSpacing(4)
        self.reference_panel_drag_handle = QLabel(
            "Move painting controls", reference_panel_header
        )
        self.reference_panel_drag_handle.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )
        self.reference_panel_drag_handle.setToolTip(
            "Drag this bar to reposition the painting controls over the image."
        )
        self.reference_panel_drag_handle.setStyleSheet(
            "padding: 3px; font-weight: 600; background: palette(midlight); "
            "border: 1px solid palette(mid); border-radius: 2px;"
        )
        self.reference_undo_button = QPushButton("Undo", reference_panel_header)
        self.reference_undo_button.setEnabled(False)
        self.reference_undo_button.setToolTip(
            "Undo the latest painting command for this image (Ctrl+Z). "
            f"Seed Fiddle retains the latest {self.REFERENCE_UNDO_LIMIT} edits."
        )
        self.reference_undo_button.clicked.connect(
            self._undo_active_reference_edit
        )
        reference_panel_header_layout.addWidget(
            self.reference_panel_drag_handle, 1
        )
        reference_panel_header_layout.addWidget(self.reference_undo_button)
        layout.addWidget(reference_panel_header)

        self.reference_visibility_controls = QWidget(self.reference_panel)
        reference_visibility_layout = QHBoxLayout(
            self.reference_visibility_controls
        )
        reference_visibility_layout.setContentsMargins(0, 0, 0, 0)
        reference_visibility_layout.setSpacing(7)
        reference_visibility_layout.addWidget(
            QLabel("Show:", self.reference_visibility_controls)
        )
        self.show_material_references_checkbox = QCheckBox(
            "Materials", self.reference_visibility_controls
        )
        self.show_material_references_checkbox.setChecked(True)
        self.show_material_references_checkbox.setToolTip(
            "Show or hide painted Background, Foreground, and Other material "
            "marks. This changes display only; it does not edit the references."
        )
        self.show_material_references_checkbox.toggled.connect(
            self.image_view.set_material_reference_annotations_visible
        )
        self.show_instance_annotations_checkbox = QCheckBox(
            "Seed instances", self.reference_visibility_controls
        )
        self.show_instance_annotations_checkbox.setChecked(True)
        self.show_instance_annotations_checkbox.setToolTip(
            "Show or hide all painted seed-instance labels. This changes display "
            "only; labels remain available for editing, saving, and analysis."
        )
        self.show_instance_annotations_checkbox.toggled.connect(
            self.image_view.set_instance_annotations_visible
        )
        reference_visibility_layout.addWidget(
            self.show_material_references_checkbox
        )
        reference_visibility_layout.addWidget(
            self.show_instance_annotations_checkbox
        )
        reference_visibility_layout.addStretch(1)
        layout.addWidget(self.reference_visibility_controls)

        self.reference_panel_scroll = QScrollArea(self.reference_panel)
        self.reference_panel_scroll.setWidgetResizable(True)
        self.reference_panel_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.reference_panel_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.reference_panel_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.reference_panel_contents = QWidget(self.reference_panel_scroll)
        self.reference_panel_contents.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        reference_panel_layout = QVBoxLayout(self.reference_panel_contents)
        reference_panel_layout.setContentsMargins(0, 0, 0, 0)
        reference_panel_layout.setSpacing(4)

        self.reference_controls = QWidget(self.reference_panel_contents)
        reference_layout = QVBoxLayout(self.reference_controls)
        reference_layout.setContentsMargins(0, 0, 0, 0)
        reference_layout.setSpacing(4)

        reference_layout.addWidget(self._section_label("Material references"))
        self.background_enabled_checkbox = QCheckBox(
            "Use background colour analysis", self.reference_controls
        )
        self.background_enabled_checkbox.setChecked(
            self.pipeline.node("background_likelihood").enabled
        )
        self.background_enabled_checkbox.toggled.connect(
            self._background_enabled_toggled
        )
        reference_options = QWidget(self.reference_controls)
        reference_options_layout = QHBoxLayout(reference_options)
        reference_options_layout.setContentsMargins(0, 0, 0, 0)
        reference_options_layout.setSpacing(8)
        reference_options_layout.addWidget(self.background_enabled_checkbox)
        reference_options_layout.addStretch(1)
        reference_layout.addWidget(reference_options)

        material_buttons = QWidget(self.reference_controls)
        reference_buttons = material_buttons
        reference_button_layout = QHBoxLayout(reference_buttons)
        reference_button_layout.setContentsMargins(0, 0, 0, 0)
        reference_button_layout.setSpacing(3)
        self.background_point_button = QPushButton(
            "Background", reference_buttons
        )
        self.background_point_button.setCheckable(True)
        self.background_point_button.setEnabled(False)
        self.background_point_button.toggled.connect(
            self._background_point_editing_changed
        )
        self.erase_background_points_button = QPushButton("Erase", reference_buttons)
        self.erase_background_points_button.hide()
        self.erase_background_points_button.setEnabled(False)
        self.erase_background_points_button.setToolTip(
            "Select the background-reference layer and erase from it with left-drag."
        )
        self.erase_background_points_button.clicked.connect(
            lambda: self._start_reference_eraser("background")
        )
        self.clear_background_points_button = QPushButton("Clear", reference_buttons)
        self.clear_background_points_button.hide()
        self.clear_background_points_button.setEnabled(False)
        self.clear_background_points_button.clicked.connect(
            self._clear_background_points
        )
        reference_button_layout.addWidget(self.background_point_button)
        self.background_reference_label = self._muted_label(
            "Automatic background colour selection; no painted area."
        )
        self.background_reference_label.hide()

        background_exclusion_buttons = QWidget(self.reference_controls)
        background_exclusion_layout = QHBoxLayout(background_exclusion_buttons)
        background_exclusion_layout.setContentsMargins(0, 0, 0, 0)
        self.background_exclusion_button = QPushButton(
            "Other", background_exclusion_buttons
        )
        self.background_exclusion_button.setCheckable(True)
        self.background_exclusion_button.setToolTip(
            "Mark neither seed foreground nor ordinary dish background. This is an "
            "exclusive third material class. Its learned colour distribution competes "
            "with foreground and background where it is a better match; it does not "
            "veto colours shared with either class."
        )
        self.background_exclusion_button.toggled.connect(
            self._background_exclusion_editing_changed
        )
        self.erase_background_exclusion_button = QPushButton(
            "Erase", background_exclusion_buttons
        )
        self.erase_background_exclusion_button.hide()
        self.erase_background_exclusion_button.setEnabled(False)
        self.erase_background_exclusion_button.setToolTip(
            "Select the negative-background layer and erase from it with left-drag."
        )
        self.erase_background_exclusion_button.clicked.connect(
            lambda: self._start_reference_eraser("background_exclusion")
        )
        self.clear_background_exclusion_button = QPushButton(
            "Clear", background_exclusion_buttons
        )
        self.clear_background_exclusion_button.hide()
        self.clear_background_exclusion_button.clicked.connect(
            self._clear_background_exclusion
        )
        reference_button_layout.addWidget(self.background_exclusion_button)
        self.background_exclusion_label = self._muted_label(
            "No painted negative background examples."
        )
        self.background_exclusion_label.hide()

        foreground_buttons = QWidget(self.reference_controls)
        foreground_button_layout = QHBoxLayout(foreground_buttons)
        foreground_button_layout.setContentsMargins(0, 0, 0, 0)
        self.foreground_point_button = QPushButton(
            "Foreground", foreground_buttons
        )
        self.foreground_point_button.setCheckable(True)
        self.foreground_point_button.setEnabled(False)
        self.foreground_point_button.toggled.connect(
            self._foreground_point_editing_changed
        )
        self.erase_foreground_points_button = QPushButton(
            "Erase", foreground_buttons
        )
        self.erase_foreground_points_button.hide()
        self.erase_foreground_points_button.setEnabled(False)
        self.erase_foreground_points_button.setToolTip(
            "Select the foreground-reference layer and erase from it with left-drag."
        )
        self.erase_foreground_points_button.clicked.connect(
            lambda: self._start_reference_eraser("foreground")
        )
        self.clear_foreground_points_button = QPushButton(
            "Clear", foreground_buttons
        )
        self.clear_foreground_points_button.hide()
        self.clear_foreground_points_button.setEnabled(False)
        self.clear_foreground_points_button.clicked.connect(
            self._clear_foreground_points
        )
        reference_button_layout.addWidget(self.foreground_point_button)
        reference_layout.addWidget(material_buttons)
        self.foreground_reference_label = self._muted_label(
            "No painted foreground reference."
        )
        self.foreground_reference_label.hide()

        foreground_exclusion_buttons = QWidget(self.reference_controls)
        foreground_exclusion_layout = QHBoxLayout(foreground_exclusion_buttons)
        foreground_exclusion_layout.setContentsMargins(0, 0, 0, 0)
        self.foreground_exclusion_button = QPushButton(
            "Exclude from foreground", foreground_exclusion_buttons
        )
        self.foreground_exclusion_button.hide()
        self.foreground_exclusion_button.setCheckable(True)
        self.foreground_exclusion_button.setToolTip(
            "Paint negative examples for the fitted foreground colour and texture "
            "models; painted coordinates are not forcibly zeroed."
        )
        self.foreground_exclusion_button.toggled.connect(
            self._foreground_exclusion_editing_changed
        )
        self.erase_foreground_exclusion_button = QPushButton(
            "Erase", foreground_exclusion_buttons
        )
        self.erase_foreground_exclusion_button.hide()
        self.erase_foreground_exclusion_button.setEnabled(False)
        self.erase_foreground_exclusion_button.setToolTip(
            "Select the negative-foreground layer and erase from it with left-drag."
        )
        self.erase_foreground_exclusion_button.clicked.connect(
            lambda: self._start_reference_eraser("foreground_exclusion")
        )
        self.clear_foreground_exclusion_button = QPushButton(
            "Clear", foreground_exclusion_buttons
        )
        self.clear_foreground_exclusion_button.hide()
        self.clear_foreground_exclusion_button.clicked.connect(
            self._clear_foreground_exclusion
        )
        foreground_exclusion_layout.addWidget(self.foreground_exclusion_button, 1)
        foreground_exclusion_layout.addWidget(self.erase_foreground_exclusion_button)
        foreground_exclusion_layout.addWidget(self.clear_foreground_exclusion_button)
        self.foreground_exclusion_label = self._muted_label(
            "No painted negative foreground examples."
        )
        self.foreground_exclusion_label.hide()

        # Inert compatibility controls: old archives may still contain manual
        # Physical-edge/Non-edge masks, but the retired boundary painter is no
        # longer laid out, enabled, or connected to the view.
        boundary_buttons = QWidget(self.reference_controls)
        boundary_buttons.hide()
        boundary_layout = QHBoxLayout(boundary_buttons)
        boundary_layout.setContentsMargins(0, 0, 0, 0)
        boundary_layout.setSpacing(3)
        self.physical_edge_button = QPushButton("Physical edge", boundary_buttons)
        self.physical_edge_button.setCheckable(True)
        self.physical_edge_button.setEnabled(False)
        self.non_edge_button = QPushButton("Non-edge", boundary_buttons)
        self.non_edge_button.setCheckable(True)
        self.non_edge_button.setEnabled(False)
        self.edge_snap_checkbox = QCheckBox("Snap", boundary_buttons)
        self.edge_snap_checkbox.setChecked(True)
        self.edge_snap_checkbox.setEnabled(False)
        boundary_layout.addWidget(self.physical_edge_button)
        boundary_layout.addWidget(self.non_edge_button)
        boundary_layout.addWidget(self.edge_snap_checkbox)
        snap_strength_widget = QWidget(self.reference_controls)
        snap_strength_widget.hide()
        snap_strength_layout = QHBoxLayout(snap_strength_widget)
        snap_strength_layout.setContentsMargins(0, 0, 0, 0)
        snap_strength_layout.addWidget(QLabel("Snap strength", snap_strength_widget))
        self.edge_snap_strength_slider = QSlider(
            Qt.Orientation.Horizontal, snap_strength_widget
        )
        self.edge_snap_strength_slider.setRange(0, 100)
        self.edge_snap_strength_slider.setValue(70)
        self.edge_snap_strength_slider.setToolTip(
            "How far each boundary-reference dab moves toward the strongest nearby "
            "analysed edge: 0% keeps the cursor position and 100% snaps exactly."
        )
        self.edge_snap_strength_slider.valueChanged.connect(
            lambda value: self.image_view.set_edge_reference_snap_strength(
                float(value) / 100.0
            )
        )
        self.edge_snap_strength_label = QLabel("70%", snap_strength_widget)
        self.edge_snap_strength_slider.valueChanged.connect(
            lambda value: self.edge_snap_strength_label.setText(f"{value}%")
        )
        snap_strength_layout.addWidget(self.edge_snap_strength_slider, 1)
        snap_strength_layout.addWidget(self.edge_snap_strength_label)

        brush_widget = QWidget(self.reference_controls)
        brush_layout = QHBoxLayout(brush_widget)
        brush_layout.setContentsMargins(0, 0, 0, 0)
        self.reference_brush_slider = QSlider(
            Qt.Orientation.Horizontal, brush_widget
        )
        self.reference_brush_slider.setRange(2, 200)
        self.reference_brush_slider.setValue(12)
        self.reference_brush_slider.setToolTip(
            "Full-resolution brush radius in corrected-image pixels. "
            "The viewer outline shows the exact covered area. Right-drag remains "
            "a temporary eraser shortcut in Paint mode."
        )
        self.reference_brush_slider.valueChanged.connect(
            self._reference_brush_radius_changed
        )
        self.reference_brush_label = QLabel("12 px", brush_widget)
        self.reference_brush_label.setMinimumWidth(48)
        brush_layout.addWidget(self.reference_brush_slider, 1)
        brush_layout.addWidget(self.reference_brush_label)

        brush_mode_widget = QWidget(self.reference_controls)
        brush_mode_layout = QHBoxLayout(brush_mode_widget)
        brush_mode_layout.setContentsMargins(0, 0, 0, 0)
        self.reference_paint_mode_button = QPushButton("Paint", brush_mode_widget)
        self.reference_paint_mode_button.setCheckable(True)
        self.reference_paint_mode_button.setChecked(True)
        self.reference_paint_mode_button.setToolTip(
            "Left-drag adds pixels to the selected reference mask."
        )
        self.reference_eraser_button = QPushButton("Eraser", brush_mode_widget)
        self.reference_eraser_button.setCheckable(True)
        self.reference_eraser_button.setToolTip(
            "Left-drag removes pixels from the selected reference mask. The dashed "
            "red brush outline indicates erasing."
        )
        self.reference_brush_mode_group = QButtonGroup(brush_mode_widget)
        self.reference_brush_mode_group.setExclusive(True)
        self.reference_brush_mode_group.addButton(self.reference_paint_mode_button)
        self.reference_brush_mode_group.addButton(self.reference_eraser_button)
        self.reference_eraser_button.toggled.connect(
            self._reference_eraser_toggled
        )
        self.clear_reference_layer_button = QPushButton(
            "Clear layer", brush_mode_widget
        )
        self.clear_reference_layer_button.setToolTip(
            "Clear only the currently selected material or boundary class."
        )
        self.clear_reference_layer_button.clicked.connect(
            self._clear_active_reference_layer
        )
        brush_mode_layout.addWidget(self.reference_paint_mode_button)
        brush_mode_layout.addWidget(self.reference_eraser_button)
        brush_mode_layout.addWidget(self.clear_reference_layer_button)

        brush_form = QFormLayout()
        brush_form.setVerticalSpacing(4)
        brush_form.addRow("Brush mode", brush_mode_widget)
        brush_form.addRow("Brush radius", brush_widget)
        reference_layout.addLayout(brush_form)

        confirmation_buttons = QWidget(self.reference_controls)
        confirmation_layout = QHBoxLayout(confirmation_buttons)
        confirmation_layout.setContentsMargins(0, 0, 0, 0)
        self.apply_reference_masks_button = QPushButton(
            "Apply + save", confirmation_buttons
        )
        self.apply_reference_masks_button.setEnabled(False)
        self.apply_reference_masks_button.setToolTip(
            "Confirm all edited material classes, atomically save "
            "the complete applied reference snapshot, and run only the affected "
            "pipeline nodes. Unapplied strokes are never saved."
        )
        self.apply_reference_masks_button.clicked.connect(
            self._apply_reference_masks
        )
        self.revert_reference_masks_button = QPushButton(
            "Revert", confirmation_buttons
        )
        self.revert_reference_masks_button.setEnabled(False)
        self.revert_reference_masks_button.clicked.connect(
            self._revert_reference_masks
        )
        self.save_reference_regions_button = QPushButton(
            "Save", confirmation_buttons
        )
        self.save_reference_regions_button.setEnabled(False)
        self.save_reference_regions_button.setToolTip(
            "Save all applied material and seed-instance reference "
            "layers. Seed Fiddle automatically restores them only while the source "
            "image SHA-256 remains unchanged."
        )
        self.save_reference_regions_button.clicked.connect(
            self._save_reference_regions
        )
        confirmation_layout.addWidget(self.apply_reference_masks_button, 1)
        confirmation_layout.addWidget(self.revert_reference_masks_button)
        confirmation_layout.addWidget(self.save_reference_regions_button)
        reference_layout.addWidget(confirmation_buttons)
        self.reference_confirmation_label = self._muted_label("")
        reference_layout.addWidget(self.reference_confirmation_label)

        self.instance_annotation_controls = QWidget(self.reference_panel_contents)
        instance_layout = QVBoxLayout(self.instance_annotation_controls)
        instance_layout.setContentsMargins(0, 0, 0, 0)
        instance_layout.setSpacing(5)
        instance_layout.addWidget(self._section_label("Seed instance annotations"))
        self.instance_boundary_supervision_label = self._muted_label(
            "Applied complete seed masks automatically supply physical contours "
            "and safely inset internal candidate edges as non-physical supervision."
        )
        self.instance_boundary_supervision_label.setWordWrap(True)
        self.instance_boundary_supervision_label.setToolTip(
            "Draft edits do not influence boundary learning until Apply + save."
        )
        instance_layout.addWidget(self.instance_boundary_supervision_label)
        self.instance_annotation_help_label = self._muted_label(
            "Give each seed a separate colour ID. Assisted tools preview their snapped "
            "result under the cursor and apply it on click; smart fill can start an "
            "unmarked seed or extend a partial annotation."
        )
        self.instance_annotation_help_label.setToolTip(
            self.instance_annotation_help_label.text()
        )
        self.instance_annotation_help_label.hide()

        instance_selector = QWidget(self.instance_annotation_controls)
        selector_layout = QHBoxLayout(instance_selector)
        selector_layout.setContentsMargins(0, 0, 0, 0)
        self.instance_colour_swatch = QLabel(instance_selector)
        self.instance_colour_swatch.setFixedSize(22, 22)
        self.instance_id_spin = QSpinBox(instance_selector)
        self.instance_id_spin.setRange(1, np.iinfo(np.uint16).max)
        self.instance_id_spin.setPrefix("Seed ")
        self.instance_id_spin.valueChanged.connect(self._instance_id_changed)
        self.new_instance_button = QPushButton("New seed", instance_selector)
        self.new_instance_button.clicked.connect(self._new_instance_annotation)
        selector_layout.addWidget(self.instance_colour_swatch)
        selector_layout.addWidget(self.instance_id_spin, 1)
        selector_layout.addWidget(self.new_instance_button)
        instance_layout.addWidget(instance_selector)

        self.show_selected_instance_checkbox = QCheckBox(
            "Show selected seed only", self.instance_annotation_controls
        )
        self.show_selected_instance_checkbox.setToolTip(
            "Hide the coloured marks for every other seed. Changing the Seed ID "
            "centres the image on that seed without changing the current zoom."
        )
        self.show_selected_instance_checkbox.toggled.connect(
            self._show_selected_instance_toggled
        )
        instance_layout.addWidget(self.show_selected_instance_checkbox)

        instance_edit_buttons = QWidget(self.instance_annotation_controls)
        instance_edit_layout = QHBoxLayout(instance_edit_buttons)
        instance_edit_layout.setContentsMargins(0, 0, 0, 0)
        self.clear_current_instance_button = QPushButton(
            "Clear seed", instance_edit_buttons
        )
        self.clear_current_instance_button.clicked.connect(
            self._clear_current_instance_annotation
        )
        self.clear_all_instances_button = QPushButton(
            "Clear all", instance_edit_buttons
        )
        self.clear_all_instances_button.clicked.connect(
            self._clear_all_instance_annotations
        )
        instance_edit_layout.addWidget(self.clear_current_instance_button, 1)
        instance_edit_layout.addWidget(self.clear_all_instances_button)
        instance_layout.addWidget(instance_edit_buttons)

        proposal_widget = QWidget(self.instance_annotation_controls)
        proposal_layout = QHBoxLayout(proposal_widget)
        proposal_layout.setContentsMargins(0, 0, 0, 0)
        self.instance_proposal_combo = QComboBox(proposal_widget)
        self.instance_proposal_combo.setToolTip(
            "Choose an available automatic result as an editable starting point. "
            "Predictions remain unreviewed until a person corrects every instance."
        )
        self.use_instance_proposal_button = QPushButton(
            "Use draft", proposal_widget
        )
        self.use_instance_proposal_button.setToolTip(
            "Replace the current annotation draft with the selected pipeline labels, "
            "expanded into full corrected-image coordinates."
        )
        self.use_instance_proposal_button.clicked.connect(
            self._use_instance_proposal_as_draft
        )
        proposal_layout.addWidget(self.instance_proposal_combo, 1)
        proposal_layout.addWidget(self.use_instance_proposal_button)
        proposal_form = QFormLayout()
        proposal_form.setVerticalSpacing(4)
        proposal_form.addRow("Start from result", proposal_widget)
        instance_layout.addLayout(proposal_form)

        self.load_instance_reference_button = QPushButton(
            "Load matching reference", self.instance_annotation_controls
        )
        self.load_instance_reference_button.setToolTip(
            "Load the repository's source-image-bound seed reference matching the "
            "current photograph. If none is available, a file chooser opens. IDs "
            "are preserved exactly and loaded only as an undoable draft."
        )
        self.load_instance_reference_button.clicked.connect(
            self._load_instance_reference_mask
        )
        self.choose_instance_mask_button = QPushButton(
            "Choose mask file…", self.instance_annotation_controls
        )
        self.choose_instance_mask_button.setToolTip(
            "Explicitly choose a full-resolution corrected-coordinate PNG, TIFF, "
            "or NPZ, even when a bundled reference exists for this photograph."
        )
        self.choose_instance_mask_button.clicked.connect(
            self._choose_instance_mask_file
        )
        instance_import_buttons = QWidget(self.instance_annotation_controls)
        instance_import_layout = QHBoxLayout(instance_import_buttons)
        instance_import_layout.setContentsMargins(0, 0, 0, 0)
        instance_import_layout.setSpacing(3)
        instance_import_layout.addWidget(self.load_instance_reference_button, 1)
        instance_import_layout.addWidget(self.choose_instance_mask_button)
        instance_layout.addWidget(instance_import_buttons)

        instance_brush_widget = QWidget(self.instance_annotation_controls)
        instance_brush_layout = QHBoxLayout(instance_brush_widget)
        instance_brush_layout.setContentsMargins(0, 0, 0, 0)
        self.instance_brush_slider = QSlider(
            Qt.Orientation.Horizontal, instance_brush_widget
        )
        self.instance_brush_slider.setRange(2, 200)
        self.instance_brush_slider.setValue(12)
        self.instance_brush_slider.setToolTip(
            "Full-resolution freehand brush radius. Magnetic traces are always "
            "applied as an exact one-pixel edge until a closed loop is filled."
        )
        self.instance_brush_slider.valueChanged.connect(
            self._instance_brush_radius_changed
        )
        self.instance_brush_label = QLabel("12 px", instance_brush_widget)
        self.instance_brush_label.setMinimumWidth(48)
        instance_brush_layout.addWidget(self.instance_brush_slider, 1)
        instance_brush_layout.addWidget(self.instance_brush_label)

        instance_mode_widget = QWidget(self.instance_annotation_controls)
        instance_mode_layout = QGridLayout(instance_mode_widget)
        instance_mode_layout.setContentsMargins(0, 0, 0, 0)
        instance_mode_layout.setHorizontalSpacing(3)
        instance_mode_layout.setVerticalSpacing(3)
        self.instance_paint_mode_button = QPushButton("Brush", instance_mode_widget)
        self.instance_paint_mode_button.setCheckable(True)
        self.instance_paint_mode_button.setChecked(True)
        self.instance_edge_trace_button = QPushButton("Trace edge", instance_mode_widget)
        self.instance_edge_trace_button.setCheckable(True)
        self.instance_edge_trace_button.setToolTip(
            "Click an edge anchor, move to preview the magnetic path, then click to "
            "apply an exact one-pixel segment. Return to the first anchor to fill."
        )
        self.instance_shape_guided_fill_button = QPushButton(
            "Shape fill", instance_mode_widget
        )
        self.instance_shape_guided_fill_button.setCheckable(True)
        self.instance_shape_guided_fill_button.setToolTip(
            "Fit an automatically rotated oval as an approximate maximum, validate "
            "it against a refined closed edge contour, then run Smart fill with "
            "no shape-derived inward limit, soft outward pressure, and a small hard "
            "outward cutoff. Use the wheel to resize the visible oval preference; "
            "neither preview outline is stamped."
        )
        self.instance_smart_fill_button = QPushButton("Smart fill", instance_mode_widget)
        self.instance_smart_fill_button.setCheckable(True)
        self.instance_smart_fill_button.setToolTip(
            "Preview and click a locally adaptive edge-stopped fill; no prior mark is required."
        )
        self.instance_eraser_button = QPushButton("Eraser", instance_mode_widget)
        self.instance_eraser_button.setCheckable(True)
        self.instance_eraser_button.setToolTip(
            "Erase annotation marks without changing foreground references. "
            "Right-drag is a temporary eraser shortcut."
        )
        self.instance_brush_mode_group = QButtonGroup(instance_mode_widget)
        self.instance_brush_mode_group.setExclusive(True)
        instance_tools = (
            (self.instance_paint_mode_button, "brush"),
            (self.instance_edge_trace_button, "edge_trace"),
            (self.instance_shape_guided_fill_button, "shape_guided_fill"),
            (self.instance_smart_fill_button, "smart_fill"),
            (self.instance_eraser_button, "eraser"),
        )
        for index, (button, tool) in enumerate(instance_tools):
            self.instance_brush_mode_group.addButton(button)
            button.toggled.connect(
                lambda checked, selected_tool=tool: self._instance_tool_selected(
                    selected_tool, checked
                )
            )
            instance_mode_layout.addWidget(button, index // 3, index % 3)

        instance_layout.addWidget(instance_mode_widget)

        instance_brush_form = QFormLayout()
        instance_brush_form.setVerticalSpacing(4)
        instance_brush_form.addRow("Brush radius", instance_brush_widget)
        instance_layout.addLayout(instance_brush_form)

        self.instance_tool_options_stack = QStackedWidget(
            self.instance_annotation_controls
        )
        self.instance_tool_pages: dict[str, QWidget] = {}

        brush_page = QWidget(self.instance_tool_options_stack)
        brush_page_layout = QVBoxLayout(brush_page)
        brush_page_layout.setContentsMargins(0, 0, 0, 0)
        brush_page_layout.addWidget(
            self._muted_label(
                "Freehand interior painting. Right-drag temporarily erases with any tool."
            )
        )
        self.instance_tool_pages["brush"] = brush_page
        self.instance_tool_options_stack.addWidget(brush_page)

        edge_page = QWidget(self.instance_tool_options_stack)
        edge_form = QFormLayout(edge_page)
        edge_form.setContentsMargins(0, 0, 0, 0)
        edge_form.setVerticalSpacing(4)
        edge_form.addRow(
            self._muted_label(
                "First click sets an edge anchor. Move for a live snapped preview; "
                "each later click applies a one-pixel segment. Returning to the "
                "cyan first-anchor marker closes the contour."
            )
        )
        self.edge_trace_search_spin = QSpinBox(edge_page)
        self.edge_trace_search_spin.setRange(2, 200)
        self.edge_trace_search_spin.setValue(18)
        self.edge_trace_search_spin.setSuffix(" px")
        self.edge_trace_search_spin.setToolTip(
            "Perpendicular magnetic search radius. The first-anchor closure target "
            "uses 40% of this value, clamped to 3–10 full-resolution pixels."
        )
        self.edge_trace_attraction_spin = QSpinBox(edge_page)
        self.edge_trace_attraction_spin.setRange(0, 100)
        self.edge_trace_attraction_spin.setValue(85)
        self.edge_trace_attraction_spin.setSuffix("%")
        self.edge_trace_tangent_combo = QComboBox(edge_page)
        self.edge_trace_tangent_combo.addItem("Off", "off")
        self.edge_trace_tangent_combo.addItem("Undirected", "undirected")
        self.edge_trace_tangent_combo.addItem("Directed", "directed")
        self.edge_trace_tangent_combo.setCurrentIndex(1)
        self.edge_trace_tangent_weight_spin = QSpinBox(edge_page)
        self.edge_trace_tangent_weight_spin.setRange(0, 100)
        self.edge_trace_tangent_weight_spin.setValue(45)
        self.edge_trace_tangent_weight_spin.setSuffix("%")
        self.edge_trace_smoothing_spin = QSpinBox(edge_page)
        self.edge_trace_smoothing_spin.setRange(0, 8)
        self.edge_trace_smoothing_spin.setValue(2)
        self.edge_trace_fill_closed_checkbox = QCheckBox(
            "Fill inward when trace closes", edge_page
        )
        self.edge_trace_fill_closed_checkbox.setChecked(True)
        self.edge_trace_fill_closed_checkbox.setToolTip(
            "Open segments remain exact one-image-pixel edge marks. Returning the "
            "trace to its cyan first-anchor marker previews and fills only the "
            "enclosed polygon; labels belonging to other seeds remain protected."
        )
        self.edge_trace_evidence_combo = QComboBox(edge_page)
        self._populate_annotation_edge_sources(self.edge_trace_evidence_combo)
        edge_form.addRow("Edge evidence", self.edge_trace_evidence_combo)
        edge_form.addRow("Edge search", self.edge_trace_search_spin)
        edge_form.addRow("Edge attraction", self.edge_trace_attraction_spin)
        edge_form.addRow("Edge tangents", self.edge_trace_tangent_combo)
        edge_form.addRow("Tangent influence", self.edge_trace_tangent_weight_spin)
        edge_form.addRow("Path smoothing", self.edge_trace_smoothing_spin)
        edge_form.addRow(self.edge_trace_fill_closed_checkbox)
        self.instance_tool_pages["edge_trace"] = edge_page
        self.instance_tool_options_stack.addWidget(edge_page)

        shape_fill_page = QWidget(self.instance_tool_options_stack)
        shape_fill_form = QFormLayout(shape_fill_page)
        shape_fill_form.setContentsMargins(0, 0, 0, 0)
        shape_fill_form.setVerticalSpacing(4)
        shape_fill_form.addRow(
            self._muted_label(
                "The dotted rotated oval is an approximate maximum, not a hard fill "
                "mask. A refined closed edge contour validates the fit. The shape "
                "prior imposes no inward limit, is softly penalized just outside, "
                "and stops at the outward cutoff; Smart fill still follows its "
                "selected colour and edge evidence. Weak fits are refused."
            )
        )
        self.shape_fill_shape_combo = QComboBox(shape_fill_page)
        self.shape_fill_shape_combo.addItem("Ellipse", "ellipse")
        self.shape_fill_shape_combo.addItem("Circle", "circle")
        self.shape_fill_preferred_scale_spin = QDoubleSpinBox(shape_fill_page)
        self.shape_fill_preferred_scale_spin.setRange(0.40, 2.0)
        self.shape_fill_preferred_scale_spin.setSingleStep(0.05)
        self.shape_fill_preferred_scale_spin.setDecimals(2)
        self.shape_fill_preferred_scale_spin.setValue(1.0)
        self.shape_fill_preferred_scale_spin.setSuffix(" × seed diameter")
        self.shape_fill_preferred_scale_spin.setToolTip(
            "Preferred side-to-side oval size relative to the calculated seed "
            "diameter. The edge fit may adjust around this prior. While Shape "
            "fill is active, the mouse wheel changes this value instead of zooming."
        )
        self.shape_fill_evidence_combo = QComboBox(shape_fill_page)
        self._populate_annotation_edge_sources(
            self.shape_fill_evidence_combo, include_net_physical=True
        )
        adaptive_index = self.shape_fill_evidence_combo.findData("adaptive")
        if adaptive_index >= 0:
            self.shape_fill_evidence_combo.setCurrentIndex(adaptive_index)
        self.shape_fill_auto_rotation_checkbox = QCheckBox(
            "Automatically search ellipse rotation", shape_fill_page
        )
        self.shape_fill_auto_rotation_checkbox.setChecked(True)
        self.shape_fill_auto_rotation_checkbox.setToolTip(
            "Search the full 0–180° axial orientation range. Disable to hold the "
            "manually entered ellipse rotation fixed. Circles ignore rotation."
        )
        self.shape_fill_rotation_spin = QDoubleSpinBox(shape_fill_page)
        self.shape_fill_rotation_spin.setRange(-180.0, 180.0)
        self.shape_fill_rotation_spin.setDecimals(1)
        self.shape_fill_rotation_spin.setSuffix("°")
        self.shape_fill_axis_ratio_spin = QDoubleSpinBox(shape_fill_page)
        self.shape_fill_axis_ratio_spin.setRange(1.0, 4.0)
        self.shape_fill_axis_ratio_spin.setSingleStep(0.10)
        self.shape_fill_axis_ratio_spin.setDecimals(2)
        self.shape_fill_axis_ratio_spin.setValue(2.30)
        self.shape_fill_axis_ratio_spin.setToolTip(
            "Largest major/minor axis ratio considered during the coarse rotated "
            "shape search. The result is still refined to image edges."
        )
        self.shape_fill_smoothness_spin = QSpinBox(shape_fill_page)
        self.shape_fill_smoothness_spin.setRange(0, 100)
        self.shape_fill_smoothness_spin.setValue(45)
        self.shape_fill_smoothness_spin.setSuffix("%")
        self.shape_fill_smoothness_spin.setToolTip(
            "Penalty for abrupt changes between adjacent normal offsets along the "
            "closed contour."
        )
        self.shape_fill_min_strength_spin = QSpinBox(shape_fill_page)
        self.shape_fill_min_strength_spin.setRange(0, 100)
        self.shape_fill_min_strength_spin.setValue(25)
        self.shape_fill_min_strength_spin.setSuffix("%")
        self.shape_fill_min_strength_spin.setToolTip(
            "Minimum selected edge-evidence strength counted as direct support for "
            "a refined boundary point."
        )
        self.shape_fill_min_coverage_spin = QSpinBox(shape_fill_page)
        self.shape_fill_min_coverage_spin.setRange(0, 100)
        self.shape_fill_min_coverage_spin.setValue(34)
        self.shape_fill_min_coverage_spin.setSuffix("%")
        self.shape_fill_min_coverage_spin.setToolTip(
            "Refuse the fill unless at least this fraction of refined contour points "
            "has credible edge support."
        )
        self.shape_fill_min_sector_spin = QSpinBox(shape_fill_page)
        self.shape_fill_min_sector_spin.setRange(0, 100)
        self.shape_fill_min_sector_spin.setValue(62)
        self.shape_fill_min_sector_spin.setSuffix("%")
        self.shape_fill_min_sector_spin.setToolTip(
            "Refuse edge evidence concentrated in only a few arcs; this fraction of "
            "the 16 contour sectors must contain at least one supported point."
        )
        self.shape_fill_max_gap_spin = QSpinBox(shape_fill_page)
        self.shape_fill_max_gap_spin.setRange(0, 100)
        self.shape_fill_max_gap_spin.setValue(20)
        self.shape_fill_max_gap_spin.setSuffix("%")
        self.shape_fill_max_gap_spin.setToolTip(
            "Largest unsupported continuous arc the closed-boundary optimizer may "
            "bridge. Larger gaps cause an explicit refusal."
        )
        self.shape_fill_colour_step_spin = QDoubleSpinBox(shape_fill_page)
        self.shape_fill_colour_step_spin.setRange(1.0, 100.0)
        self.shape_fill_colour_step_spin.setDecimals(1)
        self.shape_fill_colour_step_spin.setValue(18.0)
        self.shape_fill_colour_step_spin.setToolTip(
            "The same touching-pixel Lab step used by Smart fill. L uses this "
            "value and a*/b* use 72%; it is not distance from the click colour."
        )
        self.shape_fill_barrier_spin = QSpinBox(shape_fill_page)
        self.shape_fill_barrier_spin.setRange(0, 100)
        self.shape_fill_barrier_spin.setValue(58)
        self.shape_fill_barrier_spin.setSuffix("%")
        self.shape_fill_barrier_spin.setToolTip(
            "The same selected-edge threshold used by Smart fill. Growth includes "
            "the first barrier pixel so the annotation reaches a one-pixel ridge "
            "without crossing it."
        )
        self.shape_fill_pixel_limit_spin = QSpinBox(shape_fill_page)
        self.shape_fill_pixel_limit_spin.setRange(1_000, 10_000_000)
        self.shape_fill_pixel_limit_spin.setSingleStep(10_000)
        self.shape_fill_pixel_limit_spin.setValue(150_000)
        self.shape_fill_outward_half_life_spin = QDoubleSpinBox(shape_fill_page)
        self.shape_fill_outward_half_life_spin.setRange(0.005, 0.50)
        self.shape_fill_outward_half_life_spin.setSingleStep(0.005)
        self.shape_fill_outward_half_life_spin.setDecimals(3)
        self.shape_fill_outward_half_life_spin.setValue(0.05)
        self.shape_fill_outward_half_life_spin.setSuffix(" × preferred diameter")
        self.shape_fill_outward_half_life_spin.setToolTip(
            "Outside the fitted oval prior, fill support decays with this "
            "half-life. The default 0.05 diameter permits a soft one-edge-pixel "
            "correction without reverting to cursor-centred radial fall-off. It "
            "cannot exceed the hard cutoff."
        )
        self.shape_fill_outward_cutoff_spin = QDoubleSpinBox(shape_fill_page)
        self.shape_fill_outward_cutoff_spin.setRange(0.01, 0.50)
        self.shape_fill_outward_cutoff_spin.setSingleStep(0.01)
        self.shape_fill_outward_cutoff_spin.setDecimals(2)
        self.shape_fill_outward_cutoff_spin.setValue(0.10)
        self.shape_fill_outward_cutoff_spin.setSuffix(" × preferred diameter")
        self.shape_fill_outward_cutoff_spin.setToolTip(
            "Hard outward limit beyond the fitted oval prior. Candidate "
            "pixels farther than this fraction of the preferred diameter are "
            "never accepted by Shape fill. It cannot be below the soft half-life."
        )
        shape_fill_form.addRow("Shape prior", self.shape_fill_shape_combo)
        shape_fill_form.addRow(
            "Oval size preference", self.shape_fill_preferred_scale_spin
        )
        shape_fill_form.addRow("Edge evidence", self.shape_fill_evidence_combo)
        shape_fill_form.addRow(self.shape_fill_auto_rotation_checkbox)
        shape_fill_form.addRow("Ellipse rotation", self.shape_fill_rotation_spin)
        shape_fill_form.addRow("Maximum axis ratio", self.shape_fill_axis_ratio_spin)
        shape_fill_form.addRow("Boundary smoothness", self.shape_fill_smoothness_spin)
        shape_fill_form.addRow("Minimum edge strength", self.shape_fill_min_strength_spin)
        shape_fill_form.addRow("Minimum edge coverage", self.shape_fill_min_coverage_spin)
        shape_fill_form.addRow("Minimum sector coverage", self.shape_fill_min_sector_spin)
        shape_fill_form.addRow("Maximum unsupported arc", self.shape_fill_max_gap_spin)
        shape_fill_form.addRow("Neighbour colour step", self.shape_fill_colour_step_spin)
        shape_fill_form.addRow("Edge barrier threshold", self.shape_fill_barrier_spin)
        shape_fill_form.addRow(
            "Outward soft half-life", self.shape_fill_outward_half_life_spin
        )
        shape_fill_form.addRow(
            "Outward hard cutoff", self.shape_fill_outward_cutoff_spin
        )
        shape_fill_form.addRow("Maximum added pixels", self.shape_fill_pixel_limit_spin)
        self.instance_tool_pages["shape_guided_fill"] = shape_fill_page
        self.instance_tool_options_stack.addWidget(shape_fill_page)

        fill_page = QWidget(self.instance_tool_options_stack)
        fill_form = QFormLayout(fill_page)
        fill_form.setContentsMargins(0, 0, 0, 0)
        fill_form.setVerticalSpacing(4)
        fill_form.addRow(
            self._muted_label(
                "Move for a live fill preview and click to apply. An unmarked seed "
                "starts at the cursor; a partial mark supplies additional context."
            )
        )
        self.smart_fill_colour_tolerance_spin = QDoubleSpinBox(fill_page)
        self.smart_fill_colour_tolerance_spin.setRange(1.0, 100.0)
        self.smart_fill_colour_tolerance_spin.setDecimals(1)
        self.smart_fill_colour_tolerance_spin.setValue(18.0)
        self.smart_fill_colour_tolerance_spin.setToolTip(
            "Largest colour change allowed from an accepted pixel to a touching "
            "candidate pixel. L uses this value and a*/b* use 72%. Higher values "
            "cross stronger local pattern changes and may drift through gradual "
            "gradients; this is not distance from the initial click colour."
        )
        self.smart_fill_edge_stop_spin = QSpinBox(fill_page)
        self.smart_fill_edge_stop_spin.setRange(0, 100)
        self.smart_fill_edge_stop_spin.setValue(58)
        self.smart_fill_edge_stop_spin.setSuffix("%")
        self.smart_fill_edge_stop_spin.setToolTip(
            "Selected edge-evidence pixels at or above this strength stop growth. "
            "Lower values stop at weaker edges; higher values require a stronger "
            "edge and permit more growth. The fill includes the first barrier pixel "
            "so it meets the ridge without crossing it."
        )
        self.smart_fill_tunnel_combo = QComboBox(fill_page)
        for label, strength in (
            ("Off", 0.0),
            ("Gentle", 0.25),
            ("Medium", 0.50),
            ("Strong", 0.75),
            ("Maximum", 1.0),
        ):
            self.smart_fill_tunnel_combo.addItem(label, strength)
        self.smart_fill_tunnel_combo.setToolTip(
            "Optionally open one small passage through the nearest weak edge "
            "barrier. Higher strengths accept a somewhat stronger barrier and a "
            "wider local passage; other barriers are not weakened."
        )
        self.smart_fill_radius_spin = QDoubleSpinBox(fill_page)
        self.smart_fill_radius_spin.setRange(0.30, 5.0)
        self.smart_fill_radius_spin.setSingleStep(0.10)
        self.smart_fill_radius_spin.setDecimals(2)
        self.smart_fill_radius_spin.setValue(0.60)
        self.smart_fill_radius_spin.setSuffix(" × seed diameter")
        self.smart_fill_radius_spin.setToolTip(
            "Hard Euclidean limit from the cursor to every newly filled pixel. "
            "This is a radius around the cursor, not the seed's side-to-side "
            "width: 0.60 seed diameter permits a maximum span of 1.20 seed "
            "diameters when the cursor is centred. Existing pixels already "
            "labelled as this seed remain visible outside the limit."
        )
        self.smart_fill_falloff_spin = QDoubleSpinBox(fill_page)
        self.smart_fill_falloff_spin.setRange(0.05, 5.0)
        self.smart_fill_falloff_spin.setSingleStep(0.05)
        self.smart_fill_falloff_spin.setDecimals(2)
        self.smart_fill_falloff_spin.setValue(0.40)
        self.smart_fill_falloff_spin.setSuffix(" × seed diameter")
        self.smart_fill_falloff_spin.setToolTip(
            "Radial extension pressure is p(d) = 2^(-d / h), where d is "
            "distance from the cursor and h is this half-life. At one half-life "
            "the pressure is exactly 50%. Pressure multiplies the allowed "
            "neighbour Lab colour step and the continuous edge-barrier threshold, "
            "so weaker colour changes and edges stop growth farther out. Uniform "
            "no-edge areas can still reach the hard maximum distance; labels "
            "remain categorical."
        )
        self.smart_fill_pixel_limit_spin = QSpinBox(fill_page)
        self.smart_fill_pixel_limit_spin.setRange(1_000, 10_000_000)
        self.smart_fill_pixel_limit_spin.setSingleStep(10_000)
        self.smart_fill_pixel_limit_spin.setValue(150_000)
        self.smart_fill_connectivity_combo = QComboBox(fill_page)
        self.smart_fill_connectivity_combo.addItem("8 neighbours", 8)
        self.smart_fill_connectivity_combo.addItem("4 neighbours", 4)
        self.smart_fill_evidence_combo = QComboBox(fill_page)
        self._populate_annotation_edge_sources(
            self.smart_fill_evidence_combo, include_net_physical=True
        )
        fill_form.addRow("Edge evidence", self.smart_fill_evidence_combo)
        fill_form.addRow(
            "Neighbour colour step", self.smart_fill_colour_tolerance_spin
        )
        fill_form.addRow("Edge barrier threshold", self.smart_fill_edge_stop_spin)
        fill_form.addRow("Tunnelling", self.smart_fill_tunnel_combo)
        fill_form.addRow(
            "Maximum distance from cursor", self.smart_fill_radius_spin
        )
        fill_form.addRow("Fall-off half-life", self.smart_fill_falloff_spin)
        fill_form.addRow("Maximum added pixels", self.smart_fill_pixel_limit_spin)
        fill_form.addRow("Connectivity", self.smart_fill_connectivity_combo)
        self.instance_tool_pages["smart_fill"] = fill_page
        self.instance_tool_options_stack.addWidget(fill_page)

        eraser_page = QWidget(self.instance_tool_options_stack)
        eraser_page_layout = QVBoxLayout(eraser_page)
        eraser_page_layout.setContentsMargins(0, 0, 0, 0)
        eraser_page_layout.addWidget(
            self._muted_label("Erase instance labels without changing colour references.")
        )
        self.instance_tool_pages["eraser"] = eraser_page
        self.instance_tool_options_stack.addWidget(eraser_page)
        instance_layout.addWidget(self.instance_tool_options_stack)

        for control in (
            self.edge_trace_evidence_combo,
            self.edge_trace_search_spin,
            self.edge_trace_attraction_spin,
            self.edge_trace_tangent_combo,
            self.edge_trace_tangent_weight_spin,
            self.edge_trace_smoothing_spin,
            self.shape_fill_shape_combo,
            self.shape_fill_preferred_scale_spin,
            self.shape_fill_evidence_combo,
            self.shape_fill_rotation_spin,
            self.shape_fill_axis_ratio_spin,
            self.shape_fill_smoothness_spin,
            self.shape_fill_min_strength_spin,
            self.shape_fill_min_coverage_spin,
            self.shape_fill_min_sector_spin,
            self.shape_fill_max_gap_spin,
            self.shape_fill_colour_step_spin,
            self.shape_fill_barrier_spin,
            self.shape_fill_outward_half_life_spin,
            self.shape_fill_outward_cutoff_spin,
            self.shape_fill_pixel_limit_spin,
            self.smart_fill_colour_tolerance_spin,
            self.smart_fill_evidence_combo,
            self.smart_fill_edge_stop_spin,
            self.smart_fill_tunnel_combo,
            self.smart_fill_radius_spin,
            self.smart_fill_falloff_spin,
            self.smart_fill_pixel_limit_spin,
            self.smart_fill_connectivity_combo,
        ):
            signal = (
                control.currentIndexChanged
                if isinstance(control, QComboBox)
                else control.valueChanged
            )
            signal.connect(self._sync_instance_tool_settings)
        self.edge_trace_fill_closed_checkbox.toggled.connect(
            self._sync_instance_tool_settings
        )
        self.shape_fill_auto_rotation_checkbox.toggled.connect(
            self._sync_instance_tool_settings
        )
        self._sync_instance_tool_settings()

        self.instance_annotation_status_label = self._muted_label(
            "No seed instances annotated."
        )
        instance_layout.addWidget(self.instance_annotation_status_label)
        self.instance_continuity_warning_label = self._muted_label("")
        background = self.palette().color(QPalette.ColorRole.Window)
        warning_colour = "#ffc857" if background.lightnessF() < 0.50 else "#925000"
        self.instance_continuity_warning_label.setStyleSheet(
            f"color: {warning_colour}; font-weight: 600;"
        )
        self.instance_continuity_warning_label.setToolTip(
            "Seed IDs should normally form one 8-connected area. This warning "
            "does not block editing or Apply, but disconnected pieces should be "
            "reviewed for an accidental split, stray mark, or reused ID."
        )
        self.instance_continuity_warning_label.hide()
        instance_layout.addWidget(self.instance_continuity_warning_label)

        instance_confirmation = QWidget(self.instance_annotation_controls)
        instance_confirmation_layout = QHBoxLayout(instance_confirmation)
        instance_confirmation_layout.setContentsMargins(0, 0, 0, 0)
        self.apply_instance_annotations_button = QPushButton(
            "Apply + save", instance_confirmation
        )
        self.apply_instance_annotations_button.setToolTip(
            "Confirm the seed-instance draft and atomically save it with the "
            "other applied reference layers. Unapplied strokes are never saved."
        )
        self.apply_instance_annotations_button.clicked.connect(
            self._apply_instance_annotations
        )
        self.revert_instance_annotations_button = QPushButton(
            "Revert", instance_confirmation
        )
        self.revert_instance_annotations_button.clicked.connect(
            self._revert_instance_annotations
        )
        instance_confirmation_layout.addWidget(
            self.apply_instance_annotations_button, 1
        )
        instance_confirmation_layout.addWidget(
            self.revert_instance_annotations_button
        )
        instance_layout.addWidget(instance_confirmation)
        self.instance_annotation_confirmation_label = self._muted_label(
            "Applied annotations are saved automatically and then constrain the "
            "instance branch."
        )
        self.instance_annotation_confirmation_label.setToolTip(
            self.instance_annotation_confirmation_label.text()
        )
        self.instance_annotation_confirmation_label.hide()

        reference_panel_layout.addWidget(self.reference_controls)
        reference_panel_layout.addWidget(self.instance_annotation_controls)
        # The overlay may be shorter than either editor in the split workspace.
        # Preserve the editors' natural vertical layout and let the surrounding
        # scroll area reveal it instead of compressing rows into one another.
        # Keep width flexible so the contents still fit a narrow image viewer.
        self.reference_controls.setMinimumHeight(
            self.reference_controls.sizeHint().height()
        )
        self.instance_annotation_controls.setMinimumHeight(
            self.instance_annotation_controls.sizeHint().height()
        )
        self.instance_annotation_controls.hide()
        self.reference_panel_scroll.setWidget(self.reference_panel_contents)
        layout.addWidget(self.reference_panel_scroll)
        self._update_instance_colour_swatch()

        self.reference_panel.hide()
        self.image_view.set_context_panel(self.reference_panel)
        self.image_view.set_context_panel_drag_handle(
            self.reference_panel_drag_handle
        )

    def _build_toolbar(self) -> None:
        self.workflow_toolbar = QToolBar("Workflow", self)
        self.workflow_toolbar.setIconSize(QSize(18, 18))
        self.workflow_toolbar.setMovable(False)
        self.workflow_toolbar.addAction(self.open_action)
        self.workflow_toolbar.addAction(self.analyze_action)
        self.workflow_toolbar.addSeparator()
        self.workflow_toolbar.addAction(self.image_workspace_action)
        self.workflow_toolbar.addAction(self.pipeline_workspace_action)
        self.workflow_toolbar.addAction(self.split_workspace_action)
        self.workflow_toolbar.addSeparator()
        self.workflow_toolbar.addAction(self.fit_action)
        self.workflow_toolbar.addAction(self.actual_size_action)
        self.workflow_toolbar.addSeparator()

        overlay_label = QLabel("Overlay:", self.workflow_toolbar)
        overlay_label.setObjectName("overlayToolbarLabel")
        self.overlay_combo.setMinimumContentsLength(18)
        self.overlay_combo.setMaximumWidth(340)
        self.workflow_toolbar.addWidget(overlay_label)
        self.workflow_toolbar.addWidget(self.overlay_combo)
        self.workflow_toolbar.addSeparator()

        self.opacity_toolbar_label = QLabel("Opacity:", self.workflow_toolbar)
        self.opacity_toolbar_label.setObjectName("opacityToolbarLabel")
        self.overlay_opacity_slider.setMinimumWidth(90)
        self.overlay_opacity_slider.setMaximumWidth(170)
        self.opacity_toolbar_label_action = self.workflow_toolbar.addWidget(
            self.opacity_toolbar_label
        )
        self.overlay_opacity_slider_action = self.workflow_toolbar.addWidget(
            self.overlay_opacity_slider
        )
        self.overlay_opacity_label_action = self.workflow_toolbar.addWidget(
            self.overlay_opacity_label
        )

        self.hsv_value_toolbar_label = QLabel("HSV value:", self.workflow_toolbar)
        self.hsv_value_toolbar_label.setObjectName("hsvValueToolbarLabel")
        self.hsv_value_slider.setMinimumWidth(90)
        self.hsv_value_slider.setMaximumWidth(170)
        self.hsv_value_toolbar_label_action = self.workflow_toolbar.addWidget(
            self.hsv_value_toolbar_label
        )
        self.hsv_value_slider_action = self.workflow_toolbar.addWidget(
            self.hsv_value_slider
        )
        self.hsv_value_label_action = self.workflow_toolbar.addWidget(
            self.hsv_value_label
        )
        self.hsv_peak_button_action = self.workflow_toolbar.addWidget(
            self.hsv_peak_button
        )
        for action in (
            self.hsv_value_toolbar_label_action,
            self.hsv_value_slider_action,
            self.hsv_value_label_action,
            self.hsv_peak_button_action,
        ):
            action.setVisible(False)
        self.workflow_toolbar.addSeparator()
        self.workflow_toolbar.addAction(self.paint_background_action)
        self.workflow_toolbar.addAction(self.annotate_instances_action)
        self.addToolBar(self.workflow_toolbar)

    def _build_layout(self) -> None:
        self.workspace_splitter = QSplitter(Qt.Orientation.Vertical, self)
        self.workspace_splitter.addWidget(self.image_view)
        self.workspace_splitter.addWidget(self.pipeline_canvas)
        self.workspace_splitter.setStretchFactor(0, 3)
        self.workspace_splitter.setStretchFactor(1, 2)
        self.workspace_splitter.setSizes((560, 360))

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.addWidget(self._build_image_panel())
        splitter.addWidget(self.workspace_splitter)
        splitter.addWidget(self._build_metadata_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes((225, 1020, 295))
        self.setCentralWidget(splitter)

    def _build_image_panel(self) -> QWidget:
        panel = QWidget(self)
        panel.setMinimumWidth(190)
        panel.setMaximumWidth(360)
        layout = QVBoxLayout(panel)
        heading = QLabel("Images", panel)
        heading.setStyleSheet("font-size: 16px; font-weight: 600;")
        layout.addWidget(heading)
        layout.addWidget(self.image_list, 1)
        add_button = QPushButton("Add images…", panel)
        add_button.clicked.connect(self._choose_images)
        layout.addWidget(add_button)
        return panel

    def _build_metadata_panel(self) -> QScrollArea:
        content = QWidget(self)
        content.setMinimumWidth(265)
        layout = QVBoxLayout(content)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)

        heading = QLabel("Image metadata", content)
        heading.setStyleSheet("font-size: 16px; font-weight: 600;")
        layout.addWidget(heading)

        form = QFormLayout()
        form.addRow("Species", self.species_combo)
        self.filename_label = QLabel("No image selected", content)
        self.filename_label.setWordWrap(True)
        form.addRow("File", self.filename_label)
        self.dimensions_label = QLabel("—", content)
        form.addRow("Dimensions", self.dimensions_label)
        self.compute_status_label = self._muted_label("Not analysed")
        form.addRow("Analysis device", self.compute_status_label)
        layout.addLayout(form)

        # The node title identifies this section without a second heading.
        layout.addWidget(self._separator())
        layout.addWidget(self.pipeline_inspector)
        layout.addWidget(self.overlay_legend_label)

        self.calibration_section = QWidget(content)
        calibration_layout = QVBoxLayout(self.calibration_section)
        calibration_layout.setContentsMargins(0, 0, 0, 0)
        calibration_layout.addWidget(self._separator())
        calibration_layout.addWidget(self._section_label("Calibration"))
        self.dish_status_label = self._muted_label("Not analysed")
        self.colour_status_label = self._muted_label("Not analysed")
        self.ruler_status_label = self._muted_label("Not analysed")
        self.deskew_status_label = self._muted_label("Not analysed")
        self.scale_status_label = self._muted_label("Not analysed")
        self.reference_status_label = self._muted_label("Not analysed")
        calibration_form = QFormLayout()
        calibration_form.addRow("Colour card", self.colour_status_label)
        calibration_form.addRow("Ruler", self.ruler_status_label)
        calibration_form.addRow("Deskew", self.deskew_status_label)
        calibration_form.addRow("Absolute scale", self.scale_status_label)
        calibration_form.addRow("Vessel layout", self.dish_status_label)
        calibration_form.addRow("Reference seeds", self.reference_status_label)
        calibration_layout.addLayout(calibration_form)
        layout.addWidget(self.calibration_section)

        self.baseline_section = QWidget(content)
        baseline_layout = QVBoxLayout(self.baseline_section)
        baseline_layout.setContentsMargins(0, 0, 0, 0)
        baseline_layout.addWidget(self._separator())
        baseline_layout.addWidget(self._section_label("Baseline analysis"))
        self.analyze_button = QPushButton("Run active pipeline", content)
        self.analyze_button.setEnabled(False)
        self.analyze_button.clicked.connect(self._analyze_current_image)
        self.analyze_button.setVisible(False)
        self.pipeline_button = QPushButton("Open visual pipeline", content)
        self.pipeline_button.clicked.connect(self._show_pipeline_workspace)
        self.pipeline_button.setVisible(False)
        self.count_label = QLabel("—", content)
        self.count_label.setStyleSheet("font-size: 24px; font-weight: 650;")
        baseline_layout.addWidget(self.count_label)
        self.crowding_label = self._muted_label("No result")
        baseline_layout.addWidget(self.crowding_label)
        self.warning_label = self._muted_label(
            "Results are approximate proposals for correction, not validated counts."
        )
        self.warning_label.setVisible(False)
        layout.addWidget(self.baseline_section)

        layout.addStretch(1)

        self.metadata_scroll = QScrollArea(self)
        self.metadata_scroll.setWidgetResizable(True)
        self.metadata_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.metadata_scroll.setWidget(content)
        self.metadata_scroll.setMinimumWidth(285)
        self.metadata_scroll.setMaximumWidth(440)
        self.metadata_scroll.setStyleSheet(
            "QScrollBar:vertical { width: 18px; margin: 1px; "
            "background: #20262d; }"
            "QScrollBar::handle:vertical { min-height: 38px; "
            "background: #73808d; border-radius: 7px; margin: 2px; }"
            "QScrollBar::handle:vertical:hover { background: #9aabb9; }"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { "
            "height: 0px; }"
        )
        return self.metadata_scroll

    def _separator(self) -> QFrame:
        line = QFrame(self)
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        return line

    def _section_label(self, text: str) -> QLabel:
        label = QLabel(text, self)
        label.setStyleSheet("font-weight: 600;")
        return label

    def _muted_label(self, text: str) -> QLabel:
        label = QLabel(text, self)
        label.setWordWrap(True)
        background = self.palette().color(QPalette.ColorRole.Window)
        colour = "#c2cad3" if background.lightnessF() < 0.50 else "#4a5865"
        label.setStyleSheet(f"color: {colour};")
        return label

    def _load_workspace_images(self) -> None:
        image_dir = self._root / "images"
        if not image_dir.is_dir():
            return
        paths = sorted(
            path for path in image_dir.iterdir() if path.suffix.lower() in SUPPORTED_SUFFIXES
        )
        for path in paths:
            self._add_image(path, open_now=False)
        if paths:
            self._open_path(paths[0])

    def _choose_images(self) -> None:
        patterns = " ".join(f"*{suffix}" for suffix in sorted(SUPPORTED_SUFFIXES))
        filenames, _ = QFileDialog.getOpenFileNames(
            self,
            "Open laboratory images",
            "",
            f"Images ({patterns});;All files (*)",
        )
        for filename in filenames:
            self._add_image(Path(filename), open_now=False)
        if filenames:
            self._open_path(Path(filenames[-1]))

    def _add_and_open_image(self, filename: str) -> None:
        self._add_image(Path(filename), open_now=True)
        self.image_view.show()

    def _add_image(self, path: Path, *, open_now: bool) -> None:
        resolved = path.resolve()
        key = str(resolved).casefold()
        if key not in self._image_paths:
            self._image_paths[key] = resolved
            item = QListWidgetItem(resolved.name)
            item.setData(Qt.ItemDataRole.UserRole, str(resolved))
            item.setToolTip(str(resolved))
            self.image_list.addItem(item)
        if open_now:
            self._open_path(resolved)

    def _open_list_item(self, item: QListWidgetItem) -> None:
        self._open_path(Path(item.data(Qt.ItemDataRole.UserRole)))
        self.image_view.show()

    def _open_path(self, path: Path) -> None:
        succeeded, error = self.image_view.load_image(path)
        if not succeeded:
            QMessageBox.warning(self, "Could not open image", f"{path}\n\n{error}")
            return
        self.filename_label.setText(path.name)
        width, height = self.image_view.image_size or (0, 0)
        self.dimensions_label.setText(f"{width:,} × {height:,} px")
        key = str(path.resolve()).casefold()
        references_loaded = False
        if key not in self._reference_region_autoload_attempted:
            self._reference_region_autoload_attempted.add(key)
            references_loaded = self._auto_load_reference_regions(
                path, (height, width)
            )
        # load_image() intentionally clears all graphics-scene annotation
        # buffers. Restore the current per-image draft/applied state even when
        # this window already attempted disk auto-load on an earlier visit.
        self._sync_reference_masks_to_view(key, render=False)
        self.image_view.set_instance_annotations(
            self._draft_instance_annotations.get(
                key, self._applied_instance_annotations.get(key)
            ),
            copy=False,
            render=False,
        )
        self._pipeline_image_loaded(path)
        self._update_analysis_availability()
        cached = self._analyses.get(key)
        if cached is None:
            self._show_pending_result()
        else:
            if key in self._analysis_caches:
                self._analysis_caches.move_to_end(key)
            self._mark_analysis_complete(cached)
            self._show_analysis_result(cached)
        self._sync_background_controls()
        self.statusBar().showMessage(
            f"Loaded {path}"
            + (" with saved reference regions." if references_loaded else "")
        )

    def _show_pending_result(self) -> None:
        self._stop_reference_point_editing()
        self.image_view.clear_analysis()
        self.pipeline_inspector.set_analysis_result(None)
        self.overlay_combo.setEnabled(False)
        self.overlay_opacity_slider.setEnabled(False)
        self.overlay_legend_label.setText(
            "Run the analysis to inspect its intermediate raster layers."
        )
        self.dish_status_label.setText("Not analysed")
        self.colour_status_label.setText("Not analysed")
        self.ruler_status_label.setText("Not analysed")
        self.deskew_status_label.setText("Not analysed")
        self.scale_status_label.setText("Not analysed")
        self.reference_status_label.setText("Not analysed")
        self.compute_status_label.setText("Not analysed")
        self.count_label.setText("—")
        self.crowding_label.setText("No result")
        self.warning_label.setText(
            "Results are approximate proposals for correction, not validated counts."
        )
        self._sync_background_controls()

    def _baseline_settings(self) -> BaselineSettings:
        values: dict[str, object] = {}
        for node_id in (
            "seed_scale_estimation",
            "foreground_segmentation",
            "distance_candidates",
            "identification",
        ):
            if self.pipeline.is_active(node_id):
                values.update(self.pipeline.node(node_id).parameters)
        if self.pipeline.is_active("circle_candidates"):
            values.update(self.pipeline.node("circle_candidates").parameters)
        return BaselineSettings(**values)

    def _dish_settings(self) -> DishDetectionSettings:
        return DishDetectionSettings(**self.pipeline.node("layout_detection").parameters)

    def _layer_settings(self) -> AnalysisLayerSettings:
        values: dict[str, object] = {}
        for node_id in (
            "perimeter_background_reference",
            "background_likelihood",
            "refined_background_likelihood",
            "foreground_noise_likelihood",
            "edge_gradients",
            "surface_darkness_gradients",
            "lightening_gradient_ceiling",
            "darkening_gradient_ceiling",
            "frequency_noise_masks",
            "edge_ridges",
            "reference_texture_prototypes",
            "reference_edge_probability",
            "reference_edge_ridges",
            "edge_traces",
            "instance_masks",
            "seed_edge_curves",
        ):
            if self.pipeline.is_active(node_id):
                values.update(self.pipeline.node(node_id).parameters)
        return AnalysisLayerSettings(**values)

    def _advanced_settings(self) -> AdvancedAnalysisSettings:
        values: dict[str, object] = {}
        for node_id in ADVANCED_NODE_MODES:
            if self.pipeline.is_active(node_id):
                values.update(self.pipeline.node(node_id).parameters)
        return AdvancedAnalysisSettings(**values)

    def _procedural_settings(self) -> ProceduralInstanceSettings:
        if not self.pipeline.is_active("procedural_instances"):
            return ProceduralInstanceSettings()
        return ProceduralInstanceSettings(
            **self.pipeline.node("procedural_instances").parameters
        )

    def _unet_settings(self) -> UNetPipelineSettings:
        return UNetPipelineSettings(**self.pipeline.node("unet_instances").parameters)

    def _stardist_settings(self) -> StarDistPipelineSettings:
        return StarDistPipelineSettings(
            **self.pipeline.node("stardist_instances").parameters
        )

    def _calibration_settings(self) -> CalibrationSettings:
        return CalibrationSettings(
            ruler_length_mm=float(
                self.pipeline.node("ruler_detection").parameters["ruler_length_mm"]
            ),
            minor_tick_mm=float(
                self.pipeline.node("scale_calibration").parameters["minor_tick_mm"]
            ),
            max_deskew_degrees=float(
                self.pipeline.node("deskew_colour").parameters[
                    "max_deskew_degrees"
                ]
            ),
            apply_colour_balance=bool(
                self.pipeline.node("colour_reference").parameters[
                    "apply_colour_balance"
                ]
            ),
            apply_perspective_correction=bool(
                self.pipeline.node("deskew_colour").parameters[
                    "apply_perspective_correction"
                ]
            ),
            max_perspective_fraction=float(
                self.pipeline.node("deskew_colour").parameters[
                    "max_perspective_fraction"
                ]
            ),
        )

    def _analyze_current_image(
        self, *, dirty_nodes: set[str] | frozenset[str] | None = None
    ) -> None:
        path = self.image_view.image_path
        if path is None:
            return
        key = str(path.resolve()).casefold()
        pending_dirty = self._cache_dirty_nodes.setdefault(key, set())
        if dirty_nodes:
            pending_dirty.update(dirty_nodes)
        if (
            self._active_tasks
            or self._learning_training_task is not None
            or self._procedural_fit_task is not None
        ):
            # Keep only the most recently requested image. Dirty-node sets are
            # retained per image, so a coalesced rerun still has exact scope.
            self._pending_analysis_key = key
            self._update_analysis_availability()
            return
        requested_dirty = frozenset(pending_dirty)
        pending_dirty.clear()
        self._pending_analysis_key = None
        self._trim_analysis_caches(protected={key})
        node_cache = self._analysis_caches.setdefault(
            key, PipelineAnalysisCache()
        )
        self._analysis_caches.move_to_end(key)
        task = _AnalysisTask(
            path,
            self._baseline_settings(),
            self._calibration_settings(),
            self._dish_settings(),
            self._layer_settings(),
            self._advanced_settings(),
            self._procedural_settings(),
            self._unet_settings(),
            self._stardist_settings(),
            self._root,
            self.species_combo.currentText(),
            (),
            (),
            self._applied_background_reference_masks.get(key),
            self._applied_foreground_reference_masks.get(key),
            self._applied_background_exclusion_masks.get(key),
            self._applied_foreground_exclusion_masks.get(key),
            self._applied_physical_edge_reference_masks.get(key),
            self._applied_non_edge_reference_masks.get(key),
            self._applied_instance_annotations.get(key),
            self.pipeline.node("background_likelihood").enabled,
            frozenset(
                node.identifier
                for node in self.pipeline.nodes.values()
                if node.enabled
            ),
            self.pipeline.revision,
            node_cache,
            requested_dirty,
        )
        task.signals.completed.connect(self._analysis_completed)
        task.signals.failed.connect(self._analysis_failed)
        task.signals.node_progress.connect(self._analysis_node_progress)
        self._active_tasks[key] = task
        self._set_analysis_running(path, requested_dirty)
        self._update_analysis_availability()
        self.count_label.setText("Analysing…")
        self.warning_label.setText(
            "Executing the active calibration and diagnostic pipeline in the background."
        )
        self.statusBar().showMessage(f"Analysing {path.name}…")
        self._thread_pool.start(task)

    def _start_pending_analysis(self) -> None:
        """Start the latest coalesced request after the single GPU worker exits."""

        if (
            self._active_tasks
            or self._learning_training_task is not None
            or self._procedural_fit_task is not None
        ):
            return
        current_key = self._current_image_key()
        if current_key is None:
            self._pending_analysis_key = None
            return
        if (
            self._pending_analysis_key == current_key
            or bool(self._cache_dirty_nodes.get(current_key))
        ):
            self._pending_analysis_key = None
            self._analyze_current_image()

    @Slot(str, str, str, int)
    def _analysis_node_progress(
        self,
        path_text: str,
        node_id: str,
        state: str,
        pipeline_revision: int,
    ) -> None:
        """Apply worker progress to the currently visible graph on the Qt thread."""

        current_path = self.image_view.image_path
        if (
            pipeline_revision != self.pipeline.revision
            or current_path is None
            or str(current_path.resolve()).casefold()
            != str(Path(path_text).resolve()).casefold()
            or node_id not in self.pipeline.nodes
        ):
            return
        node = self.pipeline.node(node_id)
        if state == "started":
            if node.enabled and node.implemented:
                node.status = NodeStatus.RUNNING
        elif state == "completed":
            if node.enabled and node.implemented:
                node.status = NodeStatus.COMPLETE
                node.status_detail = (
                    Path(path_text).name
                    if node_id == "raw_images"
                    else "Calculated"
                )
        else:
            return
        self.pipeline_canvas.refresh((node_id,))
        if self._selected_pipeline_node == node_id:
            self.pipeline_inspector.refresh_status()

    @Slot(object, int)
    def _analysis_completed(self, result, pipeline_revision: int) -> None:
        path = result.image_path
        key = str(path.resolve()).casefold() if path is not None else ""
        self._active_tasks.pop(key, None)
        if pipeline_revision != self.pipeline.revision:
            self.statusBar().showMessage(
                "Discarded an analysis completed with superseded pipeline settings."
            )
            self._update_analysis_availability()
            if self.image_view.image_path == path:
                self._pending_analysis_key = key
            self._start_pending_analysis()
            return
        # Display-only controls deliberately do not supersede an in-flight
        # analytical task. Normalize its immutable wrapper to the latest live
        # presentation value before caching it, so an older worker snapshot
        # cannot visually undo a slider edit when it completes.
        result = self._normalize_display_only_analysis_result(result)
        if key in self._analysis_caches:
            self._analysis_caches.move_to_end(key)
        self._analyses[key] = result
        self._mark_analysis_complete(result)
        if self.image_view.image_path == path:
            self._show_analysis_result(result)
        self._update_analysis_availability()
        self._sync_background_controls()
        learned_result = (
            result.unet_instances
            if self.pipeline.node("unet_instances").enabled
            and result.unet_instances is not None
            else result.stardist_instances
            if self.pipeline.node("stardist_instances").enabled
            and result.stardist_instances is not None
            else None
        )
        if learned_result is not None:
            self.statusBar().showMessage(
                f"Generated {learned_result.count:,} reviewable learned instances "
                f"for {path.name}; scientific validation is still required."
            )
        elif (
            self.pipeline.is_active("procedural_instances")
            and result.procedural_instances is not None
        ):
            self.statusBar().showMessage(
                f"Generated {result.procedural_instances.count:,} reviewable procedural "
                f"instances for {path.name}."
            )
        elif self.pipeline.is_active("identification"):
            self.statusBar().showMessage(
                f"Generated {result.count:,} approximate proposals for {path.name}."
            )
        else:
            self.statusBar().showMessage(
                f"Completed active diagnostic pipeline for {path.name}."
            )
        current_key = self._current_image_key()
        self._trim_analysis_caches(
            protected={current_key} if current_key is not None else set()
        )
        self._start_pending_analysis()

    @Slot(str, str, int)
    def _analysis_failed(
        self, path_text: str, error: str, pipeline_revision: int
    ) -> None:
        del pipeline_revision
        path = Path(path_text)
        key = str(path.resolve()).casefold()
        self._active_tasks.pop(key, None)
        self._discard_analysis_cache(key)
        if self.image_view.image_path == path:
            self.image_view.clear_analysis()
            self.pipeline_inspector.set_analysis_result(None)
        for node_id in (
            *CALIBRATION_NODE_IDS,
            "layout_detection",
            *IDENTIFICATION_STAGE_NODE_IDS,
        ):
            self.pipeline.set_status(node_id, NodeStatus.FAILED, error)
        for node_id in OVERLAY_NODE_IDS:
            self.pipeline.set_status(node_id, NodeStatus.FAILED, error)
        if not self.pipeline.node("background_likelihood").enabled:
            self.pipeline.set_status(
                "background_likelihood", NodeStatus.BYPASSED, "Bypassed"
            )
            self.pipeline.set_status(
                "refined_background_likelihood",
                NodeStatus.BLOCKED,
                "Background colour analysis disabled",
            )
        self.pipeline_canvas.refresh(
            (
                *CALIBRATION_NODE_IDS,
                "layout_detection",
                *IDENTIFICATION_STAGE_NODE_IDS,
                *OVERLAY_NODE_IDS,
            )
        )
        self.pipeline_inspector.refresh_status()
        self._update_analysis_availability()
        self._sync_background_controls()
        if self.image_view.image_path == path:
            self.count_label.setText("Failed")
            self.warning_label.setText(error)
        QMessageBox.warning(self, "Analysis failed", f"{path}\n\n{error}")
        self._start_pending_analysis()

    def _show_analysis_result(self, result) -> None:
        self._sync_directional_overlay_choices(result)
        self.image_view.show_analysis(result, render=False)
        self.pipeline_inspector.set_analysis_result(result)
        brush_radius = max(
            2,
            round(
                result.estimated_seed_diameter_px
                * self._layer_settings().background_sample_radius_fraction
            ),
        )
        with QSignalBlocker(self.reference_brush_slider):
            self.reference_brush_slider.setValue(brush_radius)
        with QSignalBlocker(self.instance_brush_slider):
            self.instance_brush_slider.setValue(brush_radius)
        self.reference_brush_label.setText(f"{brush_radius} px")
        self.instance_brush_label.setText(f"{brush_radius} px")
        self.image_view.set_reference_brush_radius(brush_radius)
        self._sync_instance_tool_settings()
        path = result.image_path
        key = str(path.resolve()).casefold() if path is not None else ""
        self.image_view.set_reference_masks(
            self._draft_background_reference_masks.get(
                key, self._applied_background_reference_masks.get(key)
            ),
            self._draft_foreground_reference_masks.get(
                key, self._applied_foreground_reference_masks.get(key)
            ),
            self._draft_background_exclusion_masks.get(
                key, self._applied_background_exclusion_masks.get(key)
            ),
            self._draft_foreground_exclusion_masks.get(
                key, self._applied_foreground_exclusion_masks.get(key)
            ),
            physical_edge_mask=self._draft_physical_edge_reference_masks.get(
                key, self._applied_physical_edge_reference_masks.get(key)
            ),
            non_edge_mask=self._draft_non_edge_reference_masks.get(
                key, self._applied_non_edge_reference_masks.get(key)
            ),
            copy=False,
            render=False,
            normalize_material=False,
        )
        self.image_view.set_instance_annotations(
            self._draft_instance_annotations.get(
                key, self._applied_instance_annotations.get(key)
            ),
            copy=False,
            render=False,
        )
        self.image_view.refresh_analysis()
        calibration = result.calibration
        card = calibration.colour_card
        if card is None:
            self.colour_status_label.setText("Not found; no neutral correction")
        else:
            gains = calibration.channel_gains_bgr
            self.colour_status_label.setText(
                f"{card.detected_swatch_count}/24 found; BGR gains "
                f"{gains[0]:.3f}/{gains[1]:.3f}/{gains[2]:.3f}"
            )
        ruler = calibration.ruler
        if ruler is None:
            self.ruler_status_label.setText("Not found")
        else:
            self.ruler_status_label.setText(
                f"Found; angle {ruler.angle_degrees:+.2f}°; "
                f"confidence {ruler.confidence:.0%}"
            )
        self.deskew_status_label.setText(
            f"Applied {calibration.deskew_degrees:+.2f}° rotation"
        )
        if calibration.perspective_corrected:
            self.deskew_status_label.setText(
                f"Projective correction; {calibration.deskew_degrees:+.2f}° "
                f"rotation, tilt {calibration.perspective_strength:.3f}"
            )
        if calibration.pixels_per_mm is None:
            self.scale_status_label.setText("Not assigned")
        else:
            self.scale_status_label.setText(
                f"{calibration.pixels_per_mm:.3f} px/mm; "
                f"confidence {calibration.scale_confidence:.0%}"
            )
        backend = result.advanced.backend
        self.compute_status_label.setText(
            f"Full tensor pipeline — {backend.summary}; "
            f"diagnostic maps {backend.elapsed_seconds:.2f}s"
        )
        self.overlay_combo.setEnabled(True)
        self.overlay_opacity_slider.setEnabled(True)
        selected_overlay = str(self.overlay_combo.currentData())
        self._sync_overlay_display_controls(selected_overlay)
        self.image_view.set_overlay_mode(selected_overlay)
        self.image_view.set_overlay_opacity(
            self.overlay_opacity_slider.value() / 100.0
        )
        self._update_overlay_legend(selected_overlay)
        dish = result.dish
        self.dish_status_label.setText(
            f"Petri dish; lower {dish.inner_radius:,} px, upper "
            f"{dish.outer_radius:,} px; confidence {dish.confidence:.0%}"
        )
        if result.reference_seed_count:
            self.reference_status_label.setText(
                f"{result.reference_seed_count} seed(s); diameter ≈ "
                f"{result.estimated_seed_diameter_px:.0f} px"
            )
        else:
            self.reference_status_label.setText(
                f"Fallback estimate ≈ {result.estimated_seed_diameter_px:.0f} px"
            )
        displayed_count = (
            result.unet_instances.count
            if self.pipeline.node("unet_instances").enabled
            and result.unet_instances is not None
            else result.stardist_instances.count
            if self.pipeline.node("stardist_instances").enabled
            and result.stardist_instances is not None
            else result.procedural_instances.count
            if self.pipeline.is_active("procedural_instances")
            and result.procedural_instances is not None
            else result.count
        )
        self.count_label.setText(f"≈ {displayed_count:,} seeds")
        self.crowding_label.setText(
            f"Crowding: {result.crowding}. Method: {result.method}."
        )
        self.warning_label.setText("\n".join(result.warnings))
        self._sync_background_controls()

    def _sync_directional_overlay_choices(self, result) -> None:
        selected = str(self.overlay_combo.currentData() or "")
        self._overlay_entries = [
            (label, mode)
            for label, mode in self._overlay_entries
            if not mode.startswith((
                "colour_probability:",
                "pattern_probability:",
            ))
            and mode != "none"
        ]
        for probability_index, class_name in enumerate(
            result.advanced.colour_class_names
        ):
            self._overlay_entries.append(
                (
                    f"Colour probability: {class_name}",
                    f"colour_probability:{probability_index}",
                )
            )
        for probability_index, class_name in enumerate(
            result.advanced.pattern_class_names
        ):
            self._overlay_entries.append(
                (
                    f"Pattern probability: {class_name}",
                    f"pattern_probability:{probability_index}",
                )
            )
        self._overlay_entries.append(("None", "none"))
        self._rebuild_overlay_combo(selected)

    def _current_image_key(self) -> str | None:
        path = self.image_view.image_path
        if path is None:
            return None
        return str(path.resolve()).casefold()

    def _reference_group_state(
        self, key: str, context: str
    ) -> tuple[np.ndarray, ...]:
        empty = self._empty_current_image_mask()

        def current(draft, applied):
            value = draft.get(key, applied.get(key))
            return empty if value is None else np.asarray(value, dtype=bool)

        if context == "material":
            return (
                current(
                    self._draft_background_reference_masks,
                    self._applied_background_reference_masks,
                ),
                current(
                    self._draft_foreground_reference_masks,
                    self._applied_foreground_reference_masks,
                ),
                current(
                    self._draft_background_exclusion_masks,
                    self._applied_background_exclusion_masks,
                ),
            )
        if context == "boundary":
            return (
                current(
                    self._draft_physical_edge_reference_masks,
                    self._applied_physical_edge_reference_masks,
                ),
                current(
                    self._draft_non_edge_reference_masks,
                    self._applied_non_edge_reference_masks,
                ),
            )
        raise ValueError(f"Unknown reference history context {context!r}.")

    def _instance_reference_state(self, key: str) -> tuple[np.ndarray, ...]:
        values = self._draft_instance_annotations.get(
            key, self._applied_instance_annotations.get(key)
        )
        if values is None:
            values = self._empty_current_instance_annotations()
        return (np.asarray(values, dtype=np.uint16),)

    def _instance_reference_origin(self, key: str) -> str:
        return self._draft_instance_annotation_origins.get(
            key,
            self._applied_instance_annotation_origins.get(key, "manual"),
        )

    def _record_reference_undo(
        self,
        key: str,
        label: str,
        context: str,
        before: tuple[np.ndarray, ...],
        after: tuple[np.ndarray, ...],
    ) -> bool:
        history = self._reference_undo_histories.setdefault(
            key, RasterUndoHistory(self.REFERENCE_UNDO_LIMIT)
        )
        changed = history.record(label, context, before, after)
        if not changed and not len(history):
            self._reference_undo_histories.pop(key, None)
        return changed

    def _record_instance_undo(
        self,
        key: str,
        label: str,
        before: tuple[np.ndarray, ...],
        after: tuple[np.ndarray, ...],
        *,
        before_origin: str,
    ) -> bool:
        history = self._instance_undo_histories.setdefault(
            key, RasterUndoHistory(self.REFERENCE_UNDO_LIMIT)
        )
        changed = history.record(
            label,
            "instances",
            before,
            after,
            metadata=before_origin,
        )
        if not changed and not len(history):
            self._instance_undo_histories.pop(key, None)
        return changed

    @staticmethod
    def _matches_applied_raster(
        current: np.ndarray | None, applied: np.ndarray | None
    ) -> bool:
        if current is None:
            return applied is None or not bool(np.any(applied))
        values = np.asarray(current)
        if applied is None:
            return not bool(np.any(values))
        return np.array_equal(values, np.asarray(applied))

    def _set_reference_draft_state(
        self,
        key: str,
        context: str,
        rasters: tuple[np.ndarray, ...],
    ) -> None:
        if context == "material":
            if len(rasters) != 3:
                raise ValueError("Material undo state must contain three classes.")
            background, foreground, other = (
                np.asarray(value, dtype=bool) for value in rasters
            )
            self._draft_background_reference_masks[key] = background
            self._draft_foreground_reference_masks[key] = foreground
            self._draft_background_exclusion_masks[key] = other
            self._draft_foreground_exclusion_masks[key] = other
        elif context == "boundary":
            if len(rasters) != 2:
                raise ValueError("Boundary undo state must contain two classes.")
            physical, non_edge = (
                np.asarray(value, dtype=bool) for value in rasters
            )
            self._draft_physical_edge_reference_masks[key] = physical
            self._draft_non_edge_reference_masks[key] = non_edge
        else:
            raise ValueError(f"Unknown reference history context {context!r}.")

    def _reconcile_reference_draft(self, key: str) -> None:
        dirty_classes: set[str] = set()
        singles = (
            (
                "background",
                self._draft_background_reference_masks,
                self._applied_background_reference_masks,
            ),
            (
                "foreground",
                self._draft_foreground_reference_masks,
                self._applied_foreground_reference_masks,
            ),
            (
                "physical_edge",
                self._draft_physical_edge_reference_masks,
                self._applied_physical_edge_reference_masks,
            ),
            (
                "non_edge",
                self._draft_non_edge_reference_masks,
                self._applied_non_edge_reference_masks,
            ),
        )
        for class_name, draft, applied in singles:
            current = draft.get(key, applied.get(key))
            if self._matches_applied_raster(current, applied.get(key)):
                draft.pop(key, None)
            else:
                dirty_classes.add(class_name)

        other = self._draft_background_exclusion_masks.get(
            key, self._applied_background_exclusion_masks.get(key)
        )
        if self._matches_applied_raster(
            other, self._applied_background_exclusion_masks.get(key)
        ):
            self._draft_background_exclusion_masks.pop(key, None)
            self._draft_foreground_exclusion_masks.pop(key, None)
        else:
            other_values = np.asarray(other, dtype=bool)
            self._draft_background_exclusion_masks[key] = other_values
            self._draft_foreground_exclusion_masks[key] = other_values
            dirty_classes.add("other")

        if dirty_classes:
            self._reference_masks_dirty.add(key)
            self._reference_dirty_classes[key] = dirty_classes
        else:
            self._reference_masks_dirty.discard(key)
            self._reference_dirty_classes.pop(key, None)

    def _set_instance_draft_state(
        self, key: str, values: np.ndarray, origin: str | None
    ) -> None:
        self._instance_continuity_cache.pop(key, None)
        labels = np.asarray(values, dtype=np.uint16)
        applied = self._applied_instance_annotations.get(key)
        if self._matches_applied_raster(labels, applied):
            self._draft_instance_annotations.pop(key, None)
            self._draft_instance_annotation_origins.pop(key, None)
            self._instance_annotations_dirty.discard(key)
        else:
            self._draft_instance_annotations[key] = labels
            self._draft_instance_annotation_origins[key] = origin or "manual"
            self._instance_annotations_dirty.add(key)

    def _sync_reference_undo_controls(
        self, key: str | None, *, has_result: bool, running: bool
    ) -> None:
        instance_mode = self.annotate_instances_action.isChecked()
        reference_mode = self.paint_background_action.isChecked()
        histories = (
            self._instance_undo_histories
            if instance_mode
            else self._reference_undo_histories
        )
        history = None if key is None else histories.get(key)
        count = 0 if history is None else len(history)
        active = instance_mode or reference_mode
        available = active and has_result and not running and count > 0
        self.reference_undo_button.setEnabled(available)
        self.reference_undo_button.setText(
            "Undo" if not count else f"Undo ({count})"
        )
        next_label = None if history is None else history.next_label
        tooltip = (
            f"Undo {next_label} (Ctrl+Z). "
            if next_label
            else "No painting commands to undo. "
        ) + f"The latest {self.REFERENCE_UNDO_LIMIT} edits are retained per image."
        self.reference_undo_button.setToolTip(tooltip)
        self.undo_reference_edit_action.setEnabled(available)
        self.undo_reference_edit_action.setText(
            "Undo reference edit" if next_label is None else f"Undo {next_label}"
        )

    @Slot()
    def _undo_active_reference_edit(self) -> None:
        if self.annotate_instances_action.isChecked():
            self._undo_instance_reference_edit()
        elif self.paint_background_action.isChecked():
            self._undo_reference_mask_edit()

    def _undo_reference_mask_edit(self) -> None:
        key = self._current_image_key()
        history = None if key is None else self._reference_undo_histories.get(key)
        if key is None or history is None or not len(history):
            return
        context = history.next_context
        if context is None:
            return
        result = history.undo(self._reference_group_state(key, context))
        if result is None:
            return
        self._set_reference_draft_state(key, result.context, result.rasters)
        self._reconcile_reference_draft(key)
        self._sync_reference_masks_to_view(key)
        if not len(history):
            self._reference_undo_histories.pop(key, None)
        self._sync_background_controls()
        self.statusBar().showMessage(f"Undid {result.label}.")

    def _undo_instance_reference_edit(self) -> None:
        key = self._current_image_key()
        history = None if key is None else self._instance_undo_histories.get(key)
        if key is None or history is None or not len(history):
            return
        result = history.undo(self._instance_reference_state(key))
        if result is None:
            return
        self._set_instance_draft_state(key, result.rasters[0], result.metadata)
        self.image_view.set_instance_annotations(
            self._draft_instance_annotations.get(
                key, self._applied_instance_annotations.get(key)
            ),
            copy=False,
        )
        if not len(history):
            self._instance_undo_histories.pop(key, None)
        self._sync_background_controls()
        self.statusBar().showMessage(f"Undid {result.label}.")

    def _auto_load_reference_regions(
        self, path: Path, image_shape: tuple[int, int]
    ) -> bool:
        """Restore an all-or-nothing applied snapshot for an unchanged image."""

        try:
            bundle = self._reference_region_store.load_if_present(path, image_shape)
        except ImageFingerprintMismatch as error:
            QMessageBox.warning(
                self,
                "Saved reference regions not loaded",
                f"Saved reference regions exist for {path.name}, but the image "
                "contents have changed since they were saved. Seed Fiddle did not "
                "load any of those regions.\n\n"
                f"Saved SHA-256: {error.expected_sha256}\n"
                f"Current SHA-256: {error.actual_sha256}\n\n"
                f"The saved archive was left unchanged at:\n{error.archive_path}",
            )
            return False
        except (InvalidReferenceArchive, ReferenceRegionError, OSError) as error:
            QMessageBox.warning(
                self,
                "Saved reference regions not loaded",
                f"Seed Fiddle could not safely load the saved reference regions "
                f"for {path.name}. No saved regions were applied.\n\n{error}\n\n"
                "The saved archive was left unchanged at:\n"
                f"{self._reference_region_store.path_for(path)}",
            )
            return False
        if bundle is None:
            return False

        key = str(path.resolve()).casefold()
        self._install_reference_region_bundle(key, bundle)
        self._discard_analysis_cache(key)
        affected = {"reference_layers"}
        for port_id in (
            "background",
            "foreground",
            "other",
            "physical_edge",
            "non_edge",
            "annotated_seeds",
        ):
            affected.update(
                self.pipeline.downstream_from_port(
                    "reference_layers", port_id, recursive=True
                )
            )
        self.pipeline.invalidate(affected)
        self._cache_dirty_nodes.setdefault(key, set()).update(
            affected - {"reference_layers"}
        )
        return True

    def _install_reference_region_bundle(
        self, key: str, bundle: ReferenceRegionBundle
    ) -> None:
        """Commit one fully validated persistent snapshot to controller state."""

        self._instance_continuity_cache.pop(key, None)

        for draft in (
            self._draft_background_reference_masks,
            self._draft_foreground_reference_masks,
            self._draft_background_exclusion_masks,
            self._draft_foreground_exclusion_masks,
            self._draft_physical_edge_reference_masks,
            self._draft_non_edge_reference_masks,
        ):
            draft.pop(key, None)
        self._draft_instance_annotations.pop(key, None)
        self._draft_instance_annotation_origins.pop(key, None)
        self._reference_masks_dirty.discard(key)
        self._reference_dirty_classes.pop(key, None)
        self._instance_annotations_dirty.discard(key)
        self._reference_undo_histories.pop(key, None)
        self._instance_undo_histories.pop(key, None)

        def install(
            store: dict[str, np.ndarray], values: np.ndarray | None, dtype
        ) -> np.ndarray | None:
            if values is None or not np.any(values):
                store.pop(key, None)
                return None
            stored = np.asarray(values, dtype=dtype)
            stored.flags.writeable = False
            store[key] = stored
            return stored

        install(
            self._applied_background_reference_masks, bundle.background, bool
        )
        install(
            self._applied_foreground_reference_masks, bundle.foreground, bool
        )
        other = install(
            self._applied_background_exclusion_masks, bundle.other, bool
        )
        if other is None:
            self._applied_foreground_exclusion_masks.pop(key, None)
        else:
            self._applied_foreground_exclusion_masks[key] = other
        install(
            self._applied_physical_edge_reference_masks,
            bundle.physical_edge,
            bool,
        )
        install(
            self._applied_non_edge_reference_masks, bundle.non_edge, bool
        )
        annotations = install(
            self._applied_instance_annotations,
            bundle.annotated_seeds,
            np.uint16,
        )
        if annotations is None:
            self._applied_instance_annotation_origins.pop(key, None)
        else:
            self._applied_instance_annotation_origins[key] = (
                bundle.annotation_origin or "manual"
            )
        self._sync_reference_masks_to_view(key, render=False)
        self.image_view.set_instance_annotations(
            self._applied_instance_annotations.get(key),
            copy=False,
            render=False,
        )

    @Slot()
    def _save_reference_regions(self) -> None:
        """Persist all applied Reference layers outputs for the current image."""

        key = self._current_image_key()
        path = self.image_view.image_path
        image_size = self.image_view.image_size
        if key is None or path is None or image_size is None:
            return
        if key in self._reference_masks_dirty or key in self._instance_annotations_dirty:
            QMessageBox.warning(
                self,
                "Apply or revert edits first",
                "Reference-region files contain applied evidence only. Apply or "
                "revert the current material and seed-instance drafts "
                "before saving.",
            )
            return
        destination = self._persist_applied_reference_regions(automatic=False)
        if destination is None:
            return
        self.statusBar().showMessage(
            f"Saved applied reference regions for {path.name} to {destination}."
        )

    def _persist_applied_reference_regions(
        self, *, automatic: bool
    ) -> Path | None:
        """Atomically persist the authoritative applied state, never a draft."""

        key = self._current_image_key()
        path = self.image_view.image_path
        image_size = self.image_view.image_size
        if key is None or path is None or image_size is None:
            return None
        width, height = image_size
        bundle = ReferenceRegionBundle(
            shape=(height, width),
            background=self._applied_background_reference_masks.get(key),
            foreground=self._applied_foreground_reference_masks.get(key),
            other=self._applied_background_exclusion_masks.get(key),
            # Retired manual boundary classes remain readable from legacy
            # archives, but new snapshots no longer perpetuate them.
            physical_edge=None,
            non_edge=None,
            annotated_seeds=self._applied_instance_annotations.get(key),
            annotation_origin=self._applied_instance_annotation_origins.get(
                key, "manual"
            ),
        )
        try:
            destination = self._reference_region_store.save(path, bundle)
        except (ReferenceRegionError, OSError) as error:
            QMessageBox.critical(
                self,
                (
                    "Automatic reference save failed"
                    if automatic
                    else "Could not save reference regions"
                ),
                (
                    "The applied changes remain available in this Seed Fiddle "
                    "session, but they could not be saved to disk. No existing "
                    "reference-region archive was replaced.\n\n"
                    if automatic
                    else "No reference-region archive was replaced.\n\n"
                )
                + str(error),
            )
            return None
        self._reference_region_autoload_attempted.add(key)
        return destination

    def _ensure_reference_draft(self, mask_kind: str) -> None:
        """Create the selected draft; opposing classes become writable on first dab."""

        key = self._current_image_key()
        if key is None:
            return
        stores = {
            "background": (
                self._draft_background_reference_masks,
                self._applied_background_reference_masks,
            ),
            "foreground": (
                self._draft_foreground_reference_masks,
                self._applied_foreground_reference_masks,
            ),
            "background_exclusion": (
                self._draft_background_exclusion_masks,
                self._applied_background_exclusion_masks,
            ),
            "foreground_exclusion": (
                self._draft_foreground_exclusion_masks,
                self._applied_foreground_exclusion_masks,
            ),
        }
        if mask_kind == "other":
            mask_kind = "background_exclusion"
        if mask_kind not in stores:
            return
        draft, applied = stores[mask_kind]
        if key not in draft:
            source = applied.get(key)
            draft[key] = (
                self._empty_current_image_mask()
                if source is None
                else source.copy()
            )
        if mask_kind == "background_exclusion":
            self._draft_foreground_exclusion_masks[key] = draft[key]
        self._sync_reference_masks_to_view(key, render=False)

    def _ensure_instance_draft(self) -> None:
        key = self._current_image_key()
        if key is None:
            return
        if key not in self._draft_instance_annotations:
            applied = self._applied_instance_annotations.get(key)
            self._draft_instance_annotations[key] = (
                self._empty_current_instance_annotations()
                if applied is None
                else applied.copy()
            )
            self._draft_instance_annotation_origins[key] = (
                self._applied_instance_annotation_origins.get(key, "manual")
            )
        self.image_view.set_instance_annotations(
            self._draft_instance_annotations[key], copy=False, render=False
        )

    def _discard_analysis_cache(self, key: str) -> bool:
        """Drop one complete per-image cache and all of its lazy host mirrors."""

        cache = self._analysis_caches.pop(key, None)
        result = self._analyses.pop(key, None)
        self._cache_dirty_nodes.pop(key, None)
        if cache is None and result is None:
            return False
        release_host_caches(cache.values if cache is not None else result)
        if cache is not None:
            cache.values.clear()
            cache.last_computed_nodes = ()
            cache.last_reused_nodes = ()
            cache.node_timings_seconds.clear()
        return True

    @staticmethod
    def _release_unused_cuda_blocks() -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass

    def _trim_analysis_caches(
        self, *, protected: set[str] | frozenset[str] = frozenset()
    ) -> tuple[str, ...]:
        """Enforce both count and CUDA-byte limits in least-recently-used order."""

        protected_keys = set(protected) | set(self._active_tasks)
        if self._procedural_fit_task is not None:
            protected_keys.add(self._procedural_fit_task.image_key)
        evicted: list[str] = []

        def cuda_total() -> int:
            return sum(
                resident_bytes(cache.values).cuda
                for cache in self._analysis_caches.values()
            )

        while self._analysis_caches:
            over_count = len(self._analysis_caches) > self.ANALYSIS_CACHE_MAX_IMAGES
            over_cuda = cuda_total() > self.ANALYSIS_CACHE_CUDA_BUDGET_BYTES
            if not over_count and not over_cuda:
                break
            victim = next(
                (
                    key
                    for key in self._analysis_caches
                    if key not in protected_keys
                ),
                None,
            )
            if victim is None:
                break
            if self._discard_analysis_cache(victim):
                evicted.append(victim)
        if evicted:
            self._release_unused_cuda_blocks()
        return tuple(evicted)

    def _stop_background_point_editing(self) -> None:
        if not hasattr(self, "background_point_button"):
            return
        with QSignalBlocker(self.background_point_button):
            self.background_point_button.setChecked(False)
        with QSignalBlocker(self.paint_background_action):
            self.paint_background_action.setChecked(False)
        self.image_view.set_background_point_editing(False)
        self._sync_reference_panel_visibility()

    def _stop_foreground_point_editing(self) -> None:
        if not hasattr(self, "foreground_point_button"):
            return
        with QSignalBlocker(self.foreground_point_button):
            self.foreground_point_button.setChecked(False)
        with QSignalBlocker(self.paint_background_action):
            self.paint_background_action.setChecked(False)
        self.image_view.set_foreground_point_editing(False)
        self._sync_reference_panel_visibility()

    def _stop_background_exclusion_editing(self) -> None:
        if not hasattr(self, "background_exclusion_button"):
            return
        with QSignalBlocker(self.background_exclusion_button):
            self.background_exclusion_button.setChecked(False)
        self.image_view.set_background_exclusion_editing(False)
        self._sync_reference_panel_visibility()

    def _stop_foreground_exclusion_editing(self) -> None:
        if not hasattr(self, "foreground_exclusion_button"):
            return
        with QSignalBlocker(self.foreground_exclusion_button):
            self.foreground_exclusion_button.setChecked(False)
        self.image_view.set_foreground_exclusion_editing(False)
        self._sync_reference_panel_visibility()

    def _stop_edge_reference_editing(self) -> None:
        if not hasattr(self, "physical_edge_button"):
            return
        for button in (self.physical_edge_button, self.non_edge_button):
            with QSignalBlocker(button):
                button.setChecked(False)
        self.image_view.set_physical_edge_reference_editing(False)
        self.image_view.set_non_edge_reference_editing(False)
        self._sync_reference_panel_visibility()

    def _stop_instance_annotation_editing(self) -> None:
        if not hasattr(self, "annotate_instances_action"):
            return
        with QSignalBlocker(self.annotate_instances_action):
            self.annotate_instances_action.setChecked(False)
        self.image_view.set_instance_annotation_editing(False)
        self._sync_reference_panel_visibility()

    def _stop_reference_point_editing(self) -> None:
        self._stop_background_point_editing()
        self._stop_foreground_point_editing()
        self._stop_background_exclusion_editing()
        self._stop_foreground_exclusion_editing()
        self._stop_edge_reference_editing()
        self._stop_instance_annotation_editing()
        with QSignalBlocker(self.paint_background_action):
            self.paint_background_action.setChecked(False)
        self.image_view.set_context_panel_visible(False)

    def _sync_reference_panel_visibility(self) -> None:
        active = (
            self.background_point_button.isChecked()
            or self.foreground_point_button.isChecked()
            or self.background_exclusion_button.isChecked()
            or self.foreground_exclusion_button.isChecked()
            or self.annotate_instances_action.isChecked()
        )
        instance_mode = self.annotate_instances_action.isChecked()
        self.reference_controls.setVisible(active and not instance_mode)
        self.instance_annotation_controls.setVisible(instance_mode)
        self.image_view.set_context_panel_visible(active)
        if hasattr(self, "reference_undo_button"):
            key = self._current_image_key()
            self._sync_reference_undo_controls(
                key,
                has_result=self.image_view._analysis_result is not None,
                running=(
                    (key is not None and key in self._active_tasks)
                    or self._learning_training_task is not None
                    or self._procedural_fit_task is not None
                ),
            )

    def _sync_annotation_proposal_choices(self, result, *, enabled: bool) -> None:
        """Expose only calculated instance outputs as annotation starting points."""

        if not hasattr(self, "instance_proposal_combo"):
            return
        previous = self.instance_proposal_combo.currentData()
        options = []
        if result is not None:
            for label, attribute in (
                ("Procedural separation", "procedural_instances"),
                ("U-Net + watershed", "unet_instances"),
                ("StarDist", "stardist_instances"),
            ):
                proposal = getattr(result, attribute, None)
                if proposal is not None and int(getattr(proposal, "count", 0)) > 0:
                    options.append((label, attribute))
        with QSignalBlocker(self.instance_proposal_combo):
            self.instance_proposal_combo.clear()
            for label, attribute in options:
                self.instance_proposal_combo.addItem(label, attribute)
            if not options:
                self.instance_proposal_combo.addItem(
                    "No instance result available", None
                )
            elif previous is not None:
                selected = self.instance_proposal_combo.findData(previous)
                if selected >= 0:
                    self.instance_proposal_combo.setCurrentIndex(selected)
        available = bool(options) and bool(enabled)
        self.instance_proposal_combo.setEnabled(available)
        self.use_instance_proposal_button.setEnabled(available)

    def _sync_background_controls(self) -> None:
        if not hasattr(self, "background_point_button"):
            return
        key = self._current_image_key()
        enabled = self.pipeline.node("background_likelihood").enabled
        with QSignalBlocker(self.background_enabled_checkbox):
            self.background_enabled_checkbox.setChecked(enabled)
        running = (
            key in self._active_tasks if key is not None else False
        ) or self._learning_training_task is not None or (
            self._procedural_fit_task is not None
        )
        has_result = self.image_view._analysis_result is not None
        can_edit = enabled and has_result and not running
        self.background_point_button.setEnabled(can_edit)
        self.foreground_point_button.setEnabled(has_result and not running)
        self.background_exclusion_button.setEnabled(has_result and not running)
        self.foreground_exclusion_button.setEnabled(has_result and not running)
        self.physical_edge_button.setEnabled(False)
        self.non_edge_button.setEnabled(False)
        self.edge_snap_checkbox.setEnabled(False)
        self.edge_snap_strength_slider.setEnabled(False)
        self.erase_background_points_button.setEnabled(can_edit)
        self.erase_foreground_points_button.setEnabled(has_result and not running)
        self.erase_background_exclusion_button.setEnabled(has_result and not running)
        self.erase_foreground_exclusion_button.setEnabled(has_result and not running)
        self.paint_background_action.setEnabled(has_result and not running)
        self.annotate_instances_action.setEnabled(has_result and not running)
        background_mask = self._draft_background_reference_masks.get(
            key or "", self._applied_background_reference_masks.get(key or "")
        )
        foreground_mask = self._draft_foreground_reference_masks.get(
            key or "", self._applied_foreground_reference_masks.get(key or "")
        )
        background_exclusion = self._draft_background_exclusion_masks.get(
            key or "", self._applied_background_exclusion_masks.get(key or "")
        )
        foreground_exclusion = self._draft_foreground_exclusion_masks.get(
            key or "", self._applied_foreground_exclusion_masks.get(key or "")
        )
        background_count = self._mask_pixel_count(background_mask)
        foreground_count = self._mask_pixel_count(foreground_mask)
        background_exclusion_count = self._mask_pixel_count(background_exclusion)
        foreground_exclusion_count = self._mask_pixel_count(foreground_exclusion)
        dirty = key in self._reference_masks_dirty if key is not None else False
        annotations = self._draft_instance_annotations.get(
            key or "", self._applied_instance_annotations.get(key or "")
        )
        continuity = self._instance_continuity_summary(key, annotations)
        annotation_ids = continuity.identifiers
        annotation_count = continuity.pixel_count
        annotations_dirty = (
            key in self._instance_annotations_dirty if key is not None else False
        )
        can_save_references = (
            key is not None and not running and not dirty and not annotations_dirty
        )
        self.save_reference_regions_action.setEnabled(can_save_references)
        self.save_reference_regions_button.setEnabled(can_save_references)
        self.export_learning_sample_action.setEnabled(
            has_result
            and not running
            and not annotations_dirty
            and bool(
                self._mask_pixel_count(
                    self._applied_instance_annotations.get(key or "")
                )
            )
        )
        self.load_instance_labels_action.setEnabled(has_result and not running)
        self.choose_instance_labels_action.setEnabled(has_result and not running)
        self.load_instance_reference_button.setEnabled(has_result and not running)
        self.choose_instance_mask_button.setEnabled(has_result and not running)
        self.save_instance_labels_action.setEnabled(
            not running
            and bool(
                self._mask_pixel_count(
                    self._applied_instance_annotations.get(key or "")
                )
            )
        )
        self._sync_annotation_proposal_choices(
            self._analyses.get(key or ""), enabled=has_result and not running
        )
        self.clear_background_points_button.setEnabled(
            bool(background_count) and not running
        )
        self.clear_foreground_points_button.setEnabled(
            bool(foreground_count) and not running
        )
        self.clear_background_exclusion_button.setEnabled(
            bool(background_exclusion_count) and not running
        )
        self.clear_foreground_exclusion_button.setEnabled(
            bool(foreground_exclusion_count) and not running
        )
        self.apply_reference_masks_button.setEnabled(dirty and not running)
        self.revert_reference_masks_button.setEnabled(dirty and not running)
        self.reference_brush_slider.setEnabled(has_result and not running)
        self.reference_paint_mode_button.setEnabled(has_result and not running)
        self.reference_eraser_button.setEnabled(has_result and not running)
        self.clear_reference_layer_button.setEnabled(has_result and not running)
        self.instance_id_spin.setEnabled(has_result and not running)
        self.show_selected_instance_checkbox.setEnabled(has_result and not running)
        self.new_instance_button.setEnabled(has_result and not running)
        self.instance_brush_slider.setEnabled(has_result and not running)
        for button in (
            self.instance_paint_mode_button,
            self.instance_edge_trace_button,
            self.instance_shape_guided_fill_button,
            self.instance_smart_fill_button,
            self.instance_eraser_button,
        ):
            button.setEnabled(has_result and not running)
        self.instance_tool_options_stack.setEnabled(has_result and not running)
        active_id = self.instance_id_spin.value()
        self.clear_current_instance_button.setEnabled(
            active_id in annotation_ids and not running
        )
        self.clear_all_instances_button.setEnabled(
            bool(annotation_count) and not running
        )
        self.apply_instance_annotations_button.setEnabled(
            annotations_dirty and not running
        )
        self.revert_instance_annotations_button.setEnabled(
            annotations_dirty and not running
        )
        if not can_edit and self.background_point_button.isChecked():
            self._stop_background_point_editing()
        if (not has_result or running) and self.foreground_point_button.isChecked():
            self._stop_foreground_point_editing()
        if (not has_result or running) and self.background_exclusion_button.isChecked():
            self._stop_background_exclusion_editing()
        if (not has_result or running) and self.foreground_exclusion_button.isChecked():
            self._stop_foreground_exclusion_editing()
        if (not has_result or running) and (
            self.physical_edge_button.isChecked() or self.non_edge_button.isChecked()
        ):
            self._stop_edge_reference_editing()
        if (not has_result or running) and self.annotate_instances_action.isChecked():
            self._stop_instance_annotation_editing()
        if not enabled:
            text = "Disabled. Rerun to skip colour and noise background maps."
        elif background_count:
            suffix = " (unapplied draft)" if dirty else " (applied)"
            text = f"{background_count:,} background reference pixels{suffix}."
        else:
            text = "Automatic background colour selection; no painted area."
        self.background_reference_label.setText(text)
        if foreground_count:
            suffix = " (unapplied draft)" if dirty else " (applied)"
            foreground_text = (
                f"{foreground_count:,} foreground reference pixels{suffix}."
            )
        else:
            foreground_text = "No painted foreground reference."
        self.foreground_reference_label.setText(foreground_text)
        if background_exclusion_count:
            suffix = " (unapplied draft)" if dirty else " (applied)"
            background_exclusion_text = (
                f"{background_exclusion_count:,} negative background-example pixels"
                f"{suffix}."
            )
        else:
            background_exclusion_text = "No painted negative background examples."
        self.background_exclusion_label.setText(background_exclusion_text)
        if foreground_exclusion_count:
            suffix = " (unapplied draft)" if dirty else " (applied)"
            foreground_exclusion_text = (
                f"{foreground_exclusion_count:,} negative foreground-example pixels"
                f"{suffix}."
            )
        else:
            foreground_exclusion_text = "No painted negative foreground examples."
        self.foreground_exclusion_label.setText(foreground_exclusion_text)
        self.reference_confirmation_label.setText(
            f"Draft — BG {background_count:,} | FG {foreground_count:,} | "
            f"Other {max(background_exclusion_count, foreground_exclusion_count):,}"
            if dirty
            else f"Applied — BG {background_count:,} | FG {foreground_count:,} | "
            f"Other {max(background_exclusion_count, foreground_exclusion_count):,}"
        )
        self.reference_confirmation_label.setToolTip(
            "Draft changes do not affect analysis or disk storage until Apply + save. "
            "Material classes are mutually exclusive."
        )
        if annotation_ids:
            suffix = " (unapplied draft)" if annotations_dirty else " (applied)"
            self.instance_annotation_status_label.setText(
                f"{len(annotation_ids):,} seed IDs; {annotation_count:,} interior "
                f"annotation pixels{suffix}."
            )
        else:
            self.instance_annotation_status_label.setText(
                "No seed instances annotated."
            )
        disconnected = continuity.disconnected
        selected_components = continuity.component_count(active_id)
        if selected_components > 1:
            warning_text = (
                f"⚠ Seed {active_id} has {selected_components} disconnected areas."
            )
        elif disconnected:
            identifiers = ", ".join(
                str(identifier) for identifier, _count in disconnected[:8]
            )
            if len(disconnected) > 8:
                identifiers += f", and {len(disconnected) - 8} more"
            noun = "seed ID has" if len(disconnected) == 1 else "seed IDs have"
            warning_text = (
                f"⚠ {len(disconnected)} {noun} disconnected areas: {identifiers}."
            )
        else:
            warning_text = ""
        self.instance_continuity_warning_label.setText(warning_text)
        self.instance_continuity_warning_label.setVisible(bool(warning_text))
        warning_tool_tip = (
            "Seed IDs should normally form one 8-connected area. Review stray "
            "marks or reused IDs; Apply remains available."
        )
        if disconnected:
            details = "; ".join(
                f"Seed {identifier}: {count} areas"
                for identifier, count in disconnected
            )
            warning_tool_tip = (
                "8-neighbour continuity check — "
                + details
                + ". Review stray marks or reused IDs; Apply remains available."
            )
        self.instance_continuity_warning_label.setToolTip(warning_tool_tip)
        self.image_view.set_instance_continuity_warning(
            warning_text,
            tool_tip=warning_tool_tip,
        )
        self.instance_annotation_confirmation_label.setText(
            "Draft changed. Apply + save when complete; downstream calculations and "
            "the saved file still use the previous annotations."
            if annotations_dirty
            else "Applied annotations are saved automatically and constrain the "
            "instance branch."
        )
        self._sync_reference_undo_controls(
            key, has_result=has_result, running=running
        )
        self._sync_procedural_fit_controls()

    @staticmethod
    def _mask_pixel_count(mask: np.ndarray | None) -> int:
        return 0 if mask is None else int(np.count_nonzero(mask))

    @staticmethod
    def _instance_ids(annotations: np.ndarray | None) -> tuple[int, ...]:
        if annotations is None:
            return ()
        return tuple(int(value) for value in np.unique(annotations) if value > 0)

    def _instance_continuity_summary(
        self, key: str | None, annotations: np.ndarray | None
    ) -> InstanceContinuitySummary:
        """Reuse one compact continuity scan while the label object is unchanged."""

        if annotations is None:
            return InstanceContinuitySummary((), 0, ())
        values = np.asarray(annotations)
        if key is not None:
            cached = self._instance_continuity_cache.get(key)
            if cached is not None and cached[0]() is values:
                return cached[1]
        summary = summarize_instance_continuity(values)
        if key is not None:
            self._instance_continuity_cache[key] = (weakref.ref(values), summary)
        return summary

    @Slot(bool)
    def _background_enabled_toggled(self, enabled: bool) -> None:
        self._pipeline_enabled_changed("background_likelihood", enabled)

    @Slot(bool)
    def _background_toolbar_editing_changed(self, enabled: bool) -> None:
        if enabled:
            target = (
                self.background_point_button
                if self.background_point_button.isEnabled()
                else self.foreground_point_button
            )
            target.setChecked(True)
            return
        for button in (
            self.background_point_button,
            self.foreground_point_button,
            self.background_exclusion_button,
        ):
            button.setChecked(False)

    @Slot(bool)
    def _background_point_editing_changed(self, enabled: bool) -> None:
        with QSignalBlocker(self.paint_background_action):
            self.paint_background_action.setChecked(enabled)
        if enabled:
            with QSignalBlocker(self.foreground_point_button):
                self.foreground_point_button.setChecked(False)
            with QSignalBlocker(self.background_exclusion_button):
                self.background_exclusion_button.setChecked(False)
            with QSignalBlocker(self.foreground_exclusion_button):
                self.foreground_exclusion_button.setChecked(False)
            with QSignalBlocker(self.physical_edge_button):
                self.physical_edge_button.setChecked(False)
            with QSignalBlocker(self.non_edge_button):
                self.non_edge_button.setChecked(False)
            with QSignalBlocker(self.annotate_instances_action):
                self.annotate_instances_action.setChecked(False)
        if enabled:
            self._ensure_reference_draft("background")
        self.image_view.set_background_point_editing(enabled)
        if enabled:
            self.image_view.set_reference_brush_radius(
                float(self.reference_brush_slider.value())
            )
            self.image_view.set_reference_erase_mode(
                self.reference_eraser_button.isChecked()
            )
            overlay_index = self.overlay_combo.findData("background_likelihood")
            if overlay_index >= 0:
                self.overlay_combo.setCurrentIndex(overlay_index)
            self.image_view.show()
            self._sync_reference_panel_visibility()
            self.statusBar().showMessage(
                f"Background brush: left-drag to {self._reference_brush_action()}; "
                "right-drag temporarily erases."
            )
        else:
            self._sync_reference_panel_visibility()
            self.statusBar().showMessage("Background reference painting finished.")

    @Slot(bool)
    def _foreground_point_editing_changed(self, enabled: bool) -> None:
        with QSignalBlocker(self.paint_background_action):
            self.paint_background_action.setChecked(enabled)
        if enabled:
            with QSignalBlocker(self.background_point_button):
                self.background_point_button.setChecked(False)
            with QSignalBlocker(self.background_exclusion_button):
                self.background_exclusion_button.setChecked(False)
            with QSignalBlocker(self.foreground_exclusion_button):
                self.foreground_exclusion_button.setChecked(False)
            with QSignalBlocker(self.physical_edge_button):
                self.physical_edge_button.setChecked(False)
            with QSignalBlocker(self.non_edge_button):
                self.non_edge_button.setChecked(False)
            with QSignalBlocker(self.annotate_instances_action):
                self.annotate_instances_action.setChecked(False)
        if enabled:
            self._ensure_reference_draft("foreground")
        self.image_view.set_foreground_point_editing(enabled)
        if enabled:
            self.image_view.set_reference_brush_radius(
                float(self.reference_brush_slider.value())
            )
            self.image_view.set_reference_erase_mode(
                self.reference_eraser_button.isChecked()
            )
            overlay_index = self.overlay_combo.findData("foreground_mask")
            if overlay_index >= 0:
                self.overlay_combo.setCurrentIndex(overlay_index)
            self.image_view.show()
            self._sync_reference_panel_visibility()
            self.statusBar().showMessage(
                f"Foreground brush: left-drag to {self._reference_brush_action()}; "
                "right-drag temporarily erases."
            )
        else:
            self._sync_reference_panel_visibility()
            self.statusBar().showMessage("Foreground reference painting finished.")

    @Slot(bool)
    def _background_exclusion_editing_changed(self, enabled: bool) -> None:
        with QSignalBlocker(self.paint_background_action):
            self.paint_background_action.setChecked(enabled)
        if enabled:
            for button in (
                self.background_point_button,
                self.foreground_point_button,
                self.foreground_exclusion_button,
                self.physical_edge_button,
                self.non_edge_button,
            ):
                with QSignalBlocker(button):
                    button.setChecked(False)
            with QSignalBlocker(self.annotate_instances_action):
                self.annotate_instances_action.setChecked(False)
        if enabled:
            self._ensure_reference_draft("other")
        self.image_view.set_other_reference_editing(enabled)
        if enabled:
            self.image_view.set_reference_brush_radius(
                float(self.reference_brush_slider.value())
            )
            self.image_view.set_reference_erase_mode(
                self.reference_eraser_button.isChecked()
            )
            overlay_index = self.overlay_combo.findData("other_colour_probability")
            if overlay_index >= 0:
                self.overlay_combo.setCurrentIndex(overlay_index)
            self.statusBar().showMessage(
                "Other material: painting replaces foreground/background at these pixels."
            )
        self._sync_reference_panel_visibility()

    @Slot(bool)
    def _foreground_exclusion_editing_changed(self, enabled: bool) -> None:
        if enabled:
            for button in (
                self.background_point_button,
                self.foreground_point_button,
                self.background_exclusion_button,
            ):
                with QSignalBlocker(button):
                    button.setChecked(False)
            with QSignalBlocker(self.paint_background_action):
                self.paint_background_action.setChecked(False)
            with QSignalBlocker(self.annotate_instances_action):
                self.annotate_instances_action.setChecked(False)
        if enabled:
            self._ensure_reference_draft("foreground_exclusion")
        self.image_view.set_foreground_exclusion_editing(enabled)
        if enabled:
            self.image_view.set_reference_brush_radius(
                float(self.reference_brush_slider.value())
            )
            self.image_view.set_reference_erase_mode(
                self.reference_eraser_button.isChecked()
            )
            overlay_index = self.overlay_combo.findData("foreground_mask")
            if overlay_index >= 0:
                self.overlay_combo.setCurrentIndex(overlay_index)
            self.statusBar().showMessage(
                "Other material: its learned colours and textures compete with "
                "foreground/background only where they are a better fit."
            )
        self._sync_reference_panel_visibility()

    @Slot(bool)
    def _physical_edge_editing_changed(self, enabled: bool) -> None:
        if enabled:
            for button in (
                self.background_point_button,
                self.foreground_point_button,
                self.background_exclusion_button,
                self.non_edge_button,
            ):
                with QSignalBlocker(button):
                    button.setChecked(False)
            with QSignalBlocker(self.annotate_instances_action):
                self.annotate_instances_action.setChecked(False)
            with QSignalBlocker(self.paint_background_action):
                self.paint_background_action.setChecked(False)
            self._ensure_reference_draft("physical_edge")
        self.image_view.set_physical_edge_reference_editing(enabled)
        if enabled:
            self.image_view.set_reference_brush_radius(
                float(self.reference_brush_slider.value())
            )
            self.image_view.set_reference_erase_mode(
                self.reference_eraser_button.isChecked()
            )
            self.statusBar().showMessage(
                "Physical edge: paint true seed boundaries; Snap follows nearby edge evidence."
            )
        self._sync_reference_panel_visibility()

    @Slot(bool)
    def _non_edge_editing_changed(self, enabled: bool) -> None:
        if enabled:
            for button in (
                self.background_point_button,
                self.foreground_point_button,
                self.background_exclusion_button,
                self.physical_edge_button,
            ):
                with QSignalBlocker(button):
                    button.setChecked(False)
            with QSignalBlocker(self.annotate_instances_action):
                self.annotate_instances_action.setChecked(False)
            with QSignalBlocker(self.paint_background_action):
                self.paint_background_action.setChecked(False)
            self._ensure_reference_draft("non_edge")
        self.image_view.set_non_edge_reference_editing(enabled)
        if enabled:
            self.image_view.set_reference_brush_radius(
                float(self.reference_brush_slider.value())
            )
            self.image_view.set_reference_erase_mode(
                self.reference_eraser_button.isChecked()
            )
            self.statusBar().showMessage(
                "Non-edge: mark coat-pattern or lighting transitions that are not "
                "physical boundaries; Snap follows nearby edge evidence."
            )
        self._sync_reference_panel_visibility()

    @Slot(bool)
    def _instance_annotation_editing_changed(self, enabled: bool) -> None:
        if enabled:
            with QSignalBlocker(self.background_point_button):
                self.background_point_button.setChecked(False)
            with QSignalBlocker(self.foreground_point_button):
                self.foreground_point_button.setChecked(False)
            with QSignalBlocker(self.paint_background_action):
                self.paint_background_action.setChecked(False)
            with QSignalBlocker(self.background_exclusion_button):
                self.background_exclusion_button.setChecked(False)
            with QSignalBlocker(self.foreground_exclusion_button):
                self.foreground_exclusion_button.setChecked(False)
            with QSignalBlocker(self.physical_edge_button):
                self.physical_edge_button.setChecked(False)
            with QSignalBlocker(self.non_edge_button):
                self.non_edge_button.setChecked(False)
        if enabled:
            self._ensure_instance_draft()
        self.image_view.set_instance_annotation_editing(enabled)
        if enabled:
            self.image_view.set_active_instance_id(self.instance_id_spin.value())
            self.image_view.set_reference_brush_radius(
                float(self.instance_brush_slider.value())
            )
            self._sync_instance_tool_settings()
            self.image_view.set_instance_annotation_tool(
                self._current_instance_tool()
            )
            overlay_index = self.overlay_combo.findData("instance_masks")
            if overlay_index >= 0:
                self.overlay_combo.setCurrentIndex(overlay_index)
            self.image_view.show()
            self._show_instance_tool_instructions(self._current_instance_tool())
        else:
            self.statusBar().showMessage("Seed instance annotation finished.")
        self._sync_reference_panel_visibility()

    @Slot(int)
    def _reference_brush_radius_changed(self, radius: int) -> None:
        self.reference_brush_label.setText(f"{radius} px")
        self.image_view.set_reference_brush_radius(float(radius))

    @Slot(int)
    def _instance_brush_radius_changed(self, radius: int) -> None:
        self.instance_brush_label.setText(f"{radius} px")
        self.image_view.set_reference_brush_radius(float(radius))

    @staticmethod
    def _populate_annotation_edge_sources(
        combo: QComboBox, *, include_net_physical: bool = False
    ) -> None:
        """Expose the actual calculated raster used by an assisted tool."""

        sources = [
            ("Thinned edge ridges", "ridges"),
            ("Thinned reference edge ridge", "reference_ridges"),
            ("Oriented edge traces", "traces"),
            ("Combined (ridge priority)", "adaptive"),
            ("Physical-edge probability", "physical"),
            ("Edge magnitude (broad)", "magnitude"),
        ]
        if include_net_physical:
            sources.insert(
                -1,
                ("Net physical-edge probability", "net_physical"),
            )
        for label, source in sources:
            combo.addItem(label, source)
        combo.setToolTip(
            "Select the exact edge raster used for snapping or as the fill "
            "barrier. Generic thinned ridges are the precise default; the "
            "reference-ridge option similarly thins learned physical-edge "
            "probability. Combined uses the per-pixel maximum of generic ridges "
            "and binary oriented traces, plus physical-edge probability and broad "
            "edge magnitude at 45% strength. Net physical-edge probability uses "
            "the node's live display-only subtraction of non-physical evidence; "
            "changing that existing node control does not rerun analysis. Edge "
            "magnitude alone is broader and can stop short of the visible ridge."
        )

    def _current_instance_tool(self) -> str:
        for button, tool in (
            (self.instance_paint_mode_button, "brush"),
            (self.instance_edge_trace_button, "edge_trace"),
            (self.instance_shape_guided_fill_button, "shape_guided_fill"),
            (self.instance_smart_fill_button, "smart_fill"),
            (self.instance_eraser_button, "eraser"),
        ):
            if button.isChecked():
                return tool
        return "brush"

    @Slot(str, bool)
    def _instance_tool_selected(self, tool: str, checked: bool) -> None:
        if not checked:
            return
        page = self.instance_tool_pages.get(tool)
        if page is not None:
            self.instance_tool_options_stack.setCurrentWidget(page)
        self._sync_instance_tool_settings()
        self.image_view.set_instance_annotation_tool(tool)
        self.image_view._layout_context_panel()
        self._show_instance_tool_instructions(tool)

    def _show_instance_tool_instructions(self, tool: str) -> None:
        instructions = {
            "brush": "Brush: paint an interior mark for the current seed; right-drag erases.",
            "edge_trace": "Trace edge: open segments apply as one-pixel edges; return to the cyan first anchor to preview and fill a closed loop.",
            "shape_guided_fill": "Shape fill: use the wheel to size the dotted oval prior, inspect the edge-refined Smart-fill preview, then click to apply.",
            "smart_fill": "Smart fill: move to preview locally adaptive growth, then click; an existing mark is optional.",
            "eraser": "Eraser: drag over instance labels to remove them.",
        }
        self.statusBar().showMessage(instructions[tool])

    @Slot(float)
    def _shape_fill_size_preference_changed(self, value: float) -> None:
        if hasattr(self, "shape_fill_preferred_scale_spin"):
            self.shape_fill_preferred_scale_spin.setValue(float(value))

    @Slot()
    def _sync_instance_tool_settings(self, *_unused) -> None:
        if not hasattr(self, "edge_trace_search_spin"):
            return
        diameter = float(
            getattr(
                self.image_view._analysis_result,
                "estimated_seed_diameter_px",
                100.0,
            )
        )
        preferred_scale = self.shape_fill_preferred_scale_spin.value()
        outward_half_life = self.shape_fill_outward_half_life_spin.value()
        outward_cutoff = self.shape_fill_outward_cutoff_spin.value()
        if outward_half_life > outward_cutoff:
            if self.sender() is self.shape_fill_outward_cutoff_spin:
                outward_half_life = outward_cutoff
                with QSignalBlocker(self.shape_fill_outward_half_life_spin):
                    self.shape_fill_outward_half_life_spin.setValue(
                        outward_half_life
                    )
            else:
                outward_cutoff = outward_half_life
                with QSignalBlocker(self.shape_fill_outward_cutoff_spin):
                    self.shape_fill_outward_cutoff_spin.setValue(outward_cutoff)
        self.image_view.set_edge_trace_options(
            EdgeTraceOptions(
                edge_source=str(self.edge_trace_evidence_combo.currentData()),
                search_radius_px=self.edge_trace_search_spin.value(),
                edge_attraction=self.edge_trace_attraction_spin.value() / 100.0,
                tangent_mode=str(self.edge_trace_tangent_combo.currentData()),
                tangent_weight=self.edge_trace_tangent_weight_spin.value() / 100.0,
                smoothing=self.edge_trace_smoothing_spin.value(),
                fill_closed_loops=self.edge_trace_fill_closed_checkbox.isChecked(),
            )
        )
        self.image_view.set_shape_guided_fill_options(
            ShapeGuidedFillOptions(
                shape=str(self.shape_fill_shape_combo.currentData()),
                preferred_scale=preferred_scale,
                auto_rotation=self.shape_fill_auto_rotation_checkbox.isChecked(),
                initial_rotation_degrees=self.shape_fill_rotation_spin.value(),
                maximum_axis_ratio=self.shape_fill_axis_ratio_spin.value(),
                boundary_smoothness=self.shape_fill_smoothness_spin.value()
                / 100.0,
                minimum_boundary_strength=(
                    self.shape_fill_min_strength_spin.value() / 100.0
                ),
                minimum_edge_coverage=(
                    self.shape_fill_min_coverage_spin.value() / 100.0
                ),
                minimum_sector_coverage=(
                    self.shape_fill_min_sector_spin.value() / 100.0
                ),
                maximum_unsupported_arc_fraction=(
                    self.shape_fill_max_gap_spin.value() / 100.0
                ),
                colour_tolerance_lab=self.shape_fill_colour_step_spin.value(),
                edge_barrier_threshold=self.shape_fill_barrier_spin.value()
                / 100.0,
                outward_penalty_half_life_fraction=outward_half_life,
                outward_hard_cutoff_fraction=outward_cutoff,
                maximum_added_pixels=self.shape_fill_pixel_limit_spin.value(),
            ),
            edge_source=str(self.shape_fill_evidence_combo.currentData()),
        )
        maximum_distance = int(
            np.clip(
                round(diameter * self.smart_fill_radius_spin.value()),
                4,
                4096,
            )
        )
        falloff_half_life = float(
            np.clip(
                diameter * self.smart_fill_falloff_spin.value(),
                1.0,
                4096.0,
            )
        )
        smart_fill_edge_source = str(
            self.smart_fill_evidence_combo.currentData()
        )
        self.image_view.set_smart_fill_options(
            SmartFillOptions(
                edge_source=(
                    "physical"
                    if smart_fill_edge_source == "net_physical"
                    else smart_fill_edge_source
                ),
                colour_tolerance_lab=self.smart_fill_colour_tolerance_spin.value(),
                edge_stop_threshold=self.smart_fill_edge_stop_spin.value() / 100.0,
                tunnel_strength=float(self.smart_fill_tunnel_combo.currentData()),
                maximum_distance_from_cursor_px=maximum_distance,
                falloff_half_life_px=falloff_half_life,
                maximum_added_pixels=self.smart_fill_pixel_limit_spin.value(),
                connectivity=int(self.smart_fill_connectivity_combo.currentData()),
            ),
            edge_source=smart_fill_edge_source,
        )

    @Slot(bool)
    def _reference_eraser_toggled(self, enabled: bool) -> None:
        self.image_view.set_reference_erase_mode(enabled)
        active_class = (
            "background"
            if self.background_point_button.isChecked()
            else "foreground"
            if self.foreground_point_button.isChecked()
            else "other material"
            if self.background_exclusion_button.isChecked()
            else "physical edge"
            if self.physical_edge_button.isChecked()
            else "non-edge"
            if self.non_edge_button.isChecked()
            else "reference"
        )
        self.statusBar().showMessage(
            f"{active_class.capitalize()} brush set to "
            f"{'erase' if enabled else 'paint'}."
        )

    def _start_reference_eraser(self, mask_kind: str) -> None:
        """Activate one painted mask and make left-drag erase from it."""

        buttons = {
            "background": self.background_point_button,
            "foreground": self.foreground_point_button,
            "background_exclusion": self.background_exclusion_button,
            "foreground_exclusion": self.foreground_exclusion_button,
        }
        button = buttons[mask_kind]
        if not button.isEnabled():
            return
        button.setChecked(True)
        self.reference_eraser_button.setChecked(True)

    @Slot()
    def _clear_active_reference_layer(self) -> None:
        mode = self.image_view._reference_point_mode
        if mode not in {
            "background",
            "foreground",
            "other",
        }:
            return
        self._clear_reference_class(mode)

    def _clear_reference_class(self, mode: str) -> None:
        key = self._current_image_key()
        material_modes = ("background", "foreground", "other")
        if key is None or mode not in material_modes:
            return
        context = "material"
        names = material_modes
        before = self._reference_group_state(key, context)
        after = list(before)
        after[names.index(mode)] = self._empty_current_image_mask()
        after_state = tuple(after)
        if not self._record_reference_undo(
            key,
            f"clear {mode.replace('_', ' ')}",
            context,
            before,
            after_state,
        ):
            self._sync_background_controls()
            self._restore_reference_paint_mode(mode)
            return
        self._set_reference_draft_state(key, context, after_state)
        self._reconcile_reference_draft(key)
        self._sync_reference_masks_to_view(key)
        self._sync_background_controls()
        self._restore_reference_paint_mode(mode)

    def _restore_reference_paint_mode(self, mode: str | None = None) -> None:
        """Make an emptied reference layer immediately drawable again."""

        self.reference_paint_mode_button.setChecked(True)
        self.image_view.set_reference_erase_mode(False)
        if mode is not None:
            setters = {
                "background": self.image_view.set_background_point_editing,
                "foreground": self.image_view.set_foreground_point_editing,
                "other": self.image_view.set_other_reference_editing,
            }
            setters[mode](True)

    @Slot(bool)
    def _instance_eraser_toggled(self, enabled: bool) -> None:
        if enabled:
            self._instance_tool_selected("eraser", True)

    @Slot(int)
    def _instance_id_changed(self, identifier: int) -> None:
        self.image_view.set_active_instance_id(identifier)
        self._update_instance_colour_swatch()
        self._sync_background_controls()

    @Slot(bool)
    def _show_selected_instance_toggled(self, enabled: bool) -> None:
        self.image_view.set_show_selected_instance_only(enabled)

    def _update_instance_colour_swatch(self) -> None:
        if not hasattr(self, "instance_colour_swatch"):
            return
        colour = self.image_view.instance_colour(self.instance_id_spin.value())
        self.instance_colour_swatch.setStyleSheet(
            f"background: {colour.name()}; border: 1px solid palette(mid);"
        )

    def _reference_brush_action(self) -> str:
        return "erase" if self.reference_eraser_button.isChecked() else "paint"

    @Slot(str, object)
    def _reference_mask_edited(self, class_name: str, mask) -> None:
        key = self._current_image_key()
        targets = {
            "background": self._draft_background_reference_masks,
            "foreground": self._draft_foreground_reference_masks,
            "other": self._draft_background_exclusion_masks,
        }
        if key is None or class_name not in targets:
            return
        context = "material"
        before = self._reference_group_state(key, context)
        background = self.image_view.reference_mask("background", copy=False)
        foreground = self.image_view.reference_mask("foreground", copy=False)
        other = self.image_view.reference_mask("other", copy=False)
        empty = self._empty_current_image_mask()
        provided = empty.copy() if mask is None else np.asarray(mask, dtype=bool)
        background_values = empty.copy() if background is None else background
        foreground_values = empty.copy() if foreground is None else foreground
        other_values = empty.copy() if other is None else other
        if class_name == "background":
            background_values = provided
            foreground_values = foreground_values & ~background_values
            other_values = other_values & ~background_values
        elif class_name == "foreground":
            foreground_values = provided
            background_values = background_values & ~foreground_values
            other_values = other_values & ~foreground_values
        else:
            other_values = provided
            foreground_values = foreground_values & ~other_values
            background_values = background_values & ~other_values
        after = (background_values, foreground_values, other_values)
        label = f"{class_name.replace('_', ' ')} stroke"
        if not self._record_reference_undo(
            key, label, context, before, after
        ):
            self._sync_reference_masks_to_view(key, render=False)
            self._sync_background_controls()
            return
        self._set_reference_draft_state(key, context, after)
        self._reconcile_reference_draft(key)
        # The view emitted a stable stroke snapshot.  Rebind it to that same
        # array so the just-finished draft has one owner rather than retaining
        # both the view's drawing buffer and a second controller snapshot.
        self._sync_reference_masks_to_view(key, render=False)
        self._sync_background_controls()
        self.statusBar().showMessage(
            f"{class_name.replace('_', ' ').capitalize()} references edited; "
            "Apply when painting is complete."
        )

    @Slot(object)
    def _instance_annotations_edited(self, annotations) -> None:
        key = self._current_image_key()
        if key is None:
            return
        before = self._instance_reference_state(key)
        before_origin = self._instance_reference_origin(key)
        values = (
            self._empty_current_instance_annotations()
            if annotations is None
            else np.asarray(annotations, dtype=np.uint16)
        )
        tool_labels = {
            "brush": "seed-instance brush stroke",
            "eraser": "seed-instance eraser stroke",
            "edge_trace": "edge-trace segment",
            "shape_guided_fill": "shape-guided fill",
            "smart_fill": "smart fill",
        }
        label = tool_labels.get(
            self._current_instance_tool(), "seed-instance edit"
        )
        if not self._record_instance_undo(
            key,
            label,
            before,
            (values,),
            before_origin=before_origin,
        ):
            self.image_view.set_instance_annotations(
                before[0], copy=False, render=False
            )
            self._sync_background_controls()
            return
        self._set_instance_draft_state(key, values, before_origin)
        self._sync_background_controls()
        self.statusBar().showMessage(
            "Seed instance marks edited; foreground references are unchanged. "
            "Apply the annotations when the interior marks are complete."
        )

    @Slot()
    def _use_instance_proposal_as_draft(self) -> None:
        """Replace the editable label draft with one calculated instance result."""

        key = self._current_image_key()
        attribute = self.instance_proposal_combo.currentData()
        result = self._analyses.get(key or "")
        proposal = None if result is None or not attribute else getattr(
            result, str(attribute), None
        )
        if key is None or proposal is None:
            return
        before = self._instance_reference_state(key)
        before_origin = self._instance_reference_origin(key)
        current = self._draft_instance_annotations.get(
            key, self._applied_instance_annotations.get(key)
        )
        if current is not None and np.any(current):
            answer = QMessageBox.question(
                self,
                "Replace annotation draft",
                "This replaces the current seed-instance annotations with an automatic "
                "prediction. Predictions are only a starting point: inspect and correct "
                "every seed before exporting labels. Continue?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        try:
            from seedvision.learning.export import annotation_proposal_to_corrected

            labels = annotation_proposal_to_corrected(result, proposal)
        except Exception as error:  # noqa: BLE001 - user-facing conversion boundary
            QMessageBox.critical(
                self, "Could not use instance result", str(error)
            )
            return
        if not self._record_instance_undo(
            key,
            "automatic instance draft",
            before,
            (labels,),
            before_origin=before_origin,
        ):
            self._sync_background_controls()
            return
        origin = f"pipeline:{attribute}"
        self._set_instance_draft_state(key, labels, origin)
        self.image_view.set_instance_annotations(
            self._draft_instance_annotations.get(
                key, self._applied_instance_annotations.get(key)
            ),
            copy=False,
        )
        next_identifier = min(
            np.iinfo(np.uint16).max, int(labels.max(initial=0)) + 1
        )
        with QSignalBlocker(self.instance_id_spin):
            self.instance_id_spin.setValue(max(1, next_identifier))
        self.image_view.set_active_instance_id(self.instance_id_spin.value())
        self._update_instance_colour_swatch()
        self._sync_background_controls()
        self.statusBar().showMessage(
            f"Loaded {int(labels.max(initial=0)):,} predicted instances as an editable "
            "draft; review every split, merge, omission, and contour before applying."
        )

    @Slot()
    def _new_instance_annotation(self) -> None:
        key = self._current_image_key()
        if key is None:
            return
        annotations = self._draft_instance_annotations.get(
            key, self._applied_instance_annotations.get(key)
        )
        identifiers = self._instance_ids(annotations)
        identifier = (max(identifiers) + 1) if identifiers else 1
        self.instance_id_spin.setValue(
            min(identifier, np.iinfo(np.uint16).max)
        )
        tool = self._current_instance_tool()
        tool_label = {
            "brush": "Brush",
            "edge_trace": "Trace edge",
            "shape_guided_fill": "Shape fill",
            "smart_fill": "Smart fill",
            "eraser": "Eraser",
        }.get(tool, tool.replace("_", " ").title())
        self.statusBar().showMessage(
            f"Seed {self.instance_id_spin.value()} selected; {tool_label} remains active."
        )

    @Slot()
    def _clear_current_instance_annotation(self) -> None:
        key = self._current_image_key()
        if key is None:
            return
        source = self._draft_instance_annotations.get(
            key, self._applied_instance_annotations.get(key)
        )
        if source is None:
            return
        before = self._instance_reference_state(key)
        before_origin = self._instance_reference_origin(key)
        values = np.asarray(source, dtype=np.uint16).copy()
        values[values == self.instance_id_spin.value()] = 0
        if not self._record_instance_undo(
            key,
            f"clear seed {self.instance_id_spin.value()}",
            before,
            (values,),
            before_origin=before_origin,
        ):
            return
        self._set_instance_draft_state(key, values, before_origin)
        self.image_view.set_instance_annotations(
            self._draft_instance_annotations.get(
                key, self._applied_instance_annotations.get(key)
            ),
            copy=False,
        )
        self._sync_background_controls()

    @Slot()
    def _clear_all_instance_annotations(self) -> None:
        key = self._current_image_key()
        if key is None:
            return
        before = self._instance_reference_state(key)
        before_origin = self._instance_reference_origin(key)
        values = self._empty_current_instance_annotations()
        if not self._record_instance_undo(
            key,
            "clear all seed instances",
            before,
            (values,),
            before_origin=before_origin,
        ):
            return
        self._set_instance_draft_state(key, values, "manual")
        self.image_view.set_instance_annotations(
            self._draft_instance_annotations.get(
                key, self._applied_instance_annotations.get(key)
            ),
            copy=False,
        )
        self._sync_background_controls()

    @Slot()
    def _revert_instance_annotations(self) -> None:
        key = self._current_image_key()
        if key is None:
            return
        applied = self._applied_instance_annotations.get(key)
        self._draft_instance_annotations.pop(key, None)
        self._draft_instance_annotation_origins.pop(key, None)
        self._instance_annotations_dirty.discard(key)
        self._instance_undo_histories.pop(key, None)
        self.image_view.set_instance_annotations(applied, copy=False)
        self._sync_background_controls()
        self.statusBar().showMessage("Discarded unapplied instance-annotation edits.")

    @Slot()
    def _clear_background_points(self) -> None:
        self._clear_reference_class("background")

    @Slot()
    def _clear_foreground_points(self) -> None:
        self._clear_reference_class("foreground")

    @Slot()
    def _clear_background_exclusion(self) -> None:
        self._clear_reference_class("other")

    @Slot()
    def _clear_foreground_exclusion(self) -> None:
        self._clear_reference_class("other")

    @Slot()
    def _revert_reference_masks(self) -> None:
        key = self._current_image_key()
        if key is None:
            return
        for draft in (
            self._draft_background_reference_masks,
            self._draft_foreground_reference_masks,
            self._draft_background_exclusion_masks,
            self._draft_foreground_exclusion_masks,
            self._draft_physical_edge_reference_masks,
            self._draft_non_edge_reference_masks,
        ):
            draft.pop(key, None)
        self._reference_masks_dirty.discard(key)
        self._reference_dirty_classes.pop(key, None)
        self._reference_undo_histories.pop(key, None)
        self._sync_reference_masks_to_view(key)
        self._sync_background_controls()
        self.statusBar().showMessage("Discarded unapplied reference-mask edits.")

    def _sync_reference_masks_to_view(
        self, key: str, *, render: bool = True
    ) -> None:
        self.image_view.set_reference_masks(
            self._draft_background_reference_masks.get(
                key, self._applied_background_reference_masks.get(key)
            ),
            self._draft_foreground_reference_masks.get(
                key, self._applied_foreground_reference_masks.get(key)
            ),
            self._draft_background_exclusion_masks.get(
                key, self._applied_background_exclusion_masks.get(key)
            ),
            self._draft_foreground_exclusion_masks.get(
                key, self._applied_foreground_exclusion_masks.get(key)
            ),
            physical_edge_mask=self._draft_physical_edge_reference_masks.get(
                key, self._applied_physical_edge_reference_masks.get(key)
            ),
            non_edge_mask=self._draft_non_edge_reference_masks.get(
                key, self._applied_non_edge_reference_masks.get(key)
            ),
            copy=False,
            render=render,
            normalize_material=False,
        )

    def _empty_current_image_mask(self) -> np.ndarray:
        image_size = self.image_view.image_size
        if image_size is None:
            return np.zeros((0, 0), dtype=bool)
        width, height = image_size
        return np.zeros((height, width), dtype=bool)

    def _empty_current_instance_annotations(self) -> np.ndarray:
        image_size = self.image_view.image_size
        if image_size is None:
            return np.zeros((0, 0), dtype=np.uint16)
        width, height = image_size
        return np.zeros((height, width), dtype=np.uint16)

    @Slot()
    def _apply_instance_annotations(self) -> None:
        key = self._current_image_key()
        if key is None or key not in self._instance_annotations_dirty:
            return
        self._instance_continuity_cache.pop(key, None)
        self._stop_instance_annotation_editing()
        annotations = self._draft_instance_annotations.get(key)
        if annotations is None or not np.any(annotations):
            self._applied_instance_annotations.pop(key, None)
            self._applied_instance_annotation_origins.pop(key, None)
        else:
            applied = np.asarray(annotations, dtype=np.uint16)
            applied.flags.writeable = False
            self._applied_instance_annotations[key] = applied
            self._applied_instance_annotation_origins[key] = (
                self._draft_instance_annotation_origins.get(key, "manual")
            )
        self._draft_instance_annotations.pop(key, None)
        self._draft_instance_annotation_origins.pop(key, None)
        self._instance_undo_histories.pop(key, None)
        self.image_view.set_instance_annotations(
            self._applied_instance_annotations.get(key),
            copy=False,
            render=False,
        )
        self._instance_annotations_dirty.discard(key)
        autosave_destination = self._persist_applied_reference_regions(
            automatic=True
        )
        affected = {
            "reference_layers",
            "instance_masks",
            *self.pipeline.downstream_from_port(
                "reference_layers", "annotated_seeds", recursive=True
            ),
        }
        self.pipeline.invalidate(affected)
        count = len(self._instance_ids(self._applied_instance_annotations.get(key)))
        self.pipeline.set_status(
            "reference_layers",
            NodeStatus.COMPLETE,
            f"{count:,} applied seed ID(s)" if count else "No applied references",
        )
        if self.pipeline.node("instance_masks").enabled:
            self.pipeline.set_status(
                "instance_masks",
                NodeStatus.WARNING,
                f"{count:,} annotated seed interiors; updating",
            )
        if self.pipeline.is_active("procedural_instances"):
            self.pipeline.set_status(
                "procedural_instances",
                NodeStatus.WARNING,
                f"{count:,} authoritative painted marker(s); updating",
            )
        if self.pipeline.node("unet_instances").enabled:
            self.pipeline.set_status(
                "unet_instances",
                NodeStatus.WARNING,
                f"{count:,} authoritative painted marker(s); updating decoder",
            )
        self.pipeline.set_status(
            "measurements", NodeStatus.BLOCKED, "Requires reviewed masks"
        )
        self.pipeline_canvas.refresh(affected)
        self.pipeline_inspector.refresh_status()
        computational = affected - {"reference_layers"}
        self._cache_dirty_nodes.setdefault(key, set()).update(computational)
        if computational:
            self._analyses.pop(key, None)
            self._analyze_current_image(dirty_nodes=affected)
        self._sync_background_controls()
        self.statusBar().showMessage(
            f"Applied {count:,} seed instance annotations."
            + (
                " Saved the applied reference snapshot automatically."
                if autosave_destination is not None
                else " Automatic disk save failed; the applied state remains in memory."
            )
            + (
                " Procedural instances are updating."
                if self.pipeline.is_active("procedural_instances")
                else " Instance masks are updating."
                if self.pipeline.node("instance_masks").enabled
                else " Instance-derived boundary evidence is updating."
            )
        )

    @Slot()
    def _export_learning_sample(self) -> None:
        """Persist applied labels and the exact corrected model inputs."""

        key = self._current_image_key()
        if key is None:
            return
        result = self._analyses.get(key)
        labels = self._applied_instance_annotations.get(key)
        path = self.image_view.image_path
        if result is None or labels is None or path is None or not np.any(labels):
            QMessageBox.warning(
                self,
                "Nothing to export",
                "Run the analysis and apply at least one complete seed-instance mask first.",
            )
            return
        identifier_base = path.stem
        identifier = identifier_base
        revision = 1
        destination = self._root / "learning-data"
        while (destination / f"{identifier}.features.npz").exists():
            revision += 1
            identifier = f"{identifier_base}_r{revision}"
        dialog = LearningExportDialog(
            default_directory=destination,
            default_identifier=identifier,
            default_group=identifier_base,
            default_revision=str(revision),
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        options = dialog.options()
        try:
            from seedvision.learning.export import export_analysis_sample

            sample = export_analysis_sample(
                result,
                labels,
                options.manifest_path,
                dataset_id=options.dataset_id,
                identifier=options.identifier,
                species=self.species_combo.currentText(),
                group=options.group,
                split=options.split,
                reviewed=options.reviewed,
                annotation_author=options.annotation_author,
                annotation_revision=options.annotation_revision,
                notes=(
                    "Exported from applied Seed Fiddle instance annotations; "
                    f"draft origin={self._applied_instance_annotation_origins.get(key, 'manual')}; "
                    + (options.notes or "no additional notes")
                ),
            )
        except Exception as error:  # noqa: BLE001 - user-facing export boundary
            QMessageBox.critical(self, "Learning export failed", str(error))
            return
        QMessageBox.information(
            self,
            "Learning sample exported",
            f"Exported {sample.identifier} to:\n{options.manifest_path}\n\n"
            + (
                "This sample is reviewed and eligible for its assigned split."
                if sample.reviewed
                else "This sample remains unreviewed. Load and inspect its label mask, then export a reviewed revision before training."
            ),
        )

    @Slot()
    def _save_instance_labels(self) -> None:
        key = self._current_image_key()
        path = self.image_view.image_path
        labels = self._applied_instance_annotations.get(key or "")
        if path is None or labels is None or not np.any(labels):
            QMessageBox.warning(self, "Nothing to save", "Apply seed-instance labels first.")
            return
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Save full-resolution seed-label mask",
            str(path.with_name(path.stem + ".seedlabels.png")),
            "16-bit label PNG (*.png)",
        )
        if not selected:
            return
        try:
            from seedvision.learning.data import write_label_image

            destination = write_label_image(selected, labels)
        except Exception as error:  # noqa: BLE001 - user-facing file boundary
            QMessageBox.critical(self, "Could not save seed labels", str(error))
            return
        self.statusBar().showMessage(f"Saved full-resolution seed labels to {destination}.")

    def _instance_mask_import_context(self):
        """Return the current analysed image context required by either importer."""

        key = self._current_image_key()
        result = self._analyses.get(key or "")
        image_path = self.image_view.image_path
        if key is None or result is None or image_path is None:
            QMessageBox.warning(
                self,
                "Analysis required",
                "Run the analysis before loading seed-instance references. "
                "The editor stores labels in corrected-image coordinates.",
            )
            return None
        corrected_shape = tuple(
            int(value) for value in result.calibration.corrected_bgr.shape[:2]
        )
        return image_path, result, corrected_shape

    @Slot()
    def _load_instance_reference_mask(self) -> None:
        """Prefer a source-bound bundled reference, then offer a file chooser."""

        context = self._instance_mask_import_context()
        if context is None:
            return
        image_path, result, corrected_shape = context
        try:
            imported = load_bundled_instance_mask(
                self._root,
                image_path,
                source_shape=read_source_raster_shape(image_path),
                corrected_shape=corrected_shape,
                source_to_corrected=result.calibration.affine_matrix,
            )
        except InstanceMaskImportError as error:
            QMessageBox.critical(
                self, "Could not load bundled seed references", str(error)
            )
            return

        if imported is None:
            self._choose_instance_mask_file()
            return
        self._install_imported_instance_mask(imported)

    @Slot()
    def _choose_instance_mask_file(self) -> None:
        """Explicitly import a corrected-coordinate file, bypassing the bundle."""

        context = self._instance_mask_import_context()
        if context is None:
            return
        image_path, _result, corrected_shape = context
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Choose full-resolution seed-instance mask",
            str(image_path.parent),
            "Lossless categorical masks (*.png *.tif *.tiff *.npz)",
        )
        if not selected:
            return
        try:
            imported = load_corrected_instance_mask(selected, corrected_shape)
        except InstanceMaskImportError as error:
            QMessageBox.critical(
                self, "Could not load seed-instance mask", str(error)
            )
            return
        self._install_imported_instance_mask(imported)

    # Compatibility for callers and tests written before the seed-reference
    # button exposed the safer bundled-first path.
    def _load_instance_labels(self) -> None:
        self._load_instance_reference_mask()

    def _install_imported_instance_mask(
        self, imported: ImportedInstanceMask
    ) -> bool:
        """Install one already-validated mask as a reviewable, undoable draft."""

        key = self._current_image_key()
        if key is None:
            return False
        labels = np.asarray(imported.labels, dtype=np.uint16)
        before = self._instance_reference_state(key)
        before_origin = self._instance_reference_origin(key)
        if np.array_equal(before[0], labels):
            self.statusBar().showMessage(
                "The selected seed-instance mask is already the current draft."
            )
            return False
        if np.any(before[0]):
            answer = QMessageBox.question(
                self,
                "Replace seed-instance draft",
                "This replaces the current seed-instance draft with the loaded "
                "reference mask. The replacement is undoable until Apply + save. "
                "Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return False
        if not self._record_instance_undo(
            key,
            "imported seed-instance reference mask",
            before,
            (labels,),
            before_origin=before_origin,
        ):
            self._sync_background_controls()
            return False
        self._set_instance_draft_state(key, labels, imported.origin)
        self.image_view.set_instance_annotations(
            self._draft_instance_annotations.get(
                key, self._applied_instance_annotations.get(key)
            ),
            copy=False,
        )
        # Entering annotation mode exposes the draft but deliberately preserves
        # the selected seed, painter, tool parameters, and visibility choices.
        self.annotate_instances_action.setChecked(True)
        self._sync_background_controls()
        review_text = (
            "reviewed"
            if imported.reviewed is True
            else "unreviewed; inspect every instance"
        )
        external = (
            ""
            if imported.external_reference_count is None
            else f", including {imported.external_reference_count:,} external reference(s)"
        )
        self.statusBar().showMessage(
            f"Loaded {imported.instance_count:,} seed ID(s){external} as an "
            f"undoable draft ({review_text}). IDs were preserved exactly."
        )
        return True

    @Slot()
    def _audit_learning_dataset(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Choose learning manifest",
            str(self._root / "learning-data" / "manifest.json"),
            "JSON (*.json)",
        )
        if not selected:
            return
        try:
            from seedvision.learning.data import audit_manifest

            audit = audit_manifest(selected)
        except Exception as error:  # noqa: BLE001 - user-facing audit boundary
            QMessageBox.critical(self, "Learning dataset audit failed", str(error))
            return
        method = QMessageBox.information if audit["valid"] else QMessageBox.warning
        method(self, "Learning dataset audit", format_learning_audit(audit))

    @Slot()
    def _start_learning_training(self) -> None:
        if (
            self._learning_training_task is not None
            or self._procedural_fit_task is not None
            or self._active_tasks
        ):
            QMessageBox.information(
                self,
                "CUDA worker is busy",
                "Wait for the current analysis or training run to finish.",
            )
            return
        dialog = LearningTrainingDialog(project_root=self._root, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        request = dialog.request()
        try:
            from seedvision.learning.data import LearningManifest, audit_manifest

            audit = audit_manifest(request.manifest_path)
            manifest = LearningManifest.load(request.manifest_path)
        except Exception as error:  # noqa: BLE001 - user-facing audit boundary
            QMessageBox.critical(self, "Learning dataset audit failed", str(error))
            return
        errors = list(audit["errors"])
        if audit["splits"].get("train", 0) < 1:
            errors.append("At least one reviewed training sample is required.")
        if audit["splits"].get("validation", 0) < 1:
            errors.append("At least one independently grouped validation sample is required.")
        supervised = tuple(
            sample for sample in manifest.samples if sample.split in {"train", "validation"}
        )
        if any(not sample.reviewed for sample in supervised):
            errors.append("Every training/validation sample must be marked human-reviewed.")
        if errors:
            failed_audit = dict(audit)
            failed_audit["errors"] = errors
            failed_audit["valid"] = False
            QMessageBox.warning(
                self, "Dataset is not ready for training", format_learning_audit(failed_audit)
            )
            return
        task = _LearningTrainingTask(request)
        progress = QProgressDialog(
            "Preparing reviewed learning tiles…",
            "Cancel",
            0,
            request.configuration.epochs,
            self,
        )
        progress.setWindowTitle("Training learned seed model")
        progress.setWindowModality(Qt.WindowModality.NonModal)
        progress.setAutoClose(False)
        progress.setMinimumDuration(0)
        task.signals.progress.connect(self._learning_training_progressed)
        task.signals.completed.connect(self._learning_training_completed)
        task.signals.failed.connect(self._learning_training_failed)
        task.signals.cancelled.connect(self._learning_training_cancelled)
        progress.canceled.connect(task.cancel)
        progress.canceled.connect(
            lambda: progress.setLabelText(
                "Cancelling after the current training batch…"
            )
        )
        self._learning_training_task = task
        self._learning_training_progress = progress
        self.train_learning_model_action.setEnabled(False)
        self._update_analysis_availability()
        self._sync_background_controls()
        progress.show()
        self.statusBar().showMessage(
            f"Training {request.configuration.family} from {request.manifest_path.name}…"
        )
        self._thread_pool.start(task)

    @Slot(int, int, float, float)
    def _learning_training_progressed(
        self, epoch: int, maximum: int, train_loss: float, validation_loss: float
    ) -> None:
        progress = self._learning_training_progress
        if progress is None:
            return
        progress.setMaximum(maximum)
        progress.setValue(epoch)
        progress.setLabelText(
            f"Epoch {epoch}/{maximum}\nTraining loss {train_loss:.5f}; validation loss {validation_loss:.5f}"
        )

    @Slot(object)
    def _learning_training_completed(self, report) -> None:
        task = self._learning_training_task
        request = task.request if task is not None else None
        self._finish_learning_training_ui()
        if request is None:
            return
        node_id = (
            "unet_instances"
            if request.configuration.family == "unet_watershed"
            else "stardist_instances"
        )
        affected = set(
            self.pipeline.set_parameter(
                node_id, "checkpoint_path", str(request.output_path.resolve())
            )
        )
        if request.activate_after_training:
            affected.update(self.pipeline.set_enabled(node_id, True))
        for image_key in set(self._analysis_caches) | set(self._analyses):
            self._cache_dirty_nodes.setdefault(image_key, set()).update(affected)
            self._analyses.pop(image_key, None)
        self.pipeline_canvas.refresh(affected or (node_id,))
        self.pipeline_canvas.select_node(node_id)
        self.pipeline_inspector.set_node(self.pipeline.node(node_id))
        QMessageBox.information(
            self,
            "Training completed",
            f"Best epoch: {report['best_epoch']}\n"
            f"Best validation loss: {report['best_validation_loss']:.6f}\n"
            f"Checkpoint: {request.output_path}\n\n"
            "This checkpoint still requires evaluation on a frozen, independently reviewed test split.",
        )
        key = self._current_image_key()
        if request.activate_after_training and key is not None and key in self._analysis_caches:
            self._analyze_current_image(dirty_nodes=affected)

    @Slot(str)
    def _learning_training_failed(self, error: str) -> None:
        self._finish_learning_training_ui()
        self._start_pending_analysis()
        QMessageBox.critical(self, "Learned-model training failed", error)

    @Slot()
    def _learning_training_cancelled(self) -> None:
        self._finish_learning_training_ui()
        self._start_pending_analysis()
        self.statusBar().showMessage("Learned-model training cancelled.")

    def _finish_learning_training_ui(self) -> None:
        if self._learning_training_progress is not None:
            self._learning_training_progress.close()
        self._learning_training_progress = None
        self._learning_training_task = None
        self.train_learning_model_action.setEnabled(True)
        self._update_analysis_availability()
        self._sync_background_controls()

    @Slot(str, str, object)
    def _pipeline_node_action_requested(
        self, node_id: str, action_id: str, payload: object
    ) -> None:
        if (
            node_id == "procedural_instances"
            and action_id == PipelineInspector.PROCEDURAL_FIT_ACTION
        ):
            values = payload if isinstance(payload, dict) else {}
            self._start_procedural_fit(
                float(values.get("false_positive_weight", 2.0)),
                float(values.get("overreach_distance_scale_fraction", 0.50)),
                bool(values.get("annotations_are_complete", False)),
            )

    def _sync_procedural_fit_controls(self) -> None:
        """Explain whether the current image supplies safe fitting targets."""

        if not hasattr(self, "pipeline_inspector"):
            return
        task = self._procedural_fit_task
        if task is not None:
            self.pipeline_inspector.set_procedural_fit_busy(
                True, "Evaluating bounded procedural-setting proposals…"
            )
            return
        self.pipeline_inspector.set_procedural_fit_busy(False)
        key = self._current_image_key()
        if key is None:
            self.pipeline_inspector.set_procedural_fit_eligibility(
                False, "Open and analyse an image before fitting."
            )
            return
        if not self.pipeline.is_active("procedural_instances") or not (
            self.pipeline.node("procedural_instances").enabled
        ):
            self.pipeline_inspector.set_procedural_fit_eligibility(
                False, "Enable the procedural seed-separation node before fitting."
            )
            return
        if key in self._instance_annotations_dirty:
            self.pipeline_inspector.set_procedural_fit_eligibility(
                False,
                "Apply + save the current seed-instance draft before fitting; "
                "unapplied edits are never training targets.",
            )
            return
        annotations = self._applied_instance_annotations.get(key)
        if annotations is None or not np.any(annotations):
            self.pipeline_inspector.set_procedural_fit_eligibility(
                False, "Apply at least one complete seed-instance annotation first."
            )
            return
        continuity = self._instance_continuity_summary(key, annotations)
        if continuity.disconnected:
            self.pipeline_inspector.set_procedural_fit_eligibility(
                False,
                "Repair disconnected seed IDs before fitting; each target ID must "
                "describe one complete contiguous seed.",
            )
            return
        if key not in self._analyses:
            self.pipeline_inspector.set_procedural_fit_eligibility(
                False, "Run the current analysis before fitting procedural settings."
            )
            return
        if self._active_tasks or self._learning_training_task is not None:
            self.pipeline_inspector.set_procedural_fit_eligibility(
                False, "Wait for the current analysis or training job to finish."
            )
            return
        self.pipeline_inspector.set_procedural_fit_eligibility(True)

    @Slot(float, float, bool)
    def _start_procedural_fit(
        self,
        false_positive_weight: float = 2.0,
        overreach_distance_scale_fraction: float = 0.50,
        annotations_are_complete: bool = False,
    ) -> None:
        """Search current-image procedural settings without mutating the graph."""

        self._sync_procedural_fit_controls()
        if self._procedural_fit_task is not None:
            return
        key = self._current_image_key()
        if key is None:
            return
        annotations = self._applied_instance_annotations.get(key)
        result = self._analyses.get(key)
        if annotations is None or result is None or not np.any(annotations):
            self._sync_procedural_fit_controls()
            return
        continuity = self._instance_continuity_summary(key, annotations)
        if continuity.disconnected or key in self._instance_annotations_dirty:
            self._sync_procedural_fit_controls()
            return

        warnings = [
            "This is an image-local, in-sample adjustment—not scientific validation.",
            "The annotations are withheld as watershed markers during fitting. "
            + (
                "You declared that the applied annotations cover the whole dish, so "
                "every predicted object will be scored, including standalone "
                "background false objects."
                if annotations_are_complete
                else "This is a partial review: predictions that never overlap an "
                "annotated seed are not scored, but every annotated ID must completely "
                "outline its seed."
            ),
            "Overreach is distance weighted: an outside pixel at "
            f"{overreach_distance_scale_fraction:.2f} seed diameters costs "
            f"{false_positive_weight:.2f}× a missing pixel, immediately adjacent "
            "pixels cost very little, and farther pixels cost exponentially more.",
            "Although the fit is measured on this one image, accepting its proposal "
            "changes the procedural node settings globally for every image.",
        ]
        origin = self._applied_instance_annotation_origins.get(key, "manual")
        if origin.startswith("pipeline:"):
            warnings.append(
                "These labels began as an automatic pipeline result. Correct every "
                "error before fitting, or the search can reinforce that result."
            )
        if len(continuity.identifiers) == 1:
            warnings.append(
                "Only one annotated seed is available, so the fitted settings are "
                "especially likely to overfit."
            )
        answer = QMessageBox.question(
            self,
            "Fit procedural settings to this image?",
            "\n\n".join(warnings),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        offset_x, offset_y = result.crop_offset
        layer_shape = tuple(int(value) for value in result.layers.valid_mask.shape[-2:])
        height, width = layer_shape
        local_annotations = np.asarray(
            annotations[
                offset_y : offset_y + height,
                offset_x : offset_x + width,
            ],
            dtype=np.uint16,
        )
        if not np.any(local_annotations):
            QMessageBox.warning(
                self,
                "No annotations in the dish region",
                "The applied seed IDs do not overlap the detected dish crop.",
            )
            return

        task = _ProceduralFitTask(
            result,
            local_annotations,
            self._procedural_settings(),
            false_positive_weight=false_positive_weight,
            overreach_distance_scale_fraction=(
                overreach_distance_scale_fraction
            ),
            annotations_are_complete=annotations_are_complete,
            image_key=key,
            pipeline_revision=self.pipeline.revision,
            annotation_source=annotations,
        )
        progress = QProgressDialog(
            "Evaluating procedural setting 0/33…",
            "Cancel",
            0,
            33,
            self,
        )
        progress.setWindowTitle("Fit procedural seed separation")
        progress.setWindowModality(Qt.WindowModality.NonModal)
        progress.setAutoClose(False)
        progress.setMinimumDuration(0)
        progress.canceled.connect(task.cancel)
        progress.canceled.connect(
            lambda: progress.setLabelText(
                "Cancelling after the current procedural evaluation…"
            )
        )
        task.signals.progress.connect(self._procedural_fit_progressed)
        task.signals.completed.connect(self._procedural_fit_completed)
        task.signals.failed.connect(self._procedural_fit_failed)
        task.signals.cancelled.connect(self._procedural_fit_cancelled)
        self._procedural_fit_task = task
        self._procedural_fit_progress = progress
        self.pipeline_inspector.set_procedural_fit_busy(
            True, "Evaluating bounded procedural-setting proposals…"
        )
        self._update_analysis_availability()
        self._sync_background_controls()
        progress.show()
        self.statusBar().showMessage(
            f"Fitting procedural settings to {len(continuity.identifiers):,} "
            "applied seed annotations "
            f"({'whole-dish' if annotations_are_complete else 'partial-review'} scoring)…"
        )
        self._thread_pool.start(task)

    @Slot(int, int, float)
    def _procedural_fit_progressed(
        self, evaluation: int, maximum: int, loss: float
    ) -> None:
        progress = self._procedural_fit_progress
        if progress is not None:
            progress.setMaximum(maximum)
            progress.setValue(evaluation)
            progress.setLabelText(
                f"Evaluating procedural setting {evaluation}/{maximum}\n"
                f"Current asymmetric loss: {loss:.4f}"
            )
        self.pipeline_inspector.set_procedural_fit_busy(
            True,
            f"Evaluation {evaluation}/{maximum}; current asymmetric loss {loss:.4f}.",
        )

    @Slot(object)
    def _procedural_fit_completed(self, report) -> None:
        task = self._procedural_fit_task
        if task is None:
            return
        stale = (
            self.pipeline.revision != task.pipeline_revision
            or self._current_image_key() != task.image_key
            or self._applied_instance_annotations.get(task.image_key)
            is not task.annotation_source
        )
        self._finish_procedural_fit_ui()
        if stale:
            self.pipeline_inspector.set_procedural_fit_result_summary(
                "Discarded a fit whose image, annotations, or pipeline changed."
            )
            self.statusBar().showMessage(
                "Discarded superseded procedural fitting results."
            )
            self._start_pending_analysis()
            return

        initial = report.initial_score
        proposed = report.proposed_score
        coverage_label = (
            "Whole-dish metrics"
            if task.annotations_are_complete
            else "Annotated-subset metrics (disjoint predictions unscored)"
        )
        summary = (
            f"{coverage_label}: loss {initial.loss:.4f} → {proposed.loss:.4f}; "
            f"precision {proposed.pixel_precision:.1%}, "
            f"recall {proposed.pixel_recall:.1%}; "
            f"FP {proposed.false_positive_pixels:,}, "
            "distance-weighted FP equivalents "
            f"{proposed.distance_weighted_false_positive_pixels:,.1f} "
            f"(scale {proposed.overreach_distance_scale_px:.1f}px), "
            f"FN {proposed.false_negative_pixels:,}; "
            f"{report.evaluations} evaluations."
        )
        if not report.improved:
            self.pipeline_inspector.set_procedural_fit_result_summary(
                "No tested parameter change improved the current asymmetric loss. "
                + summary
            )
            QMessageBox.information(
                self,
                "Procedural fit found no improvement",
                summary
                + "\n\nThe active node settings were not changed.",
            )
            self._start_pending_analysis()
            return

        node = self.pipeline.node("procedural_instances")
        changes = {
            key: getattr(report.proposed_settings, key)
            for key in node.parameters
            if getattr(report.proposed_settings, key) != node.parameters[key]
        }
        labels = {spec.key: spec.label for spec in node.parameter_specs}
        changed_lines = "\n".join(
            f"• {labels.get(key, key)}: {node.parameters[key]:.4g} → {value:.4g}"
            for key, value in changes.items()
        )
        answer = QMessageBox.question(
            self,
            "Apply fitted procedural settings?",
            summary
            + "\n\nProposed changes:\n"
            + (changed_lines or "No parameter values changed.")
            + "\n\nThe fit was measured on this image, but accepting applies these "
            "settings globally to the procedural node for every image. Apply them?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes or not changes:
            self.pipeline_inspector.set_procedural_fit_result_summary(
                "Kept current settings. " + summary
            )
            self._start_pending_analysis()
            return

        affected = set(
            self.pipeline.set_parameters("procedural_instances", changes)
        )
        for image_key in set(self._analysis_caches) | set(self._analyses):
            self._cache_dirty_nodes.setdefault(image_key, set()).update(affected)
            self._analyses.pop(image_key, None)
        self.pipeline_canvas.refresh(affected or {"procedural_instances"})
        self.pipeline_inspector.set_node(
            self.pipeline.node("procedural_instances")
        )
        self.pipeline_inspector.set_procedural_fit_result_summary(
            "Applied fitted settings. " + summary
        )
        current_key = self._current_image_key()
        if current_key is not None and current_key in self._analysis_caches:
            self._analyze_current_image(dirty_nodes=affected)
        else:
            self._start_pending_analysis()
        self.statusBar().showMessage(
            "Applied annotation-fitted procedural settings; recomputing the "
            "procedural node and its dependents."
        )

    @Slot(str)
    def _procedural_fit_failed(self, error: str) -> None:
        self._finish_procedural_fit_ui()
        self._start_pending_analysis()
        QMessageBox.critical(self, "Procedural parameter fit failed", error)

    @Slot()
    def _procedural_fit_cancelled(self) -> None:
        self._finish_procedural_fit_ui()
        self._start_pending_analysis()
        self.pipeline_inspector.set_procedural_fit_result_summary(
            "Fitting cancelled; active settings were not changed."
        )
        self.statusBar().showMessage("Procedural parameter fitting cancelled.")

    def _finish_procedural_fit_ui(self) -> None:
        if self._procedural_fit_progress is not None:
            self._procedural_fit_progress.close()
        self._procedural_fit_progress = None
        self._procedural_fit_task = None
        self.pipeline_inspector.set_procedural_fit_busy(False)
        self._update_analysis_availability()
        self._sync_background_controls()

    @Slot()
    def _apply_reference_masks(self) -> None:
        key = self._current_image_key()
        if key is None or key not in self._reference_masks_dirty:
            return
        self._stop_reference_point_editing()
        dirty_classes = self._reference_dirty_classes.pop(key, set())
        for draft, applied in (
            (
                self._draft_background_reference_masks,
                self._applied_background_reference_masks,
            ),
            (
                self._draft_foreground_reference_masks,
                self._applied_foreground_reference_masks,
            ),
            (
                self._draft_background_exclusion_masks,
                self._applied_background_exclusion_masks,
            ),
            (
                self._draft_foreground_exclusion_masks,
                self._applied_foreground_exclusion_masks,
            ),
            (
                self._draft_physical_edge_reference_masks,
                self._applied_physical_edge_reference_masks,
            ),
            (
                self._draft_non_edge_reference_masks,
                self._applied_non_edge_reference_masks,
            ),
        ):
            if key not in draft:
                continue
            mask = draft.pop(key)
            if mask is None or not np.any(mask):
                applied.pop(key, None)
            else:
                stored = np.asarray(mask, dtype=bool)
                stored.flags.writeable = False
                applied[key] = stored
        self._normalize_applied_reference_classes(key)
        self._reference_undo_histories.pop(key, None)
        self._sync_reference_masks_to_view(key, render=False)
        self._reference_masks_dirty.discard(key)
        autosave_destination = self._persist_applied_reference_regions(
            automatic=True
        )
        material_changed = bool(
            dirty_classes
            & {"background", "foreground", "other", "background_exclusion", "foreground_exclusion"}
        )
        affected = {"reference_layers"}
        if material_changed or not dirty_classes:
            for port_id in ("background", "foreground", "other"):
                affected.update(
                    self.pipeline.downstream_from_port(
                        "reference_layers", port_id, recursive=True
                    )
                )
        self.pipeline.invalidate(affected)
        background_count = self._mask_pixel_count(
            self._applied_background_reference_masks.get(key)
        )
        foreground_count = self._mask_pixel_count(
            self._applied_foreground_reference_masks.get(key)
        )
        background_exclusion_count = self._mask_pixel_count(
            self._applied_background_exclusion_masks.get(key)
        )
        foreground_exclusion_count = self._mask_pixel_count(
            self._applied_foreground_exclusion_masks.get(key)
        )
        self.pipeline.set_status(
            "reference_layers",
            NodeStatus.COMPLETE,
            f"BG {background_count:,}; FG {foreground_count:,}; "
            f"Other {max(background_exclusion_count, foreground_exclusion_count):,}",
        )
        detail = (
            f"{background_count:,} confirmed reference pixels; updating"
            if background_count
            else "Automatic selection; updating"
        )
        if material_changed or not dirty_classes:
            self.pipeline.set_status("background_likelihood", NodeStatus.WARNING, detail)
            self.pipeline.set_status(
                "foreground_segmentation",
                NodeStatus.WARNING,
                f"{foreground_count:,} confirmed reference pixels; updating",
            )
            self.pipeline.set_status(
                "refined_background_likelihood", NodeStatus.IDLE, "Updating"
            )
            self.pipeline.set_status(
                "foreground_noise_likelihood", NodeStatus.IDLE, "Updating"
            )
            self.pipeline.set_status("instance_masks", NodeStatus.IDLE, "Updating")
            self.pipeline.set_status(
                "measurements", NodeStatus.BLOCKED, "Requires reviewed masks"
            )
        self.pipeline_canvas.refresh(affected)
        self.pipeline_inspector.refresh_status()
        computational = affected - {"reference_layers"}
        if computational:
            self._cache_dirty_nodes.setdefault(key, set()).update(computational)
            self._analyses.pop(key, None)
        self._sync_background_controls()
        if computational:
            self._analyze_current_image()
        else:
            self.image_view.refresh_analysis()
            self.statusBar().showMessage(
                "Applied reference edits and saved the complete applied snapshot "
                "automatically."
                if autosave_destination is not None
                else "Applied reference edits in memory; automatic disk save failed."
            )

    def _normalize_applied_reference_classes(self, key: str) -> None:
        """Enforce the two categorical-layer exclusivity invariants."""

        empty = self._empty_current_image_mask()
        other = (
            np.asarray(
                self._applied_background_exclusion_masks.get(key, empty), dtype=bool
            )
            | np.asarray(
                self._applied_foreground_exclusion_masks.get(key, empty), dtype=bool
            )
        )
        foreground = np.asarray(
            self._applied_foreground_reference_masks.get(key, empty), dtype=bool
        ) & ~other
        background = np.asarray(
            self._applied_background_reference_masks.get(key, empty), dtype=bool
        ) & ~other & ~foreground
        physical = np.asarray(
            self._applied_physical_edge_reference_masks.get(key, empty), dtype=bool
        )
        non_edge = np.asarray(
            self._applied_non_edge_reference_masks.get(key, empty), dtype=bool
        )
        physical = physical & ~non_edge
        for values, stores in (
            (background, (self._applied_background_reference_masks,)),
            (foreground, (self._applied_foreground_reference_masks,)),
            (
                other,
                (
                    self._applied_background_exclusion_masks,
                    self._applied_foreground_exclusion_masks,
                ),
            ),
            (physical, (self._applied_physical_edge_reference_masks,)),
            (non_edge, (self._applied_non_edge_reference_masks,)),
        ):
            if np.any(values):
                stored = np.asarray(values, dtype=bool)
                stored.flags.writeable = False
                for store in stores:
                    store[key] = stored
            else:
                for store in stores:
                    store.pop(key, None)

    @Slot(int)
    def _overlay_changed(self, index: int) -> None:
        del index
        mode = str(self.overlay_combo.currentData())
        self._sync_overlay_display_controls(mode)
        self.image_view.set_overlay_mode(mode)
        self._update_overlay_legend(mode)
        owner = _overlay_node_owner(mode)
        if owner is None:
            self.overlay_owner_label.setText("No graph node")
            self._sync_inspector_overlay_options()
            return
        self.overlay_owner_label.setText(self.pipeline.node(owner).title)
        if (
            owner != self._selected_pipeline_node
            and not self._selecting_overlay_from_node
        ):
            self._selecting_node_from_overlay = True
            try:
                self.pipeline_canvas.select_node(owner)
            finally:
                self._selecting_node_from_overlay = False
        self._sync_inspector_overlay_options()
        self._sync_procedural_fit_controls()

    @Slot(str)
    def _inspector_overlay_selected(self, mode: str) -> None:
        index = self.overlay_combo.findData(mode)
        if index < 0:
            return
        self._selecting_overlay_from_node = True
        try:
            self.overlay_combo.setCurrentIndex(index)
        finally:
            self._selecting_overlay_from_node = False

    @Slot(int)
    def _overlay_opacity_changed(self, value: int) -> None:
        self.overlay_opacity_label.setText(f"{value}%")
        self.image_view.set_overlay_opacity(value / 100.0)

    @Slot(int)
    def _hsv_value_changed(self, value: int) -> None:
        self.hsv_value_label.setText(f"{value}%")
        mode = str(self.overlay_combo.currentData())
        if mode == "background_colour_gamut":
            self._hsv_gamut_values["background"] = value
        elif mode == "foreground_colour_gamut":
            self._hsv_gamut_values["foreground"] = value
        self.image_view.set_hsv_gamut_value(value / 100.0)

    @Slot()
    def _hsv_peak_requested(self) -> None:
        mode = str(self.overlay_combo.currentData())
        class_name = {
            "background_colour_gamut": "background",
            "foreground_colour_gamut": "foreground",
        }.get(mode)
        if class_name is None:
            return
        peak = self.image_view.dominant_hsv_gamut_value(class_name)
        if peak is None:
            return
        self.hsv_value_slider.setValue(round(peak * 100.0))

    def _sync_overlay_display_controls(self, mode: str) -> None:
        gamut_modes = {
            "background_colour_gamut": (
                "background",
                "background_likelihood",
            ),
            "foreground_colour_gamut": (
                "foreground",
                "foreground_segmentation",
            ),
        }
        gamut = mode in gamut_modes
        full_pane = gamut or mode == "reference_texture_prototypes"
        for action in (
            self.opacity_toolbar_label_action,
            self.overlay_opacity_slider_action,
            self.overlay_opacity_label_action,
        ):
            action.setVisible(not full_pane)
        for control in (
            self.opacity_toolbar_label,
            self.overlay_opacity_slider,
            self.overlay_opacity_label,
        ):
            control.setVisible(not full_pane)
        for action in (
            self.hsv_value_toolbar_label_action,
            self.hsv_value_slider_action,
            self.hsv_value_label_action,
            self.hsv_peak_button_action,
        ):
            action.setVisible(gamut)
        for control in (
            self.hsv_value_toolbar_label,
            self.hsv_value_slider,
            self.hsv_value_label,
            self.hsv_peak_button,
        ):
            control.setVisible(gamut)
        if gamut:
            class_name, node_id = gamut_modes[mode]
            self.image_view.set_colour_gamut_parameters(
                class_name, self.pipeline.node(node_id).parameters
            )
            peak = self.image_view.dominant_hsv_gamut_value(class_name)
            stored_value = self._hsv_gamut_values[class_name]
            if stored_value is None and peak is not None:
                stored_value = round(peak * 100.0)
                self._hsv_gamut_values[class_name] = stored_value
            if stored_value is not None:
                with QSignalBlocker(self.hsv_value_slider):
                    self.hsv_value_slider.setValue(stored_value)
                self.hsv_value_label.setText(f"{stored_value}%")
            self.hsv_peak_button.setEnabled(peak is not None)
            self.image_view.set_hsv_gamut_value(
                self.hsv_value_slider.value() / 100.0
            )

    def _update_overlay_legend(self, mode: str) -> None:
        if (
            mode in {"background_likelihood", "refined_background_likelihood"}
            and not self.pipeline.node("background_likelihood").enabled
        ):
            self.overlay_legend_label.setText(
                "Background colour analysis is disabled; this layer is not generated."
            )
            return
        legends = {
            "raw_image": (
                "Original decoded image before deskew, colour balance, or scale overlay."
            ),
            "calibrated_image": (
                "Deskewed, neutral-balanced analysis image with its assigned "
                "absolute metric scale bar."
            ),
            "deskew_colour": (
                "Deskewed, gamut-safe colour-balanced image. Cyan outlines the "
                "corrected card extent; yellow outlines corrected swatch locations."
            ),
            "colour_reference": (
                "Raw image before calibration. Cyan: fitted colour-card extent. Yellow: automatically "
                "located 4×6 swatch sampling regions."
            ),
            "ruler_detection": (
                "Magenta: the calibrated span between the large 0 cm and "
                "15 cm scale dashes. White circles: those scale endpoints."
            ),
            "layout_detection": (
                "Petri dish — magenta: lower/inner glass edge; cyan: upper/outer "
                "glass edge used by every downstream analysis. Dashed circles indicate an "
                "inferred pair when two supported radial peaks were unavailable."
            ),
            "perimeter_background_reference": (
                "Cyan outlines the detected upper/outer dish edge. The filled annulus "
                "shows the exact buffered band used for the initial median background "
                "colour: cyan when it is outside the dish, or orange for the inside-rim "
                "fallback when too little outer band is visible. The swatch shows the "
                "selected median starting colour and remains undimmed by overlay opacity."
            ),
            "seed_scale_estimation": "Cyan: isolated-reference search region. Yellow: the image-specific master seed diameter.",
            "foreground_feature": "Brightness is Lab colour distance from the estimated dish background before thresholding.",
            "foreground_mask": "Soft foreground probability: black is tray, rim, or dark inter-seed gap; white is seed-surface evidence. The binary proposal mask is available separately.",
            "foreground_colour_gamut": (
                "Full-size exact HSV hue/saturation slice of the fitted foreground "
                "Lab probability model. Adjust HSV value to scan brightness; contour "
                "lines show 25/50/75/90% membership and the neutral swatch reports "
                "achromatic membership without repeating undefined hue."
            ),
            "foreground_binary_mask": "Binary, morphologically cleaned foreground supplied to the distance-transform proposal branch.",
            "distance_transform": "Brightness is distance from the nearest foreground boundary; local maxima can become seed centres.",
            "distance_candidates": "Yellow circles are the centres/radii proposed by distance-transform peaks before fusion.",
            "circle_candidates": "Yellow circles are CUDA multiradius proposals supported by edge, sensor/noise, flattened-grayscale, shadow, and highlight boundaries.",
            "proposals": (
                "Cyan: upper/outer dish analysis rim. Yellow: provisional seed "
                "extent. Red: detected centre."
            ),
            "instance_masks": (
                "Each provisional seed has a unique RGB colour; nearby seeds "
                "are assigned colours chosen to contrast strongly."
            ),
            "background_likelihood": (
                "Colour-based likelihood: dark = high tray/dish background "
                "match; light = low match. The translucent cyan annulus is the "
                "outer-rim band sampled for the initial colour estimate; orange "
                "indicates the inside-rim fallback when too little outer band was visible."
            ),
            "other_colour_probability": (
                "Other-colour probability learned from painted Other references: black = "
                "no matching colour evidence; white = a strong match. Painted pixels are "
                "training evidence, not forced output values. This colour-only diagnostic "
                "is distinct from the multifeature Reference Other-material probability."
            ),
            "refined_background_likelihood": (
                "Noise-frequency likelihood: dark = background-like local "
                "texture; light = non-background-like texture. Fine, medium, "
                "and coarse profiles are learned per image from confident "
                "pixels in the colour layer, using equal class priors. The same "
                "learned texture classifier is also displayed across the initial "
                "dish-surrounding sampling annulus."
            ),
            "other_noise_probability": (
                "Other-texture probability learned from painted Other versus non-Other "
                "references: black = non-Other-like local frequency; white = Other-like "
                "evidence. Its base score is 72% learned three-band texture probability "
                "plus 28% Other-colour probability. Directional continuation then uses "
                "the noise node's configured ray integration. This class-specific "
                "diagnostic is distinct from the "
                "multifeature Reference Other-material probability."
            ),
            "background_colour_gamut": (
                "Full-size exact HSV hue/saturation slice of the fitted background "
                "Lab probability model. Adjust HSV value to scan brightness; contour "
                "lines show 25/50/75/90% membership and the neutral swatch reports "
                "achromatic membership without repeating undefined hue."
            ),
            "foreground_noise_likelihood": (
                "Foreground texture probability: black = non-foreground-like local "
                "frequency; white = foreground-like texture. Fine, medium, and coarse "
                "profiles are learned from foreground-colour pseudo-labels without "
                "forcing painted reference pixels to one."
            ),
            "reference_texture_prototypes": (
                "Full-pane collage of every retained image-local material and boundary "
                "feature medoid. Edge patches are rotated to a common tangent; percentages "
                "report cluster support, not forced probability."
            ),
            "reference_seed_surface_probability": (
                "Seed-surface likelihood from the nearest supported Foreground prototype "
                "relative to Background and Other prototype banks. Painted and unpainted "
                "pixels are evaluated identically."
            ),
            "reference_background_texture_probability": (
                "Background-texture likelihood from the many reviewed Background prototypes, "
                "attenuated only where Foreground or Other prototypes fit more specifically."
            ),
            "reference_other_texture_probability": (
                "Other-material likelihood from reviewed glass, rim, ruler, or other neither-"
                "seed-nor-background examples. It is classifier evidence, not a painted override."
            ),
            "edge_gradients": (
                "Brightness is the shared CUDA multichannel edge magnitude. "
                "The same cached continuous tangent field feeds both direction encoders."
            ),
            "surface_lightening_gradient": (
                "Hue is the one-sided direction from each query pixel toward the target "
                "with the greatest positive CIE L* slope; brightness is that maximum "
                "lightening slope after display normalization."
            ),
            "surface_lightening_magnitude": (
                "Brightness is the maximum positive target-minus-query CIE L* slope "
                "found along any sampled one-sided ray."
            ),
            "surface_darkening_gradient": (
                "Hue is the one-sided direction from each query pixel toward the target "
                "with the greatest negative CIE L* slope; brightness is the corresponding "
                "positive darkening magnitude."
            ),
            "surface_darkening_magnitude": (
                "Brightness is the maximum positive query-minus-target CIE L* slope "
                "found along any sampled one-sided ray."
            ),
            "weak_lightening_gradient": (
                "Lightening direction and brightness after responses above the editable "
                "raw L*/pixel ceiling have been set exactly to zero to suppress edges."
            ),
            "weak_lightening_magnitude": (
                "Filtered lightening magnitude; black includes every strong response "
                "removed by the upper L*/pixel ceiling."
            ),
            "weak_darkening_gradient": (
                "Darkening direction and brightness after responses above the editable "
                "raw L*/pixel ceiling have been set exactly to zero to suppress edges."
            ),
            "weak_darkening_magnitude": (
                "Filtered darkening magnitude; black includes every strong response "
                "removed by the upper L*/pixel ceiling."
            ),
            "darkness_noise_fine": "Fine-band surrounding RMS energy of CIE L* variation.",
            "darkness_noise_medium": "Medium-band surrounding RMS energy of CIE L* variation.",
            "darkness_noise_coarse": "Coarse-band surrounding RMS energy of CIE L* variation.",
            "colour_noise_fine": "Fine-band surrounding RMS energy of Lab a*/b* colour variation.",
            "colour_noise_medium": "Medium-band surrounding RMS energy of Lab a*/b* colour variation.",
            "colour_noise_coarse": "Coarse-band surrounding RMS energy of Lab a*/b* colour variation.",
            "undirected_edges": (
                "Hue = undirected edge tangent (0° = 180°); brightness = "
                "maximum multichannel edge likelihood. Opposite tangent "
                "directions share the same colour."
            ),
            "directed_edges": (
                "Hue = directed edge tangent (0° = 360°), oriented with the "
                "brighter side on its right; brightness = maximum multichannel "
                "edge likelihood. Direction is less stable when both sides have "
                "nearly equal lightness."
            ),
            "edge_ridges": (
                "Non-maximum-suppressed one-pixel ridges after high/low GPU "
                "hysteresis. Brightness is retained continuous edge strength."
            ),
            "physical_edge_probability": (
                "Yellow brightness is the probability that a transition is a true "
                "physical seed boundary, learned from annotated seed instances."
            ),
            "non_edge_probability": (
                "Blue brightness is the probability that a transition is an apparent "
                "coat-pattern or lighting boundary rather than a physical edge."
            ),
            "reference_edge_comparison": (
                "Physical-edge evidence is blue and non-physical edge evidence is "
                "red. Both channels are inferred from annotated seed instances; "
                "Magenta marks overlap where both interpretations receive support."
            ),
            "net_physical_edge_probability": (
                "Positive physical-edge evidence margin: max(physical - "
                f"{float(self.pipeline.node('reference_edge_probability').parameters['net_physical_edge_internal_scale']):g} "
                "× internal edge, 0). Ties and negative results are clamped to a "
                "black floor. This diagnostic difference is not a calibrated posterior "
                "probability and does not modify either source raster."
            ),
            "reference_edge_ridges": (
                "Reference-trained physical-edge probability after normal-direction "
                "non-maximum suppression and CUDA high/low hysteresis. Yellow "
                "brightness preserves the retained continuous probability."
            ),
            "edge_traces": (
                "Unique colours identify tangent-compatible connected traces. "
                "Crossing junctions are split and short aligned gaps may be bridged."
            ),
            "edge_trace_continuity": (
                "Tangent-following support on both sides of every ridge pixel. "
                "Bright traces remain coherent across a seed-relative window."
            ),
            "edge_trace_gap_confidence": (
                "Geometric path support that penalizes repeated missing samples "
                "without requiring a completely unbroken visible boundary."
            ),
            "edge_radius_confirmation": (
                "Hue is fitted radius ratio (half to twice nominal across the hue "
                "wheel); brightness combines arc, circle, ellipse and centre-vote support."
            ),
            "edge_circle_fit": "Circle-arc and proposal-conditioned radial fit confidence.",
            "edge_ellipse_fit": "Batched proposal-conditioned ellipse/conic fit confidence, including elongated lupin boundaries.",
            "edge_fit_residual": "Bright pixels have high residual under both fitted circle and ellipse models.",
            "edge_centre_votes": "Normal-and-radius votes accumulated from fragmented oriented traces; bright maxima are plausible seed centres.",
            "edge_semantic_sides": "Soft agreement that the inferred inside is foreground/non-background. It boosts but never gates an edge.",
            "edge_rejections": (
                "Hue categorizes rejection: weak hysteresis, junction/short trace, "
                "poor continuity, implausible shape/radius, or low combined confidence."
            ),
            "edge_fit_geometry": "Vector centres from trace-normal voting and fitted per-proposal ellipses; compact geometry transfers only when selected.",
            "seed_edge_curves": (
                "Final seed-boundary confidence after ridge thinning, oriented trace "
                "continuity, dense arc radii, centre voting, circle/ellipse residuals, "
                "and soft semantic/lightness confirmation."
            ),
            "procedural_seed_material": "Combined foreground colour/noise and inverse-background likelihood used to gate procedural instances.",
            "procedural_seed_mask": "Thresholded seed material after dish-margin removal and seed-sized enclosed coat-hole filling.",
            "procedural_boundary_cost": "Normalized physical boundary cost fused from edges, sensor/noise, ridges and local shadow.",
            "procedural_centres": "Marker likelihood from smoothed material, physical-boundary depth and annular boundary support; dots are retained markers.",
            "procedural_instances": "Marker-controlled watershed instance identities. Dot colour runs red to green with per-instance confidence.",
            "procedural_confidence": "Per-instance marker, boundary and calibrated-area confidence mapped back onto every assigned pixel.",
            "unet_interior": "Learned probability that each pixel belongs to visible seed material.",
            "unet_physical_boundary": "Learned probability of a true physical seed/background or seed/seed boundary.",
            "unet_pattern_boundary": "Learned probability of an apparent but non-physical internal coat-pattern boundary.",
            "unet_centres": "Learned centre likelihood after support from normalized interior depth.",
            "unet_distance": "Learned normalized distance from seed interior pixels to their physical boundary.",
            "unet_uncertainty": "Aleatoric uncertainty aggregated across the U-Net dense tasks.",
            "unet_instances": "Pattern-aware marker-controlled watershed identities decoded from the five-head U-Net.",
            "unet_confidence": "Per-instance U-Net confidence mapped onto decoded pixels; it is not a validation guarantee.",
            "stardist_object_probability": "Learned probability that a pixel is a suitable centre for a star-convex seed polygon.",
            "stardist_radial_uncertainty": "Learned uncertainty of the StarDist radial boundary regression.",
            "stardist_instances": "Star-convex seed polygons retained after score ordering and overlap-aware non-maximum suppression.",
            "stardist_confidence": "Per-polygon StarDist score mapped onto decoded pixels; it is not a validation guarantee.",
            "seed_interior_probability": "Soft seed-interior evidence from foreground colour, learned foreground noise, and inverse background support.",
            "boundary_confidence": "Hue is the continuous directed boundary normal (0° = 360°); brightness is boundary confidence.",
            "boundary_magnitude": "Boundary confidence without direction encoding.",
            "touching_split_likelihood": "High values mark shallow foreground necks and concave distance-transform saddles that may separate touching seeds.",
            "ellipse_likelihood": "Hue is axial ellipse orientation (0° = 180°); brightness is seed-radius boundary and structure-tensor support.",
            "proposal_disagreement": "Variation among CUDA ring, distance-peak, interior, and ellipse evidence; bright regions merit review.",
            "instance_assignment_confidence": "Confidence that a provisional instance owns each pixel after boundary and label-contact penalties.",
            "contested_pixels": "Pixels bordering two different provisional instance labels.",
            "contact_graph": "Shared-boundary likelihood plus vector links between proposals close enough to touch or overlap.",
            "illumination_field": "Estimated broad illumination component; shown as grayscale intensity.",
            "flattened_grayscale": "Grayscale after a local illumination log-ratio flattens broad lighting variation; middle gray means locally expected brightness.",
            "shadow_likelihood": "Nonlinear likelihood that a pixel is unusually dark relative to local lighting and local absolute deviation.",
            "highlight_likelihood": "Nonlinear likelihood that a pixel is unusually bright relative to local lighting and local absolute deviation.",
            "reflectance_image": "Approximate illumination-normalized luminance used by coat analyses.",
            "glare_likelihood": "Bright, low-saturation pixels likely affected by specular reflection.",
            "image_quality_risk": "Composite of low local focus, highlight clipping, underexposure, and high-frequency noise.",
            "focus_quality": "Local gradient-derived focus evidence; brighter is sharper.",
            "clipped_highlights": "Near-saturated highlight risk.",
            "underexposure": "Very low-luminance exposure risk.",
            "sensor_noise": "Local high-frequency residual used as a sensor/noise diagnostic.",
            "radial_profile_residual": "Difference from the expected centre-to-edge seed lightness profile.",
            "radial_coordinate": "Assigned seed distance: dark at proposal centre and brighter toward/beyond its radius.",
            "wrinkling_likelihood": "Seed-scaled internal ridge/valley response, excluding the main outer boundary.",
            "coat_damage_likelihood": "Local colour anomaly reinforced by internal edge or radial-profile evidence.",
            "pattern_classes": "Hue is the winning broad coat-pattern class; brightness is classification confidence.",
            "pattern_confidence": "Confidence of the winning broad pattern class.",
            "colour_classes": "Hue is the winning broad colour class; brightness is class confidence.",
            "colour_uncertainty": "Entropy of the broad colour-class probabilities; bright pixels are ambiguous.",
            "calibration_residual_risk": "Reference confidence plus spatial extrapolation risk away from the detected card and ruler.",
            "none": "No analysis layer is shown.",
        }
        if mode.startswith("colour_probability:"):
            index = int(mode.partition(":")[2])
            name = self.image_view._analysis_result.advanced.colour_class_names[index]
            text = f"Per-pixel broad {name} probability after colour balance; this is a transparent prototype model, not a trained classifier."
        elif mode.startswith("pattern_probability:"):
            index = int(mode.partition(":")[2])
            name = self.image_view._analysis_result.advanced.pattern_class_names[index]
            text = f"Per-pixel broad {name} pattern probability from seed-relative spatial-frequency evidence."
        else:
            text = legends.get(mode, "")
        self.overlay_legend_label.setText(text)

    def _mark_analysis_complete(self, result) -> None:
        node_timings = getattr(result, "node_timings_seconds", {})
        for node_id, elapsed in node_timings.items():
            if node_id in self.pipeline.nodes:
                self.pipeline.node(node_id).calculation_seconds = float(elapsed)
        calibration = result.calibration
        card = calibration.colour_card
        self.pipeline.set_status(
            "colour_reference",
            NodeStatus.COMPLETE if card is not None else NodeStatus.WARNING,
            (
                f"{card.detected_swatch_count}/24 swatches; "
                f"confidence {card.confidence:.0%}"
                if card is not None
                else "Swatch grid not found"
            ),
        )
        ruler = calibration.ruler
        self.pipeline.set_status(
            "ruler_detection",
            NodeStatus.COMPLETE if ruler is not None else NodeStatus.WARNING,
            (
                f"Angle {ruler.angle_degrees:+.2f}°; "
                f"confidence {ruler.confidence:.0%}"
                if ruler is not None
                else "Ruler not found"
            ),
        )
        self.pipeline.set_status(
            "deskew_colour",
            NodeStatus.COMPLETE,
            f"Rotated {calibration.deskew_degrees:+.2f}° and balanced colour",
        )
        if calibration.perspective_corrected:
            self.pipeline.set_status(
                "deskew_colour",
                NodeStatus.COMPLETE,
                f"Projective tilt {calibration.perspective_strength:.3f}; "
                f"rotation {calibration.deskew_degrees:+.2f}°; balanced colour",
            )
        self.pipeline.set_status(
            "scale_calibration",
            (
                NodeStatus.COMPLETE
                if calibration.pixels_per_mm is not None
                else NodeStatus.WARNING
            ),
            (
                f"{calibration.pixels_per_mm:.3f} px/mm"
                if calibration.pixels_per_mm is not None
                else "Tick spacing unresolved"
            ),
        )
        self.pipeline.set_status(
            "layout_detection",
            (
                NodeStatus.COMPLETE
                if result.dish.rim_pair_detected
                else NodeStatus.WARNING
            ),
            (
                f"Centre ({result.dish.center_x}, {result.dish.center_y}); "
                f"lower {result.dish.inner_radius}px; upper "
                f"{result.dish.outer_radius}px; confidence "
                f"{result.dish.confidence:.0%}"
                + (
                    ""
                    if result.dish.rim_pair_detected
                    else "; second edge inferred"
                )
            ),
        )
        self.pipeline.set_status(
            "seed_scale_estimation",
            NodeStatus.COMPLETE if result.reference_seed_count else NodeStatus.WARNING,
            (
                f"{result.reference_seed_count} reference component(s); "
                f"diameter {result.estimated_seed_diameter_px:.1f}px"
                if result.reference_seed_count
                else f"Dish fallback diameter {result.estimated_seed_diameter_px:.1f}px"
            ),
        )
        sampling_band = result.perimeter_background_band
        self.pipeline.set_status(
            "perimeter_background_reference",
            NodeStatus.COMPLETE if sampling_band.outside_vessel else NodeStatus.WARNING,
            (
                f"{sampling_band.buffer_cm:.2f} cm buffer; "
                f"{sampling_band.thickness_cm:.2f} cm band; "
                f"{sampling_band.sample_count:,} pixels"
                + ("" if sampling_band.outside_vessel else "; inside-rim fallback")
            ),
        )
        foreground_fraction = result.foreground_pixel_count / max(
            1, result.analysis_region_pixel_count
        )
        self.pipeline.set_status(
            "foreground_segmentation",
            NodeStatus.COMPLETE,
            (
                f"Threshold {result.foreground_threshold:.1f}; "
                f"{result.foreground_pixel_count:,} pixels ({foreground_fraction:.1%}); "
                f"{getattr(result, 'foreground_reference_count', 0):,} painted reference pixels"
            ),
        )
        self.pipeline.set_status(
            "distance_candidates",
            NodeStatus.COMPLETE,
            f"{result.distance_candidate_count:,} distance peak(s)",
        )
        self.pipeline.set_status(
            "circle_candidates",
            NodeStatus.COMPLETE,
            f"{result.circle_candidate_count:,} CUDA edge/noise-supported ring(s)",
        )
        self.pipeline.set_status(
            "identification",
            NodeStatus.COMPLETE,
            f"{result.count:,} approximate proposals",
        )
        procedural = getattr(result, "procedural_instances", None)
        if procedural is not None:
            median_confidence = (
                float(np.median(procedural.instance_confidences))
                if procedural.count
                else 0.0
            )
            low_fraction = (
                float(np.mean(procedural.instance_confidences < 0.55))
                if procedural.count
                else 1.0
            )
            self.pipeline.set_status(
                "procedural_instances",
                NodeStatus.WARNING if low_fraction >= 0.20 else NodeStatus.COMPLETE,
                f"{procedural.count:,} instances; median confidence "
                f"{median_confidence:.0%}; {low_fraction:.0%} require review",
            )
        for node_id, learned in (
            ("unet_instances", getattr(result, "unet_instances", None)),
            ("stardist_instances", getattr(result, "stardist_instances", None)),
        ):
            if learned is None:
                continue
            median_confidence = (
                float(np.median(learned.instance_confidences))
                if learned.count
                else 0.0
            )
            low_fraction = (
                float(np.mean(learned.instance_confidences < 0.55))
                if learned.count
                else 1.0
            )
            self.pipeline.set_status(
                node_id,
                NodeStatus.WARNING,
                f"{learned.count:,} instances; median confidence "
                f"{median_confidence:.0%}; {low_fraction:.0%} require review; "
                "not publication-validated",
            )
        background_mode = result.layers.background_mode
        colour_profile = result.layers.background_colour_profile
        background_deviation = float(
            getattr(result, "background_prior_deviation", 0.0)
        )
        prior_warning = background_deviation >= 18.0
        prior_suffix = (
            f"; perimeter deviation ΔLab {background_deviation:.1f}"
            if background_mode != "disabled"
            else ""
        )
        sampling_band = getattr(result, "perimeter_background_band", None)
        if sampling_band is not None and background_mode != "disabled":
            prior_suffix += (
                f"; initial {'outer' if sampling_band.outside_vessel else 'inner fallback'} "
                f"band {sampling_band.sample_count:,} px; "
                f"{sampling_band.buffer_cm:.2f} cm buffer / "
                f"{sampling_band.thickness_cm:.2f} cm thickness"
            )
        if background_mode == "disabled":
            self.pipeline.set_status(
                "background_likelihood", NodeStatus.BYPASSED, "Disabled for this run"
            )
        elif background_mode == "manual":
            self.pipeline.set_status(
                "background_likelihood",
                NodeStatus.WARNING if prior_warning else NodeStatus.COMPLETE,
                _background_profile_detail(
                    colour_profile,
                    f"{result.layers.background_reference_count:,} painted reference pixels",
                ) + prior_suffix,
            )
        else:
            self.pipeline.set_status(
                "background_likelihood",
                NodeStatus.WARNING if prior_warning else NodeStatus.COMPLETE,
                _background_profile_detail(colour_profile, "Automatic range")
                + prior_suffix,
            )
        self.pipeline.set_status(
            "instance_masks",
            NodeStatus.COMPLETE,
            f"{result.count:,} uniquely coloured masks",
        )
        profile = result.layers.noise_frequency_profile
        if background_mode == "disabled":
            self.pipeline.set_status(
                "refined_background_likelihood",
                NodeStatus.BLOCKED,
                "Background colour analysis disabled",
            )
        else:
            self.pipeline.set_status(
                "refined_background_likelihood",
                NodeStatus.COMPLETE,
                f"{len(result.layers.directional_background_angles_degrees)} directions integrated; "
                f"three-band separation {profile.separation:.2f}; surrounding annulus shown",
            )
        foreground_noise_profile = result.layers.foreground_noise_frequency_profile
        if foreground_noise_profile is not None:
            self.pipeline.set_status(
                "foreground_noise_likelihood",
                NodeStatus.COMPLETE,
                "Foreground/non-foreground three-band separation "
                f"{foreground_noise_profile.separation:.2f}",
            )
        self.pipeline.set_status(
            "edge_gradients",
            NodeStatus.COMPLETE,
            "Shared Lab/Scharr magnitude and continuous tangent field cached",
        )
        if (
            self.pipeline.is_active("surface_darkness_gradients")
            and self.pipeline.node("surface_darkness_gradients").enabled
        ):
            direction_count = len(
                range(
                    0,
                    360,
                    int(
                        self.pipeline.node("surface_darkness_gradients").parameters[
                            "surface_gradient_direction_step_degrees"
                        ]
                    ),
                )
            )
            self.pipeline.set_status(
                "surface_darkness_gradients",
                NodeStatus.COMPLETE,
                "Maximum lightening/darkening slopes from "
                f"{direction_count} one-sided directions",
            )
        if (
            self.pipeline.is_active("lightening_gradient_ceiling")
            and self.pipeline.node("lightening_gradient_ceiling").enabled
        ):
            self.pipeline.set_status(
                "lightening_gradient_ceiling",
                NodeStatus.COMPLETE,
                "Strong lightening edges suppressed above "
                f"{self.pipeline.node('lightening_gradient_ceiling').parameters['lightening_gradient_maximum_slope']:.2f} L*/px",
            )
        if (
            self.pipeline.is_active("darkening_gradient_ceiling")
            and self.pipeline.node("darkening_gradient_ceiling").enabled
        ):
            self.pipeline.set_status(
                "darkening_gradient_ceiling",
                NodeStatus.COMPLETE,
                "Strong darkening edges suppressed above "
                f"{self.pipeline.node('darkening_gradient_ceiling').parameters['darkening_gradient_maximum_slope']:.2f} L*/px",
            )
        noise_scales = result.layers.frequency_noise_band_scales_px
        self.pipeline.set_status(
            "frequency_noise_masks",
            NodeStatus.COMPLETE,
            "Darkness/colour RMS masks at "
            + ", ".join(f"{value:.1f}px" for value in noise_scales),
        )
        self.pipeline.set_status(
            "undirected_edges",
            NodeStatus.COMPLETE,
            "Axial hue map ready (0° = 180°)",
        )
        self.pipeline.set_status(
            "directed_edges",
            NodeStatus.COMPLETE,
            "Polarity-aware hue map ready (0° = 360°)",
        )
        curve_raster = result.layers.seed_edge_curve_likelihood
        supported_curve_pixels = (
            curve_raster.count_above(128)
            if hasattr(curve_raster, "count_above")
            else int((curve_raster >= 128).sum())
        )
        trace_raster = result.layers.edge_trace_labels
        trace_pixels = (
            trace_raster.count_above(1)
            if hasattr(trace_raster, "count_above")
            else int((trace_raster >= 1).sum())
        )
        self.pipeline.set_status(
            "edge_ridges", NodeStatus.COMPLETE, "Float NMS and hysteresis cached on GPU"
        )
        texture_profile = getattr(
            result.layers, "reference_texture_profile", None
        )
        texture_counts = (
            {}
            if texture_profile is None
            else dict(texture_profile.class_sample_counts)
        )
        material_reference_count = sum(
            int(texture_counts.get(name, 0))
            for name in ("background", "foreground", "other")
        )
        edge_reference_count = sum(
            int(texture_counts.get(name, 0))
            for name in ("physical_edge", "non_edge")
        )
        prototype_parts = []
        if material_reference_count:
            prototype_parts.append(f"{material_reference_count:,} material samples")
        if edge_reference_count:
            prototype_parts.append(
                f"{edge_reference_count:,} instance-derived edge samples"
            )
        self.pipeline.set_status(
            "reference_texture_prototypes",
            NodeStatus.COMPLETE,
            (
                f"{len(texture_profile.prototypes):,} prototypes from "
                + " and ".join(prototype_parts)
                if texture_profile is not None
                and texture_profile.prototypes
                and prototype_parts
                else "No applied material or complete instance references"
            ),
        )
        self.pipeline.set_status(
            "reference_edge_probability",
            NodeStatus.COMPLETE if edge_reference_count else NodeStatus.WARNING,
            (
                "Physical/non-physical supervision derived from annotated seed instances"
                if edge_reference_count
                else "No complete instance annotations; semantic edge evidence is neutral"
            ),
        )
        reference_ridge_raster = result.layers.reference_edge_ridges
        reference_ridge_pixels = (
            reference_ridge_raster.count_above(1)
            if hasattr(reference_ridge_raster, "count_above")
            else int((reference_ridge_raster >= 1).sum())
        )
        self.pipeline.set_status(
            "reference_edge_ridges",
            NodeStatus.COMPLETE,
            f"{reference_ridge_pixels:,} thinned reference-edge pixels",
        )
        self.pipeline.set_status(
            "edge_traces", NodeStatus.COMPLETE, f"{trace_pixels:,} oriented trace pixels"
        )
        self.pipeline.set_status(
            "seed_edge_curves",
            NodeStatus.COMPLETE,
            f"{supported_curve_pixels:,} strongly supported boundary pixels",
        )
        backend = result.advanced.backend
        colour_distribution = _format_class_distribution(
            result.advanced.colour_class_names,
            result.advanced.seed_colour_proportions,
        )
        pattern_distribution = _format_class_distribution(
            result.advanced.pattern_class_names,
            result.advanced.seed_pattern_proportions,
        )
        advanced_details = {
            "seed_interior": "Foreground colour/noise and background probability fused",
            "boundary_normals": "Boundary magnitude and directed normals ready",
            "touching_split": "Neck and distance-saddle evidence ready",
            "ellipse_likelihood": "Multiscale orientation and radial support ready",
            "proposal_disagreement": "Four proposal evidence sources compared",
            "assignment_confidence": "Assignment and contested-pixel maps ready",
            "contact_graph": f"{len(result.advanced.contact_pairs):,} candidate contact link(s)",
            "illumination_decomposition": "Flattened grayscale plus local shadow/highlight maps ready",
            "image_quality": "Focus, clipping, exposure, and noise maps ready",
            "radial_profile": "Normalized radial coordinates and residuals ready",
            "wrinkling": f"Mean provisional likelihood {result.advanced.mean_wrinkling_likelihood:.0%}",
            "coat_damage": f"Mean provisional likelihood {result.advanced.mean_coat_damage_likelihood:.0%}",
            "pattern_decomposition": pattern_distribution,
            "colour_probabilities": colour_distribution,
            "calibration_residuals": "Reference confidence/extrapolation risk ready",
        }
        for node_id, detail in advanced_details.items():
            self.pipeline.set_status(
                node_id,
                NodeStatus.COMPLETE,
                f"{detail}; {backend.used.upper()}",
            )
        self.pipeline.set_status(
            "review", NodeStatus.WARNING, "Awaiting mask-review tools"
        )
        self.pipeline.set_status(
            "measurements", NodeStatus.BLOCKED, "Requires reviewed masks"
        )
        self.pipeline.set_status(
            "classification",
            NodeStatus.WARNING,
            f"Provisional seeds — colour: {colour_distribution}; pattern: {pattern_distribution}",
        )
        for node in self.pipeline.nodes.values():
            if not node.enabled:
                self.pipeline.set_status(
                    node.identifier,
                    NodeStatus.BYPASSED,
                    "Disabled by pipeline dependency",
                )
        self.pipeline_canvas.refresh()
        self.pipeline_inspector.refresh_status()

    def _pipeline_image_loaded(self, path: Path) -> None:
        for node in self.pipeline.nodes.values():
            node.calculation_seconds = None
        self.pipeline.set_status("raw_images", NodeStatus.COMPLETE, path.name)
        self.pipeline.set_status(
            "metadata", NodeStatus.COMPLETE, self.species_combo.currentText()
        )
        key = str(path.resolve()).casefold()
        reference_counts = (
            self._mask_pixel_count(
                self._applied_background_reference_masks.get(key)
            ),
            self._mask_pixel_count(
                self._applied_foreground_reference_masks.get(key)
            ),
            self._mask_pixel_count(
                self._applied_background_exclusion_masks.get(key)
            ),
            self._mask_pixel_count(
                self._applied_foreground_exclusion_masks.get(key)
            ),
            len(self._instance_ids(self._applied_instance_annotations.get(key))),
        )
        self.pipeline.set_status(
            "reference_layers",
            NodeStatus.COMPLETE,
            (
                f"BG {reference_counts[0]:,}; FG {reference_counts[1]:,}; "
                f"Other {max(reference_counts[2], reference_counts[3]):,}; "
                f"seeds {reference_counts[4]:,}"
                if any(reference_counts)
                else "No applied references"
            ),
        )
        for node_id in (*CALIBRATION_NODE_IDS, "layout_detection"):
            self.pipeline.set_status(node_id, NodeStatus.IDLE, "Ready")
        for node_id in IDENTIFICATION_STAGE_NODE_IDS:
            self.pipeline.set_status(node_id, NodeStatus.IDLE, "Ready")
        for node_id in (
            "background_likelihood",
            "refined_background_likelihood",
            "foreground_noise_likelihood",
            "edge_gradients",
            "surface_darkness_gradients",
            "lightening_gradient_ceiling",
            "darkening_gradient_ceiling",
            "frequency_noise_masks",
            "undirected_edges",
            "directed_edges",
            "edge_ridges",
            "reference_texture_prototypes",
            "reference_edge_probability",
            "reference_edge_ridges",
            "edge_traces",
            "procedural_instances",
            "unet_instances",
            "stardist_instances",
        ):
            node = self.pipeline.node(node_id)
            if self.pipeline.is_active(node_id) and node.enabled:
                self.pipeline.set_status(node_id, NodeStatus.IDLE, "Ready")
            elif not node.enabled:
                self.pipeline.set_status(
                    node_id,
                    NodeStatus.BYPASSED,
                    (
                        "Disabled in the unused-node toolbox"
                        if not self.pipeline.is_active(node_id)
                        else "Disabled by pipeline dependency"
                    ),
                )
        self.pipeline.set_status("seed_edge_curves", NodeStatus.IDLE, "Ready")
        for node_id in ADVANCED_NODE_MODES:
            self.pipeline.set_status(node_id, NodeStatus.IDLE, "Ready")
        self.pipeline.set_status(
            "instance_masks",
            NodeStatus.BLOCKED,
            "Stored in the unused-node toolbox",
        )
        background = self.pipeline.node("background_likelihood")
        if not background.enabled:
            self.pipeline.set_status(
                "background_likelihood", NodeStatus.BYPASSED, "Bypassed"
            )
            self.pipeline.set_status(
                "refined_background_likelihood",
                NodeStatus.BLOCKED,
                "Background colour analysis disabled",
            )
        self.pipeline.set_status("review", NodeStatus.PLANNED, "Planned")
        self.pipeline.set_status(
            "measurements", NodeStatus.BLOCKED, "Requires reviewed masks"
        )
        self.pipeline.set_status("classification", NodeStatus.PLANNED, "Planned")
        self.pipeline.set_status("aggregation", NodeStatus.PLANNED, "Planned")
        self.pipeline.set_status("output", NodeStatus.PLANNED, "Planned")
        for node in self.pipeline.nodes.values():
            if not node.enabled:
                self.pipeline.set_status(
                    node.identifier,
                    NodeStatus.BYPASSED,
                    "Disabled by pipeline dependency",
                )
        self.pipeline_canvas.refresh()
        self.pipeline_inspector.refresh_status()

    def _set_analysis_running(
        self, path: Path, dirty_nodes: frozenset[str] = frozenset()
    ) -> None:
        details = {
            "colour_reference": "Detecting the 4×6 swatch grid",
            "ruler_detection": "Locating 0 and terminal scale dashes",
            "deskew_colour": "Deskewing and balancing colour",
            "scale_calibration": "Resolving ruler tick spacing",
            "layout_detection": "Detecting both corrected Petri-dish glass edges",
            "seed_scale_estimation": "Measuring isolated reference components",
            "perimeter_background_reference": "Sampling the buffered outside-dish colour band",
            "foreground_segmentation": "Estimating Lab background and thresholding",
            "distance_candidates": "Finding distance-transform peaks",
            "circle_candidates": "Fusing edge and local-lighting ring evidence",
            "identification": "Fusing candidate centres",
            "instance_masks": "Separating provisional instances",
            "edge_gradients": "Computing shared Lab/Scharr gradient field",
            "surface_darkness_gradients": "Searching one-sided lightening and darkening surface slopes",
            "lightening_gradient_ceiling": "Suppressing strong lightening edges",
            "darkening_gradient_ceiling": "Suppressing strong darkening edges",
            "frequency_noise_masks": "Calculating multiscale darkness and colour RMS energy",
            "undirected_edges": "Encoding axial edge tangents",
            "directed_edges": "Resolving edge polarity",
            "edge_ridges": "Thinning float edges and reconstructing hysteresis",
            "reference_texture_prototypes": "Fitting many material and edge prototypes from reviewed examples",
            "reference_edge_probability": "Classifying physical and apparent edges from reviewed examples",
            "reference_edge_ridges": "Thinning reference-trained physical-edge probability",
            "edge_traces": "Linking orientation-compatible ridge fragments",
            "seed_edge_curves": "Confirming radii, circles, ellipses, centres, and semantic sides",
            "background_likelihood": "Estimating the likely colour range",
            "refined_background_likelihood": "Evaluating directed texture rays",
            "foreground_noise_likelihood": "Evaluating foreground texture rays",
            "seed_interior": "Fusing seed-interior evidence on the tensor device",
            "boundary_normals": "Estimating boundary confidence and normals",
            "touching_split": "Scoring necks and distance saddles",
            "ellipse_likelihood": "Testing multiscale elliptical support",
            "proposal_disagreement": "Comparing independent proposal evidence",
            "assignment_confidence": "Scoring provisional pixel ownership",
            "contact_graph": "Resolving touching and overlapping proposals",
            "illumination_decomposition": "Flattening grayscale and classifying local lighting extremes",
            "image_quality": "Mapping focus, exposure, glare, and noise",
            "radial_profile": "Accumulating seed radial profiles",
            "wrinkling": "Detecting seed-scaled ridge texture",
            "coat_damage": "Fusing coat-anomaly evidence",
            "pattern_decomposition": "Estimating broad coat-pattern probabilities",
            "colour_probabilities": "Estimating broad colour probabilities",
            "calibration_residuals": "Mapping calibration extrapolation risk",
        }
        targets = set(dirty_nodes) if dirty_nodes else set(details)
        if not dirty_nodes:
            self.pipeline.set_status("raw_images", NodeStatus.COMPLETE, path.name)
            self.pipeline.set_status(
                "metadata", NodeStatus.COMPLETE, self.species_combo.currentText()
            )
        for node_id in targets & details.keys():
            if not self.pipeline.node(node_id).enabled:
                self.pipeline.set_status(
                    node_id, NodeStatus.BYPASSED, "Disabled by pipeline dependency"
                )
            elif (
                node_id == "refined_background_likelihood"
                and not self.pipeline.node("background_likelihood").enabled
            ):
                self.pipeline.set_status(
                    node_id, NodeStatus.BLOCKED, "Background colour disabled"
                )
            else:
                self.pipeline.set_status(node_id, NodeStatus.RUNNING, details[node_id])
        self.pipeline_canvas.refresh(targets)
        self.pipeline_inspector.refresh_status()
        self._sync_background_controls()

    @Slot(str)
    def _pipeline_node_selected(self, node_id: str) -> None:
        self._selected_pipeline_node = node_id
        self.pipeline_inspector.set_node(self.pipeline.node(node_id))
        # Selected-node controls live at the top of the panel, so reset from
        # whichever summary section the user had previously been viewing.
        self.metadata_scroll.verticalScrollBar().setValue(0)
        self.calibration_section.setVisible(
            node_id in {"deskew_colour", "output"}
        )
        self.baseline_section.setVisible(node_id in {"identification", "output"})
        mode = VIEWER_NODE_MODES.get(node_id)
        if mode is not None and not self._selecting_node_from_overlay:
            index = self.overlay_combo.findData(mode)
            if index >= 0:
                self._selecting_overlay_from_node = True
                try:
                    self.overlay_combo.setCurrentIndex(index)
                finally:
                    self._selecting_overlay_from_node = False
        self._sync_inspector_overlay_options()

    @Slot(str)
    def _pipeline_unused_node_restored(self, node_id: str) -> None:
        """Expose a toolbox node's viewer layer after explicit restoration."""

        mode = VIEWER_NODE_MODES.get(node_id)
        self._rebuild_overlay_combo(mode)
        overlay_index = self.overlay_combo.findData(mode or "")
        if overlay_index >= 0:
            self.overlay_combo.setCurrentIndex(overlay_index)
        self.pipeline_inspector.set_node(self.pipeline.node(node_id))
        self._sync_inspector_overlay_options()
        self.statusBar().showMessage(
            f"Restored {self.pipeline.node(node_id).title} to the graph; "
            "it remains disabled until explicitly enabled."
        )

    @Slot(str)
    def _pipeline_unused_node_shelved(self, node_id: str) -> None:
        """Remove an optional node's viewer product and stale cached result."""

        self._rebuild_overlay_combo("raw_image")
        for image_key in set(self._analysis_caches) | set(self._analyses):
            self._cache_dirty_nodes.setdefault(image_key, set()).add(node_id)
            self._analyses.pop(image_key, None)
        fallback = "illumination_decomposition"
        if self.pipeline.is_active(fallback):
            self.pipeline_canvas.select_node(fallback)
            self.pipeline_inspector.set_node(self.pipeline.node(fallback))
        current_key = self._current_image_key()
        if current_key is not None and current_key in self._analysis_caches:
            self._analyze_current_image(dirty_nodes={node_id})
        self.statusBar().showMessage(
            f"Moved {self.pipeline.node(node_id).title} to Unused nodes."
        )

    @Slot(str, str, object)
    def _pipeline_parameter_changed(self, node_id: str, key: str, value) -> None:
        parameter_spec = next(
            (
                spec
                for spec in self.pipeline.node(node_id).parameter_specs
                if spec.key == key
            ),
            None,
        )
        try:
            self._validate_settings_override(node_id, key, value)
            affected = self.pipeline.set_parameter(node_id, key, value)
        except (KeyError, TypeError, ValueError) as error:
            QMessageBox.warning(self, "Invalid pipeline parameter", str(error))
            self.pipeline_inspector.set_node(self.pipeline.node(node_id))
            return
        if not affected:
            return
        if parameter_spec is not None and parameter_spec.display_only:
            self._refresh_display_only_analysis_parameter(node_id, key)
            return
        if "measurements" in affected:
            self.pipeline.set_status(
                "measurements", NodeStatus.BLOCKED, "Requires reviewed masks"
            )
        affected_set = set(affected)
        known_keys = set(self._analysis_caches) | set(self._analyses)
        for image_key in known_keys:
            self._cache_dirty_nodes.setdefault(image_key, set()).update(
                affected_set
            )
            self._analyses.pop(image_key, None)
        self.pipeline_canvas.refresh(affected)
        self.pipeline_inspector.refresh_status()
        current_key = self._current_image_key()
        if current_key is not None and current_key in self._analysis_caches:
            self._analyze_current_image(dirty_nodes=affected_set)
        self.statusBar().showMessage(
            f"Changed {self.pipeline.node(node_id).title}; recomputing graph dependents."
        )

    @Slot(str)
    def _pipeline_parameters_reset(self, node_id: str) -> None:
        node = self.pipeline.node(node_id)
        parameter_specs = {spec.key: spec for spec in node.parameter_specs}
        changed_keys = {
            key
            for key, default in node.default_parameters.items()
            if node.parameters.get(key) != default
        }
        display_only_reset = bool(changed_keys) and all(
            parameter_specs.get(key) is not None
            and parameter_specs[key].display_only
            for key in changed_keys
        )
        affected = self.pipeline.reset_parameters(node_id)
        self.pipeline_inspector.set_node(self.pipeline.node(node_id))
        self._sync_background_controls()
        if not affected:
            self.statusBar().showMessage(
                f"{self.pipeline.node(node_id).title} already uses its defaults."
            )
            return
        if display_only_reset:
            for key in changed_keys:
                self._refresh_display_only_analysis_parameter(node_id, key)
            return
        affected_set = set(affected)
        if "measurements" in affected_set:
            self.pipeline.set_status(
                "measurements", NodeStatus.BLOCKED, "Requires reviewed masks"
            )
        for image_key in set(self._analysis_caches) | set(self._analyses):
            self._cache_dirty_nodes.setdefault(image_key, set()).update(
                affected_set
            )
            self._analyses.pop(image_key, None)
        self.pipeline_canvas.refresh(affected)
        current_key = self._current_image_key()
        if current_key is not None and current_key in self._analysis_caches:
            self._analyze_current_image(dirty_nodes=affected_set)
        self.statusBar().showMessage(
            f"Reset {self.pipeline.node(node_id).title} to defaults; recomputing dependents."
        )

    def _refresh_display_only_analysis_parameter(
        self, node_id: str, key: str
    ) -> None:
        """Refresh a diagnostic setting without scheduling analytical work."""

        if (
            node_id != "reference_edge_probability"
            or key != "net_physical_edge_internal_scale"
        ):
            # Future display-only controls still get an honest graph-card
            # refresh.  A control that affects an AnalysisLayers field must add
            # its explicit immutable-result replacement below.
            self.pipeline_canvas.refresh((node_id,))
            self.pipeline_inspector.refresh_status()
            return

        current_key = self._current_image_key()
        current_result = None
        for image_key, result in tuple(self._analyses.items()):
            updated_result = self._normalize_display_only_analysis_result(result)
            self._analyses[image_key] = updated_result
            if image_key == current_key:
                current_result = updated_result

        if current_result is not None:
            self.image_view.show_analysis(current_result, render=False)
            self.pipeline_inspector.set_analysis_result(current_result)
            self.image_view.refresh_analysis()
        selected_overlay = str(self.overlay_combo.currentData() or "")
        self._update_overlay_legend(selected_overlay)
        self.pipeline_canvas.refresh((node_id,))
        self.pipeline_inspector.refresh_status()
        self.statusBar().showMessage(
            f"Changed {self.pipeline.node(node_id).title} display; "
            "reused all cached analysis products."
        )

    def _normalize_display_only_analysis_result(self, result):
        """Return a result wrapper carrying the current presentation settings."""

        layers = getattr(result, "layers", None)
        if layers is None:
            return result
        value = float(
            self.pipeline.node("reference_edge_probability").parameters[
                "net_physical_edge_internal_scale"
            ]
        )
        if float(layers.net_physical_edge_internal_scale) == value:
            return result
        return replace(
            result,
            layers=replace(
                layers,
                net_physical_edge_internal_scale=value,
            ),
        )

    def _validate_settings_override(
        self, node_id: str, key: str, value: object
    ) -> None:
        baseline_nodes = {
            "seed_scale_estimation",
            "foreground_segmentation",
            "distance_candidates",
            "identification",
        }
        if self.pipeline.is_active("circle_candidates"):
            baseline_nodes.add("circle_candidates")
        layer_nodes = {
            "perimeter_background_reference",
            "background_likelihood",
            "refined_background_likelihood",
            "foreground_noise_likelihood",
            "edge_gradients",
            "surface_darkness_gradients",
            "lightening_gradient_ceiling",
            "darkening_gradient_ceiling",
            "frequency_noise_masks",
            "edge_ridges",
            "reference_texture_prototypes",
            "reference_edge_probability",
            "reference_edge_ridges",
            "edge_traces",
            "instance_masks",
            "seed_edge_curves",
        }
        advanced_nodes = set(ADVANCED_NODE_MODES)
        if node_id == "layout_detection":
            values = dict(self.pipeline.node(node_id).parameters)
            values[key] = value
            DishDetectionSettings(**values)
        elif node_id in baseline_nodes:
            values: dict[str, object] = {}
            for stage_id in baseline_nodes:
                values.update(self.pipeline.node(stage_id).parameters)
            values[key] = value
            BaselineSettings(**values)
        elif node_id in layer_nodes:
            values = {}
            for layer_id in layer_nodes:
                values.update(self.pipeline.node(layer_id).parameters)
            values[key] = value
            AnalysisLayerSettings(**values)
        elif node_id in advanced_nodes:
            values = {}
            for advanced_id in advanced_nodes:
                values.update(self.pipeline.node(advanced_id).parameters)
            values[key] = value
            AdvancedAnalysisSettings(**values)
        elif node_id == "procedural_instances":
            values = dict(self.pipeline.node(node_id).parameters)
            values[key] = value
            ProceduralInstanceSettings(**values)
        elif node_id == "unet_instances":
            values = dict(self.pipeline.node(node_id).parameters)
            values[key] = value
            UNetPipelineSettings(**values)
        elif node_id == "stardist_instances":
            values = dict(self.pipeline.node(node_id).parameters)
            values[key] = value
            StarDistPipelineSettings(**values)

    @Slot(object)
    def _pipeline_connections_changed(self, affected) -> None:
        """Invalidate calculations after an authored edge edit."""

        affected_set = set(affected)
        if not affected_set:
            return
        for image_key in set(self._analysis_caches) | set(self._analyses):
            self._cache_dirty_nodes.setdefault(image_key, set()).update(
                affected_set
            )
            self._analyses.pop(image_key, None)
        self.pipeline_canvas.refresh(affected_set)
        if self._selected_pipeline_node in self.pipeline.nodes:
            self.pipeline_inspector.set_node(
                self.pipeline.node(self._selected_pipeline_node)
            )
        current_key = self._current_image_key()
        if current_key is not None and current_key in self._analysis_caches:
            self._analyze_current_image(dirty_nodes=affected_set)
        self._update_analysis_availability()
        disabled_count = sum(
            not self.pipeline.node(node_id).enabled
            for node_id in affected_set
            if self.pipeline.is_active(node_id)
        )
        self.statusBar().showMessage(
            "Pipeline wiring changed; recalculating affected nodes."
            if not disabled_count
            else f"Pipeline wiring changed; {disabled_count} dependent node(s) "
            "are disabled until their required inputs are reconnected."
        )

    @Slot(str, bool)
    def _pipeline_enabled_changed(self, node_id: str, enabled: bool) -> None:
        if enabled and node_id in {"unet_instances", "stardist_instances"}:
            configured = Path(
                str(self.pipeline.node(node_id).parameters["checkpoint_path"])
            ).expanduser()
            checkpoint = (
                configured.resolve()
                if configured.is_absolute()
                else (self._root / configured).resolve()
            )
            if not checkpoint.is_file():
                QMessageBox.warning(
                    self,
                    "Learned-model checkpoint not found",
                    f"{checkpoint}\n\nTrain or copy a compatible checkpoint, then enable this node.",
                )
                self.pipeline_inspector.set_node(self.pipeline.node(node_id))
                return
        try:
            affected = self.pipeline.set_enabled(node_id, enabled)
        except ValueError as error:
            QMessageBox.warning(self, "Cannot enable pipeline node", str(error))
            self.pipeline_inspector.set_node(self.pipeline.node(node_id))
            return
        if not affected:
            return
        affected_set = set(affected)
        for image_key in set(self._analysis_caches) | set(self._analyses):
            self._cache_dirty_nodes.setdefault(image_key, set()).update(
                affected_set
            )
            self._analyses.pop(image_key, None)
        if node_id == "background_likelihood":
            self._stop_background_point_editing()
            if enabled:
                self.pipeline.set_status(
                    "background_likelihood", NodeStatus.IDLE, "Ready"
                )
                self.pipeline.set_status(
                    "refined_background_likelihood", NodeStatus.IDLE, "Ready"
                )
            else:
                self.pipeline.set_status(
                    "background_likelihood", NodeStatus.BYPASSED, "Bypassed"
                )
                self.pipeline.set_status(
                    "refined_background_likelihood",
                    NodeStatus.BLOCKED,
                    "Background colour analysis disabled",
                )
        for affected_id in affected:
            if not self.pipeline.node(affected_id).enabled:
                self.pipeline.set_status(
                    affected_id,
                    NodeStatus.BYPASSED,
                    "Disabled by pipeline dependency",
                )
        self.pipeline_canvas.refresh(affected)
        self.pipeline_inspector.set_node(self.pipeline.node(node_id))
        current_key = self._current_image_key()
        if (
            current_key is not None
            and current_key in self._analysis_caches
        ):
            self._analyze_current_image(dirty_nodes=affected_set)
        self._update_analysis_availability()
        self._sync_background_controls()

    @Slot(str)
    def _species_changed(self, species: str) -> None:
        if self.image_view.image_path is None:
            return
        self.pipeline.set_status("metadata", NodeStatus.COMPLETE, species)
        affected = {
            node_id
            for node_id in ("unet_instances", "stardist_instances")
            if self.pipeline.node(node_id).enabled
        }
        if affected:
            self.pipeline.invalidate(affected)
            key = self._current_image_key()
            if key is not None:
                self._cache_dirty_nodes.setdefault(key, set()).update(affected)
                self._analyses.pop(key, None)
                if key in self._analysis_caches:
                    self._analyze_current_image(dirty_nodes=affected)
        self.pipeline_canvas.refresh(("metadata", *affected))

    def _update_analysis_availability(self) -> None:
        path = self.image_view.image_path
        running = (
            bool(self._active_tasks)
            or self._learning_training_task is not None
            or self._procedural_fit_task is not None
        )
        available = (
            path is not None
            and not running
        )
        self.analyze_button.setEnabled(available)
        self.analyze_action.setEnabled(available)
        if hasattr(self, "train_learning_model_action"):
            self.train_learning_model_action.setEnabled(not running)

    def _show_pipeline_workspace(self) -> None:
        self.pipeline_canvas.show()
        self.image_view.hide()
        self.pipeline_canvas.select_node(self._selected_pipeline_node)

    def _show_image_workspace(self) -> None:
        self.image_view.show()
        self.pipeline_canvas.hide()
        self.actual_size_action.setEnabled(True)

    def _show_split_workspace(self) -> None:
        self.image_view.show()
        self.pipeline_canvas.show()
        self.workspace_splitter.setSizes((560, 360))
        self.actual_size_action.setEnabled(True)
        self.pipeline_canvas.refresh()

    def _fit_active_workspace(self) -> None:
        if self.pipeline_canvas.isVisible():
            self.pipeline_canvas.fit_graph()
        if self.image_view.isVisible():
            self.image_view.fit_image()

    def _show_runtime_summary(self) -> None:
        from seedvision import __version__

        QMessageBox.information(
            self,
            "Seed Fiddle runtime",
            f"Seed Fiddle {__version__}\n"
            f"Application folder: {self._root}\n\n"
            "Run seed_vision.py --diagnostics for package and GPU details.",
        )

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        """Finish the sole GPU job and deterministically release owned caches."""

        self._pending_analysis_key = None
        if self._learning_training_task is not None:
            self._learning_training_task.cancel()
        if self._procedural_fit_task is not None:
            self._procedural_fit_task.cancel()
        self._thread_pool.clear()
        if not self._thread_pool.waitForDone(10_000):
            self._thread_pool.waitForDone()
        self._active_tasks.clear()
        if self._learning_training_progress is not None:
            self._learning_training_progress.close()
        self._learning_training_progress = None
        self._learning_training_task = None
        if self._procedural_fit_progress is not None:
            self._procedural_fit_progress.close()
        self._procedural_fit_progress = None
        self._procedural_fit_task = None
        self.image_view.clear_analysis()
        self.pipeline_inspector.set_analysis_result(None)
        for key in tuple(self._analysis_caches):
            self._discard_analysis_cache(key)
        self._release_unused_cuda_blocks()
        super().closeEvent(event)


def _background_profile_detail(profile, prefix: str) -> str:
    if profile is None:
        return f"{prefix}; colour range unavailable"
    low = "/".join(str(value) for value in profile.bgr_low)
    high = "/".join(str(value) for value in profile.bgr_high)
    mode_frequencies = "/".join(
        f"{weight:.0%}" for weight in profile.component_weights
    )
    mode_detail = (
        f"; painted-mode frequencies {mode_frequencies}"
        if mode_frequencies
        else ""
    )
    return (
        f"{prefix}; likely BGR {low} to {high}; "
        f"{profile.sample_count:,} px ({profile.sample_fraction:.1%}); "
        f"{max(1, len(profile.component_centres_lab))} colour mode(s){mode_detail}; "
        f"{profile.refinement_iterations} refinement round(s)"
    )


def _format_class_distribution(names, proportions) -> str:
    populated = [
        f"{name} {proportion:.0%}"
        for name, proportion in zip(names, proportions, strict=True)
        if proportion > 0.0
    ]
    return ", ".join(populated) if populated else "no assigned seeds"
