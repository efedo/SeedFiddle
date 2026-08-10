"""Seed Vision desktop interface for pipeline control and image review."""

from __future__ import annotations

import json
from pathlib import Path

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
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from seedvision.pipeline import NodeStatus, build_default_pipeline
from seedvision.segmentation import (
    AdvancedAnalysisSettings,
    AnalysisLayerSettings,
    BaselineSettings,
    CalibrationSettings,
    DishDetectionSettings,
    PipelineAnalysisCache,
)
from seedvision.visualization import ADVANCED_NODE_MODES, ADVANCED_OVERLAY_LABELS
from seedvision.ui.image_view import ImageView, SUPPORTED_SUFFIXES
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
    "edge_gradients",
    "undirected_edges",
    "directed_edges",
    "edge_ridges",
    "edge_traces",
    "seed_edge_curves",
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
    "foreground_segmentation",
    "distance_candidates",
    "circle_candidates",
    "identification",
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

OVERLAY_NODE_OWNERS = {
    "raw_image": "raw_images",
    "calibrated_image": "scale_calibration",
    "deskew_colour": "deskew_colour",
    "colour_reference": "colour_reference",
    "ruler_detection": "ruler_detection",
    "layout_detection": "layout_detection",
    "seed_scale_estimation": "seed_scale_estimation",
    "foreground_feature": "foreground_segmentation",
    "foreground_mask": "foreground_segmentation",
    "foreground_binary_mask": "foreground_segmentation",
    "distance_transform": "distance_candidates",
    "distance_candidates": "distance_candidates",
    "circle_candidates": "circle_candidates",
    "proposals": "identification",
    "instance_masks": "instance_masks",
    "background_likelihood": "background_likelihood",
    "refined_background_likelihood": "refined_background_likelihood",
    "edge_gradients": "edge_gradients",
    "undirected_edges": "undirected_edges",
    "directed_edges": "directed_edges",
    "edge_ridges": "edge_ridges",
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
}
for _node_id, _mode in ADVANCED_NODE_MODES.items():
    OVERLAY_NODE_OWNERS[_mode] = _node_id
OVERLAY_NODE_OWNERS.update(
    {
        "boundary_magnitude": "boundary_normals",
        "contested_pixels": "assignment_confidence",
        "shadow_likelihood": "illumination_decomposition",
        "reflectance_image": "illumination_decomposition",
        "glare_likelihood": "image_quality",
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
    if mode.startswith("directional_background:"):
        return "refined_background_likelihood"
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
        background_reference_points: tuple[tuple[float, float], ...],
        foreground_reference_points: tuple[tuple[float, float], ...],
        background_reference_mask: np.ndarray | None,
        foreground_reference_mask: np.ndarray | None,
        background_colour_enabled: bool,
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
        self.background_reference_points = background_reference_points
        self.foreground_reference_points = foreground_reference_points
        self.background_reference_mask = (
            None
            if background_reference_mask is None
            else np.asarray(background_reference_mask, dtype=bool).copy()
        )
        self.foreground_reference_mask = (
            None
            if foreground_reference_mask is None
            else np.asarray(foreground_reference_mask, dtype=bool).copy()
        )
        self.background_colour_enabled = background_colour_enabled
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
                background_reference_points=self.background_reference_points,
                foreground_reference_points=self.foreground_reference_points,
                background_reference_mask=self.background_reference_mask,
                foreground_reference_mask=self.foreground_reference_mask,
                background_colour_enabled=self.background_colour_enabled,
                node_cache=self.node_cache,
                dirty_nodes=self.dirty_nodes,
                progress_callback=lambda node_id, state: self.signals.node_progress.emit(
                    str(self.path), node_id, state, self.pipeline_revision
                ),
            )
        except Exception as error:  # noqa: BLE001 - cross-thread error boundary
            self.signals.failed.emit(
                str(self.path), str(error), self.pipeline_revision
            )
            return
        self.signals.completed.emit(result, self.pipeline_revision)


class MainWindow(QMainWindow):
    """Main desktop window for visual pipeline control and seed review."""

    def __init__(self, root: Path, parent=None) -> None:
        super().__init__(parent)
        self._root = root
        self._image_paths: dict[str, Path] = {}
        self._analyses: dict[str, object] = {}
        self._analysis_caches: dict[str, PipelineAnalysisCache] = {}
        self._cache_dirty_nodes: dict[str, set[str]] = {}
        # Draft masks are edited by the viewer. Applied masks are immutable
        # analysis inputs until the user explicitly confirms the draft.
        self._draft_background_reference_masks: dict[str, np.ndarray] = {}
        self._draft_foreground_reference_masks: dict[str, np.ndarray] = {}
        self._applied_background_reference_masks: dict[str, np.ndarray] = {}
        self._applied_foreground_reference_masks: dict[str, np.ndarray] = {}
        self._reference_masks_dirty: set[str] = set()
        self._active_tasks: dict[str, _AnalysisTask] = {}
        self._thread_pool = QThreadPool.globalInstance()
        self._selected_pipeline_node = "identification"
        self._selecting_node_from_overlay = False
        self._selecting_overlay_from_node = False
        self.pipeline = build_default_pipeline()

        self.setWindowTitle("Seed Vision — visual analysis pipeline")
        self.setMinimumSize(1100, 700)
        self.resize(1540, 920)

        self.image_view = ImageView(self)
        self.image_view.image_dropped.connect(self._add_and_open_image)
        self.image_view.reference_mask_edited.connect(
            self._reference_mask_edited
        )
        self.pipeline_canvas = PipelineCanvas(self.pipeline, self)
        self.pipeline_canvas.node_selected.connect(self._pipeline_node_selected)
        self.pipeline_canvas.parameter_changed.connect(
            self._pipeline_parameter_changed
        )
        self.pipeline_inspector = PipelineInspector(self)
        self.pipeline_inspector.parameter_changed.connect(
            self._pipeline_parameter_changed
        )
        self.pipeline_inspector.enabled_changed.connect(self._pipeline_enabled_changed)

        self.image_list = QListWidget(self)
        self.image_list.setAlternatingRowColors(True)
        self.image_list.itemActivated.connect(self._open_list_item)
        self.species_combo = QComboBox(self)
        self.species_combo.addItems(self._load_species_names())
        self.species_combo.currentTextChanged.connect(self._species_changed)

        self._build_actions()
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

        self.analyze_action = QAction("Run to seed identification", self)
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

        self.diagnostics_action = QAction("Runtime summary", self)
        self.diagnostics_action.triggered.connect(self._show_runtime_summary)

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(self.open_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)

        analysis_menu = self.menuBar().addMenu("&Analysis")
        analysis_menu.addAction(self.analyze_action)

        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(self.image_workspace_action)
        view_menu.addAction(self.pipeline_workspace_action)
        view_menu.addAction(self.split_workspace_action)
        view_menu.addSeparator()
        view_menu.addAction(self.fit_action)
        view_menu.addAction(self.actual_size_action)

        help_menu = self.menuBar().addMenu("&Help")
        help_menu.addAction(self.diagnostics_action)

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Workflow", self)
        toolbar.setIconSize(QSize(18, 18))
        toolbar.setMovable(False)
        toolbar.addAction(self.open_action)
        toolbar.addAction(self.analyze_action)
        toolbar.addSeparator()
        toolbar.addAction(self.image_workspace_action)
        toolbar.addAction(self.pipeline_workspace_action)
        toolbar.addAction(self.split_workspace_action)
        toolbar.addSeparator()
        toolbar.addAction(self.fit_action)
        toolbar.addAction(self.actual_size_action)
        self.addToolBar(toolbar)

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

        # Keep the selected node inspector near the top of the panel. It is
        # the primary control surface while working in the graph; calibration
        # summaries and global actions remain available immediately below it.
        layout.addWidget(self._separator())
        layout.addWidget(self._section_label("Selected pipeline node"))
        layout.addWidget(self.pipeline_inspector)

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
        self.analyze_button = QPushButton("Run to seed identification", content)
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

        layout.addWidget(self._separator())
        layout.addWidget(self._section_label("Viewer overlay"))
        self.overlay_combo = QComboBox(content)
        self.overlay_combo.addItem("Raw image", "raw_image")
        self.overlay_combo.addItem("Detected colour swatches", "colour_reference")
        self.overlay_combo.addItem(
            "Deskewed colour image + swatches", "deskew_colour"
        )
        self.overlay_combo.addItem("Detected ruler", "ruler_detection")
        self.overlay_combo.addItem(
            "Calibrated image & 5 cm scale", "calibrated_image"
        )
        self.overlay_combo.addItem("Detected vessel layout", "layout_detection")
        self.overlay_combo.addItem("Reference seed scale", "seed_scale_estimation")
        self.overlay_combo.addItem("Foreground strength", "foreground_feature")
        self.overlay_combo.addItem("Foreground probability", "foreground_mask")
        self.overlay_combo.addItem(
            "Foreground binary proposal mask", "foreground_binary_mask"
        )
        self.overlay_combo.addItem("Distance transform", "distance_transform")
        self.overlay_combo.addItem("Distance-peak candidates", "distance_candidates")
        self.overlay_combo.addItem("Circle candidates", "circle_candidates")
        self.overlay_combo.addItem("Seed proposals", "proposals")
        self.overlay_combo.addItem("Instance colour masks", "instance_masks")
        self.overlay_combo.addItem(
            "Background colour likelihood", "background_likelihood"
        )
        self.overlay_combo.addItem(
            "Refined background (noise frequency)",
            "refined_background_likelihood",
        )
        self.overlay_combo.addItem("Shared edge magnitude", "edge_gradients")
        self.overlay_combo.addItem("Edge tangent (undirected)", "undirected_edges")
        self.overlay_combo.addItem("Edge tangent (directed)", "directed_edges")
        self.overlay_combo.addItem("Thinned edge ridges", "edge_ridges")
        self.overlay_combo.addItem("Oriented edge traces", "edge_traces")
        self.overlay_combo.addItem("Trace continuity", "edge_trace_continuity")
        self.overlay_combo.addItem("Trace gap confidence", "edge_trace_gap_confidence")
        self.overlay_combo.addItem("Radius confirmation", "edge_radius_confirmation")
        self.overlay_combo.addItem("Circle-fit confidence", "edge_circle_fit")
        self.overlay_combo.addItem("Ellipse-fit confidence", "edge_ellipse_fit")
        self.overlay_combo.addItem("Circle/ellipse fit residual", "edge_fit_residual")
        self.overlay_combo.addItem("Seed-centre votes", "edge_centre_votes")
        self.overlay_combo.addItem("Semantic boundary side", "edge_semantic_sides")
        self.overlay_combo.addItem("Rejected edge reasons", "edge_rejections")
        self.overlay_combo.addItem("Fitted centres and ellipses", "edge_fit_geometry")
        self.overlay_combo.addItem("Final seed-boundary confidence", "seed_edge_curves")
        for label, mode in ADVANCED_OVERLAY_LABELS:
            self.overlay_combo.addItem(label, mode)
        self.overlay_combo.addItem("None", "none")
        self.overlay_combo.setEnabled(False)
        self.overlay_combo.currentIndexChanged.connect(self._overlay_changed)
        overlay_form = QFormLayout()
        overlay_form.addRow("Layer", self.overlay_combo)

        opacity_widget = QWidget(content)
        opacity_layout = QHBoxLayout(opacity_widget)
        opacity_layout.setContentsMargins(0, 0, 0, 0)
        self.overlay_opacity_slider = QSlider(Qt.Orientation.Horizontal, content)
        self.overlay_opacity_slider.setRange(0, 100)
        self.overlay_opacity_slider.setValue(68)
        self.overlay_opacity_slider.setEnabled(False)
        self.overlay_opacity_slider.valueChanged.connect(
            self._overlay_opacity_changed
        )
        self.overlay_opacity_label = QLabel("68%", content)
        self.overlay_opacity_label.setMinimumWidth(36)
        opacity_layout.addWidget(self.overlay_opacity_slider, 1)
        opacity_layout.addWidget(self.overlay_opacity_label)
        overlay_form.addRow("Opacity", opacity_widget)
        self.overlay_owner_label = self._muted_label("No graph node")
        overlay_form.addRow("Graph node", self.overlay_owner_label)
        layout.addLayout(overlay_form)
        self.overlay_legend_label = self._muted_label(
            "Run the analysis to inspect its intermediate raster layers."
        )
        layout.addWidget(self.overlay_legend_label)

        layout.addWidget(self._separator())
        layout.addWidget(self._section_label("Painted background reference"))
        self.background_enabled_checkbox = QCheckBox(
            "Use background colour analysis", content
        )
        self.background_enabled_checkbox.setChecked(
            self.pipeline.node("background_likelihood").enabled
        )
        self.background_enabled_checkbox.toggled.connect(
            self._background_enabled_toggled
        )
        layout.addWidget(self.background_enabled_checkbox)
        reference_buttons = QWidget(content)
        reference_button_layout = QHBoxLayout(reference_buttons)
        reference_button_layout.setContentsMargins(0, 0, 0, 0)
        self.background_point_button = QPushButton(
            "Paint background", reference_buttons
        )
        self.background_point_button.setCheckable(True)
        self.background_point_button.setEnabled(False)
        self.background_point_button.toggled.connect(
            self._background_point_editing_changed
        )
        self.clear_background_points_button = QPushButton(
            "Clear", reference_buttons
        )
        self.clear_background_points_button.setEnabled(False)
        self.clear_background_points_button.clicked.connect(
            self._clear_background_points
        )
        reference_button_layout.addWidget(self.background_point_button, 1)
        reference_button_layout.addWidget(self.clear_background_points_button)
        layout.addWidget(reference_buttons)
        self.background_reference_label = self._muted_label(
            "Automatic background colour selection; no painted area."
        )
        layout.addWidget(self.background_reference_label)

        layout.addWidget(self._separator())
        layout.addWidget(self._section_label("Painted foreground reference"))
        foreground_buttons = QWidget(content)
        foreground_button_layout = QHBoxLayout(foreground_buttons)
        foreground_button_layout.setContentsMargins(0, 0, 0, 0)
        self.foreground_point_button = QPushButton(
            "Paint foreground", foreground_buttons
        )
        self.foreground_point_button.setCheckable(True)
        self.foreground_point_button.setEnabled(False)
        self.foreground_point_button.toggled.connect(
            self._foreground_point_editing_changed
        )
        self.clear_foreground_points_button = QPushButton(
            "Clear", foreground_buttons
        )
        self.clear_foreground_points_button.setEnabled(False)
        self.clear_foreground_points_button.clicked.connect(
            self._clear_foreground_points
        )
        foreground_button_layout.addWidget(self.foreground_point_button, 1)
        foreground_button_layout.addWidget(self.clear_foreground_points_button)
        layout.addWidget(foreground_buttons)
        self.foreground_reference_label = self._muted_label(
            "No painted foreground reference."
        )
        layout.addWidget(self.foreground_reference_label)

        brush_widget = QWidget(content)
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

        brush_mode_widget = QWidget(content)
        brush_mode_layout = QHBoxLayout(brush_mode_widget)
        brush_mode_layout.setContentsMargins(0, 0, 0, 0)
        self.reference_paint_mode_button = QPushButton(
            "Paint", brush_mode_widget
        )
        self.reference_paint_mode_button.setCheckable(True)
        self.reference_paint_mode_button.setChecked(True)
        self.reference_paint_mode_button.setToolTip(
            "Left-drag adds pixels to the selected reference mask."
        )
        self.reference_eraser_button = QPushButton(
            "Eraser", brush_mode_widget
        )
        self.reference_eraser_button.setCheckable(True)
        self.reference_eraser_button.setToolTip(
            "Left-drag removes pixels from the selected reference mask. The dashed "
            "red brush outline indicates erasing."
        )
        self.reference_brush_mode_group = QButtonGroup(brush_mode_widget)
        self.reference_brush_mode_group.setExclusive(True)
        self.reference_brush_mode_group.addButton(
            self.reference_paint_mode_button
        )
        self.reference_brush_mode_group.addButton(self.reference_eraser_button)
        self.reference_eraser_button.toggled.connect(
            self._reference_eraser_toggled
        )
        brush_mode_layout.addWidget(self.reference_paint_mode_button)
        brush_mode_layout.addWidget(self.reference_eraser_button)

        brush_form = QFormLayout()
        brush_form.addRow("Brush mode", brush_mode_widget)
        brush_form.addRow("Brush radius", brush_widget)
        layout.addLayout(brush_form)

        confirmation_buttons = QWidget(content)
        confirmation_layout = QHBoxLayout(confirmation_buttons)
        confirmation_layout.setContentsMargins(0, 0, 0, 0)
        self.apply_reference_masks_button = QPushButton(
            "Apply reference masks", confirmation_buttons
        )
        self.apply_reference_masks_button.setEnabled(False)
        self.apply_reference_masks_button.setToolTip(
            "Confirm both painted masks and run only the affected pipeline nodes."
        )
        self.apply_reference_masks_button.clicked.connect(
            self._apply_reference_masks
        )
        self.revert_reference_masks_button = QPushButton(
            "Revert edits", confirmation_buttons
        )
        self.revert_reference_masks_button.setEnabled(False)
        self.revert_reference_masks_button.clicked.connect(
            self._revert_reference_masks
        )
        confirmation_layout.addWidget(self.apply_reference_masks_button, 1)
        confirmation_layout.addWidget(self.revert_reference_masks_button)
        layout.addWidget(confirmation_buttons)
        self.reference_confirmation_label = self._muted_label(
            "Painting is a draft and does not run calculations until applied."
        )
        layout.addWidget(self.reference_confirmation_label)

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
        self._pipeline_image_loaded(path)
        self._update_analysis_availability()
        cached = self._analyses.get(str(path.resolve()).casefold())
        if cached is None:
            self._show_pending_result()
        else:
            self._mark_analysis_complete(cached)
            self._show_analysis_result(cached)
        self._sync_background_controls()
        self.statusBar().showMessage(f"Loaded {path}")

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
            "circle_candidates",
            "identification",
        ):
            values.update(self.pipeline.node(node_id).parameters)
        return BaselineSettings(**values)

    def _dish_settings(self) -> DishDetectionSettings:
        return DishDetectionSettings(**self.pipeline.node("layout_detection").parameters)

    def _layer_settings(self) -> AnalysisLayerSettings:
        values: dict[str, object] = {}
        for node_id in (
            "background_likelihood",
            "refined_background_likelihood",
            "edge_gradients",
            "edge_ridges",
            "edge_traces",
            "instance_masks",
            "seed_edge_curves",
        ):
            values.update(self.pipeline.node(node_id).parameters)
        return AnalysisLayerSettings(**values)

    def _advanced_settings(self) -> AdvancedAnalysisSettings:
        values: dict[str, object] = {}
        for node_id in ADVANCED_NODE_MODES:
            values.update(self.pipeline.node(node_id).parameters)
        return AdvancedAnalysisSettings(**values)

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
        identification = self.pipeline.node("identification")
        if path is None or not identification.enabled:
            return
        key = str(path.resolve()).casefold()
        pending_dirty = self._cache_dirty_nodes.setdefault(key, set())
        if dirty_nodes:
            pending_dirty.update(dirty_nodes)
        if key in self._active_tasks:
            return
        requested_dirty = frozenset(pending_dirty)
        pending_dirty.clear()
        node_cache = self._analysis_caches.setdefault(
            key, PipelineAnalysisCache()
        )
        task = _AnalysisTask(
            path,
            self._baseline_settings(),
            self._calibration_settings(),
            self._dish_settings(),
            self._layer_settings(),
            self._advanced_settings(),
            (),
            (),
            self._applied_background_reference_masks.get(key),
            self._applied_foreground_reference_masks.get(key),
            self.pipeline.node("background_likelihood").enabled,
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
            "Executing layout detection and seed identification in the background."
        )
        self.statusBar().showMessage(f"Analysing {path.name}…")
        self._thread_pool.start(task)

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
                self._analyze_current_image()
            return
        self._analyses[key] = result
        self._mark_analysis_complete(result)
        if self.image_view.image_path == path:
            self._show_analysis_result(result)
        self._update_analysis_availability()
        self._sync_background_controls()
        self.statusBar().showMessage(
            f"Generated {result.count:,} approximate proposals for {path.name}."
        )
        if self._cache_dirty_nodes.get(key):
            self._analyze_current_image()

    @Slot(str, str, int)
    def _analysis_failed(
        self, path_text: str, error: str, pipeline_revision: int
    ) -> None:
        del pipeline_revision
        path = Path(path_text)
        self._active_tasks.pop(str(path.resolve()).casefold(), None)
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

    def _show_analysis_result(self, result) -> None:
        self._sync_directional_overlay_choices(result)
        self.image_view.show_analysis(result)
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
        self.reference_brush_label.setText(f"{brush_radius} px")
        self.image_view.set_reference_brush_radius(brush_radius)
        path = result.image_path
        key = str(path.resolve()).casefold() if path is not None else ""
        self.image_view.set_reference_masks(
            self._draft_background_reference_masks.get(
                key, self._applied_background_reference_masks.get(key)
            ),
            self._draft_foreground_reference_masks.get(
                key, self._applied_foreground_reference_masks.get(key)
            ),
        )
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
        self.image_view.set_overlay_mode(str(self.overlay_combo.currentData()))
        self.image_view.set_overlay_opacity(
            self.overlay_opacity_slider.value() / 100.0
        )
        self._update_overlay_legend(str(self.overlay_combo.currentData()))
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
        self.count_label.setText(f"≈ {result.count:,} seeds")
        self.crowding_label.setText(
            f"Crowding: {result.crowding}. Method: {result.method}."
        )
        self.warning_label.setText("\n".join(result.warnings))
        self._sync_background_controls()

    def _sync_directional_overlay_choices(self, result) -> None:
        selected = self.overlay_combo.currentData()
        for index in range(self.overlay_combo.count() - 1, -1, -1):
            mode = str(self.overlay_combo.itemData(index))
            if mode.startswith((
                "directional_background:",
                "colour_probability:",
                "pattern_probability:",
            )):
                self.overlay_combo.removeItem(index)
        none_index = self.overlay_combo.findData("none")
        for direction_index, angle in enumerate(
            result.layers.directional_background_angles_degrees
        ):
            self.overlay_combo.insertItem(
                none_index,
                f"Background ray {angle:g}°",
                f"directional_background:{direction_index}",
            )
            none_index += 1
        for probability_index, class_name in enumerate(
            result.advanced.colour_class_names
        ):
            self.overlay_combo.insertItem(
                none_index,
                f"Colour probability: {class_name}",
                f"colour_probability:{probability_index}",
            )
            none_index += 1
        for probability_index, class_name in enumerate(
            result.advanced.pattern_class_names
        ):
            self.overlay_combo.insertItem(
                none_index,
                f"Pattern probability: {class_name}",
                f"pattern_probability:{probability_index}",
            )
            none_index += 1
        restored = self.overlay_combo.findData(selected)
        if restored >= 0:
            self.overlay_combo.setCurrentIndex(restored)

    def _current_image_key(self) -> str | None:
        path = self.image_view.image_path
        if path is None:
            return None
        return str(path.resolve()).casefold()

    def _stop_background_point_editing(self) -> None:
        if not hasattr(self, "background_point_button"):
            return
        with QSignalBlocker(self.background_point_button):
            self.background_point_button.setChecked(False)
        self.image_view.set_background_point_editing(False)

    def _stop_foreground_point_editing(self) -> None:
        if not hasattr(self, "foreground_point_button"):
            return
        with QSignalBlocker(self.foreground_point_button):
            self.foreground_point_button.setChecked(False)
        self.image_view.set_foreground_point_editing(False)

    def _stop_reference_point_editing(self) -> None:
        self._stop_background_point_editing()
        self._stop_foreground_point_editing()

    def _sync_background_controls(self) -> None:
        if not hasattr(self, "background_point_button"):
            return
        key = self._current_image_key()
        enabled = self.pipeline.node("background_likelihood").enabled
        with QSignalBlocker(self.background_enabled_checkbox):
            self.background_enabled_checkbox.setChecked(enabled)
        running = key in self._active_tasks if key is not None else False
        has_result = self.image_view._analysis_result is not None
        can_edit = enabled and has_result and not running
        self.background_point_button.setEnabled(can_edit)
        self.foreground_point_button.setEnabled(has_result and not running)
        background_mask = self._draft_background_reference_masks.get(
            key or "", self._applied_background_reference_masks.get(key or "")
        )
        foreground_mask = self._draft_foreground_reference_masks.get(
            key or "", self._applied_foreground_reference_masks.get(key or "")
        )
        background_count = self._mask_pixel_count(background_mask)
        foreground_count = self._mask_pixel_count(foreground_mask)
        dirty = key in self._reference_masks_dirty if key is not None else False
        self.clear_background_points_button.setEnabled(
            bool(background_count) and not running
        )
        self.clear_foreground_points_button.setEnabled(
            bool(foreground_count) and not running
        )
        self.apply_reference_masks_button.setEnabled(dirty and not running)
        self.revert_reference_masks_button.setEnabled(dirty and not running)
        self.reference_brush_slider.setEnabled(has_result and not running)
        self.reference_paint_mode_button.setEnabled(has_result and not running)
        self.reference_eraser_button.setEnabled(has_result and not running)
        if not can_edit and self.background_point_button.isChecked():
            self._stop_background_point_editing()
        if (not has_result or running) and self.foreground_point_button.isChecked():
            self._stop_foreground_point_editing()
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
        self.reference_confirmation_label.setText(
            "Draft changed. Apply once painting is complete; analysis is still using "
            "the previous confirmed masks."
            if dirty
            else "Painting is a draft and does not run calculations until applied."
        )

    @staticmethod
    def _mask_pixel_count(mask: np.ndarray | None) -> int:
        return 0 if mask is None else int(np.count_nonzero(mask))

    @Slot(bool)
    def _background_enabled_toggled(self, enabled: bool) -> None:
        self._pipeline_enabled_changed("background_likelihood", enabled)

    @Slot(bool)
    def _background_point_editing_changed(self, enabled: bool) -> None:
        if enabled:
            with QSignalBlocker(self.foreground_point_button):
                self.foreground_point_button.setChecked(False)
        self.image_view.set_background_point_editing(enabled)
        if enabled:
            overlay_index = self.overlay_combo.findData("background_likelihood")
            if overlay_index >= 0:
                self.overlay_combo.setCurrentIndex(overlay_index)
            self.image_view.show()
            self.statusBar().showMessage(
                f"Background brush: left-drag to {self._reference_brush_action()}; "
                "right-drag temporarily erases."
            )
        else:
            self.statusBar().showMessage("Background reference painting finished.")

    @Slot(bool)
    def _foreground_point_editing_changed(self, enabled: bool) -> None:
        if enabled:
            with QSignalBlocker(self.background_point_button):
                self.background_point_button.setChecked(False)
        self.image_view.set_foreground_point_editing(enabled)
        if enabled:
            overlay_index = self.overlay_combo.findData("foreground_mask")
            if overlay_index >= 0:
                self.overlay_combo.setCurrentIndex(overlay_index)
            self.image_view.show()
            self.statusBar().showMessage(
                f"Foreground brush: left-drag to {self._reference_brush_action()}; "
                "right-drag temporarily erases."
            )
        else:
            self.statusBar().showMessage("Foreground reference painting finished.")

    @Slot(int)
    def _reference_brush_radius_changed(self, radius: int) -> None:
        self.reference_brush_label.setText(f"{radius} px")
        self.image_view.set_reference_brush_radius(float(radius))

    @Slot(bool)
    def _reference_eraser_toggled(self, enabled: bool) -> None:
        self.image_view.set_reference_erase_mode(enabled)
        active_class = (
            "background"
            if self.background_point_button.isChecked()
            else "foreground"
            if self.foreground_point_button.isChecked()
            else "reference"
        )
        self.statusBar().showMessage(
            f"{active_class.capitalize()} brush set to "
            f"{'erase' if enabled else 'paint'}."
        )

    def _reference_brush_action(self) -> str:
        return "erase" if self.reference_eraser_button.isChecked() else "paint"

    @Slot(str, object)
    def _reference_mask_edited(self, class_name: str, mask) -> None:
        key = self._current_image_key()
        if key is None or class_name not in {"background", "foreground"}:
            return
        values = self._empty_current_image_mask() if mask is None else np.asarray(
            mask, dtype=bool
        ).copy()
        target = (
            self._draft_background_reference_masks
            if class_name == "background"
            else self._draft_foreground_reference_masks
        )
        target[key] = values
        self._reference_masks_dirty.add(key)
        self._sync_background_controls()
        self.statusBar().showMessage(
            f"{class_name.capitalize()} mask edited; no calculations run. "
            "Apply the reference masks when painting is complete."
        )

    @Slot()
    def _clear_background_points(self) -> None:
        key = self._current_image_key()
        if key is None:
            return
        self._draft_background_reference_masks[key] = self._empty_current_image_mask()
        self._reference_masks_dirty.add(key)
        self.image_view.set_reference_masks(
            None,
            self._draft_foreground_reference_masks.get(
                key, self._applied_foreground_reference_masks.get(key)
            ),
        )
        self._sync_background_controls()

    @Slot()
    def _clear_foreground_points(self) -> None:
        key = self._current_image_key()
        if key is None:
            return
        self._draft_foreground_reference_masks[key] = self._empty_current_image_mask()
        self._reference_masks_dirty.add(key)
        self.image_view.set_reference_masks(
            self._draft_background_reference_masks.get(
                key, self._applied_background_reference_masks.get(key)
            ),
            None,
        )
        self._sync_background_controls()

    @Slot()
    def _revert_reference_masks(self) -> None:
        key = self._current_image_key()
        if key is None:
            return
        self._copy_applied_to_draft(key)
        self._reference_masks_dirty.discard(key)
        self.image_view.set_reference_masks(
            self._draft_background_reference_masks.get(key),
            self._draft_foreground_reference_masks.get(key),
        )
        self._sync_background_controls()
        self.statusBar().showMessage("Discarded unapplied reference-mask edits.")

    def _copy_applied_to_draft(self, key: str) -> None:
        for applied, draft in (
            (
                self._applied_background_reference_masks,
                self._draft_background_reference_masks,
            ),
            (
                self._applied_foreground_reference_masks,
                self._draft_foreground_reference_masks,
            ),
        ):
            mask = applied.get(key)
            if mask is None:
                draft.pop(key, None)
            else:
                draft[key] = mask.copy()

    def _empty_current_image_mask(self) -> np.ndarray:
        image_size = self.image_view.image_size
        if image_size is None:
            return np.zeros((0, 0), dtype=bool)
        width, height = image_size
        return np.zeros((height, width), dtype=bool)

    @Slot()
    def _apply_reference_masks(self) -> None:
        key = self._current_image_key()
        if key is None or key not in self._reference_masks_dirty:
            return
        self._stop_reference_point_editing()
        for draft, applied in (
            (
                self._draft_background_reference_masks,
                self._applied_background_reference_masks,
            ),
            (
                self._draft_foreground_reference_masks,
                self._applied_foreground_reference_masks,
            ),
        ):
            mask = draft.get(key)
            if mask is None or not np.any(mask):
                applied.pop(key, None)
            else:
                applied[key] = mask.copy()
        self._reference_masks_dirty.discard(key)
        affected = {
            "foreground_segmentation",
            *self.pipeline.downstream("foreground_segmentation", recursive=True),
            "background_likelihood",
            *self.pipeline.downstream("background_likelihood", recursive=True),
        }
        self.pipeline.invalidate(affected)
        background_count = self._mask_pixel_count(
            self._applied_background_reference_masks.get(key)
        )
        foreground_count = self._mask_pixel_count(
            self._applied_foreground_reference_masks.get(key)
        )
        detail = (
            f"{background_count:,} confirmed reference pixels; updating"
            if background_count
            else "Automatic selection; updating"
        )
        self.pipeline.set_status("background_likelihood", NodeStatus.WARNING, detail)
        self.pipeline.set_status(
            "foreground_segmentation",
            NodeStatus.WARNING,
            f"{foreground_count:,} confirmed reference pixels; updating",
        )
        self.pipeline.set_status(
            "refined_background_likelihood", NodeStatus.IDLE, "Updating"
        )
        self.pipeline.set_status("instance_masks", NodeStatus.IDLE, "Updating")
        self.pipeline.set_status(
            "measurements", NodeStatus.BLOCKED, "Requires reviewed masks"
        )
        self.pipeline_canvas.refresh(affected)
        self.pipeline_inspector.refresh_status()
        self._cache_dirty_nodes.setdefault(key, set()).update(affected)
        self._analyses.pop(key, None)
        self._sync_background_controls()
        self._analyze_current_image()

    @Slot(int)
    def _overlay_changed(self, index: int) -> None:
        del index
        mode = str(self.overlay_combo.currentData())
        self.image_view.set_overlay_mode(mode)
        self._update_overlay_legend(mode)
        owner = _overlay_node_owner(mode)
        if owner is None:
            self.overlay_owner_label.setText("No graph node")
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

    @Slot(int)
    def _overlay_opacity_changed(self, value: int) -> None:
        self.overlay_opacity_label.setText(f"{value}%")
        self.image_view.set_overlay_opacity(value / 100.0)

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
            "seed_scale_estimation": "Cyan: isolated-reference search region. Yellow: the image-specific master seed diameter.",
            "foreground_feature": "Brightness is Lab colour distance from the estimated dish background before thresholding.",
            "foreground_mask": "Soft foreground probability: black is tray, rim, or dark inter-seed gap; white is seed-surface evidence. The binary proposal mask is available separately.",
            "foreground_binary_mask": "Binary, morphologically cleaned foreground supplied to the distance-transform proposal branch.",
            "distance_transform": "Brightness is distance from the nearest foreground boundary; local maxima can become seed centres.",
            "distance_candidates": "Yellow circles are the centres/radii proposed by distance-transform peaks before fusion.",
            "circle_candidates": "Yellow circles are CUDA multiradius ring-support proposals before fusion with distance peaks.",
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
            "refined_background_likelihood": (
                "Noise-frequency likelihood: dark = background-like local "
                "texture; light = non-background-like texture. Fine, medium, "
                "and coarse profiles are learned per image from confident "
                "pixels in the colour layer, using equal class priors. The same "
                "learned texture classifier is also displayed across the initial "
                "dish-surrounding sampling annulus."
            ),
            "edge_gradients": (
                "Brightness is the shared CUDA multichannel edge magnitude. "
                "The same cached continuous tangent field feeds both direction encoders."
            ),
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
            "seed_interior_probability": "Soft seed-interior evidence from colour-derived foreground and inverse background support.",
            "boundary_confidence": "Hue is the continuous directed boundary normal (0° = 360°); brightness is boundary confidence.",
            "boundary_magnitude": "Boundary confidence without direction encoding.",
            "touching_split_likelihood": "High values mark shallow foreground necks and concave distance-transform saddles that may separate touching seeds.",
            "ellipse_likelihood": "Hue is axial ellipse orientation (0° = 180°); brightness is seed-radius boundary and structure-tensor support.",
            "proposal_disagreement": "Variation among CUDA ring, distance-peak, interior, and ellipse evidence; bright regions merit review.",
            "instance_assignment_confidence": "Confidence that a provisional instance owns each pixel after boundary and label-contact penalties.",
            "contested_pixels": "Pixels bordering two different provisional instance labels.",
            "contact_graph": "Shared-boundary likelihood plus vector links between proposals close enough to touch or overlap.",
            "illumination_field": "Estimated broad illumination component; shown as grayscale intensity.",
            "shadow_likelihood": "Pixels substantially darker than the local illumination field.",
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
        if mode.startswith("directional_background:"):
            index = int(mode.partition(":")[2])
            result = self.image_view._analysis_result
            angles = result.layers.directional_background_angles_degrees
            angle = angles[index] if index < len(angles) else 0.0
            text = (
                f"Texture-background continuation along the one-sided {angle:g}° "
                "ray. Dark is high background likelihood; light is low."
            )
        elif mode.startswith("colour_probability:"):
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
            f"{result.circle_candidate_count:,} CUDA ring(s) — Hough circle replacement",
        )
        self.pipeline.set_status(
            "identification",
            NodeStatus.COMPLETE,
            f"{result.count:,} approximate proposals",
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
                f"band {sampling_band.sample_count:,} px"
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
                f"{len(result.layers.directional_background_likelihoods)} rays; "
                f"three-band separation {profile.separation:.2f}; surrounding annulus shown",
            )
        self.pipeline.set_status(
            "edge_gradients",
            NodeStatus.COMPLETE,
            "Shared Lab/Scharr magnitude and continuous tangent field cached",
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
            "seed_interior": "Foreground/background probability fused",
            "boundary_normals": "Boundary magnitude and directed normals ready",
            "touching_split": "Neck and distance-saddle evidence ready",
            "ellipse_likelihood": "Multiscale orientation and radial support ready",
            "proposal_disagreement": "Four proposal evidence sources compared",
            "assignment_confidence": "Assignment and contested-pixel maps ready",
            "contact_graph": f"{len(result.advanced.contact_pairs):,} candidate contact link(s)",
            "illumination_decomposition": "Illumination, reflectance, shadow, and glare ready",
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
        self.pipeline_canvas.refresh()
        self.pipeline_inspector.refresh_status()

    def _pipeline_image_loaded(self, path: Path) -> None:
        for node in self.pipeline.nodes.values():
            node.calculation_seconds = None
        self.pipeline.set_status("raw_images", NodeStatus.COMPLETE, path.name)
        self.pipeline.set_status(
            "metadata", NodeStatus.COMPLETE, self.species_combo.currentText()
        )
        for node_id in (*CALIBRATION_NODE_IDS, "layout_detection"):
            self.pipeline.set_status(node_id, NodeStatus.IDLE, "Ready")
        for node_id in IDENTIFICATION_STAGE_NODE_IDS[:-1]:
            self.pipeline.set_status(node_id, NodeStatus.IDLE, "Ready")
        identification = self.pipeline.node("identification")
        self.pipeline.set_status(
            "identification",
            NodeStatus.IDLE if identification.enabled else NodeStatus.BYPASSED,
            "Ready" if identification.enabled else "Bypassed",
        )
        for node_id in (
            "background_likelihood",
            "refined_background_likelihood",
            "edge_gradients",
            "undirected_edges",
            "directed_edges",
            "edge_ridges",
            "edge_traces",
        ):
            self.pipeline.set_status(node_id, NodeStatus.IDLE, "Ready")
        self.pipeline.set_status("seed_edge_curves", NodeStatus.IDLE, "Ready")
        for node_id in ADVANCED_NODE_MODES:
            self.pipeline.set_status(node_id, NodeStatus.IDLE, "Ready")
        self.pipeline.set_status(
            "instance_masks",
            NodeStatus.IDLE if identification.enabled else NodeStatus.BLOCKED,
            "Ready" if identification.enabled else "Requires seed proposals",
        )
        background = self.pipeline.node("background_likelihood")
        if identification.enabled and not background.enabled:
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
            "foreground_segmentation": "Estimating Lab background and thresholding",
            "distance_candidates": "Finding distance-transform peaks",
            "circle_candidates": "Running CUDA multiradius ring detection",
            "identification": "Fusing candidate centres",
            "instance_masks": "Separating provisional instances",
            "edge_gradients": "Computing shared Lab/Scharr gradient field",
            "undirected_edges": "Encoding axial edge tangents",
            "directed_edges": "Resolving edge polarity",
            "edge_ridges": "Thinning float edges and reconstructing hysteresis",
            "edge_traces": "Linking orientation-compatible ridge fragments",
            "seed_edge_curves": "Confirming radii, circles, ellipses, centres, and semantic sides",
            "background_likelihood": "Estimating the likely colour range",
            "refined_background_likelihood": "Evaluating directed texture rays",
            "seed_interior": "Fusing seed-interior evidence on the tensor device",
            "boundary_normals": "Estimating boundary confidence and normals",
            "touching_split": "Scoring necks and distance saddles",
            "ellipse_likelihood": "Testing multiscale elliptical support",
            "proposal_disagreement": "Comparing independent proposal evidence",
            "assignment_confidence": "Scoring provisional pixel ownership",
            "contact_graph": "Resolving touching and overlapping proposals",
            "illumination_decomposition": "Separating illumination and reflectance",
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
            if node_id == "background_likelihood" and not self.pipeline.node(
                node_id
            ).enabled:
                self.pipeline.set_status(node_id, NodeStatus.BYPASSED, "Disabled")
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

    @Slot(str, str, object)
    def _pipeline_parameter_changed(self, node_id: str, key: str, value) -> None:
        try:
            self._validate_settings_override(node_id, key, value)
            affected = self.pipeline.set_parameter(node_id, key, value)
        except (KeyError, TypeError, ValueError) as error:
            QMessageBox.warning(self, "Invalid pipeline parameter", str(error))
            self.pipeline_inspector.set_node(self.pipeline.node(node_id))
            return
        if not affected:
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

    def _validate_settings_override(
        self, node_id: str, key: str, value: object
    ) -> None:
        baseline_nodes = {
            "seed_scale_estimation",
            "foreground_segmentation",
            "distance_candidates",
            "circle_candidates",
            "identification",
        }
        layer_nodes = {
            "background_likelihood",
            "refined_background_likelihood",
            "edge_gradients",
            "edge_ridges",
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

    @Slot(str, bool)
    def _pipeline_enabled_changed(self, node_id: str, enabled: bool) -> None:
        affected = self.pipeline.set_enabled(node_id, enabled)
        if not affected:
            return
        affected_set = set(affected)
        for image_key in set(self._analysis_caches) | set(self._analyses):
            self._cache_dirty_nodes.setdefault(image_key, set()).update(
                affected_set
            )
            self._analyses.pop(image_key, None)
        if node_id == "identification" and not enabled:
            self.pipeline.set_status(
                "instance_masks", NodeStatus.BLOCKED, "Requires seed identification"
            )
            self.pipeline.set_status(
                "seed_edge_curves", NodeStatus.IDLE, "Ready from seed scale and edges"
            )
            for overlay_node_id in ("undirected_edges", "directed_edges"):
                self.pipeline.set_status(
                    overlay_node_id,
                    NodeStatus.IDLE,
                    "Ready from corrected image",
                )
            if self.pipeline.node("background_likelihood").enabled:
                self.pipeline.set_status(
                    "background_likelihood",
                    NodeStatus.IDLE,
                    "Ready from corrected image",
                )
                self.pipeline.set_status(
                    "refined_background_likelihood",
                    NodeStatus.IDLE,
                    "Ready from corrected image",
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
        elif node_id == "background_likelihood":
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
                self.pipeline.set_status(
                    "instance_masks", NodeStatus.IDLE, "Ready without background map"
                )
        self.pipeline_canvas.refresh(affected)
        self.pipeline_inspector.set_node(self.pipeline.node(node_id))
        current_key = self._current_image_key()
        if (
            current_key is not None
            and current_key in self._analysis_caches
            and self.pipeline.node("identification").enabled
        ):
            self._analyze_current_image(dirty_nodes=affected_set)
        self._update_analysis_availability()
        self._sync_background_controls()

    @Slot(str)
    def _species_changed(self, species: str) -> None:
        if self.image_view.image_path is None:
            return
        self.pipeline.set_status("metadata", NodeStatus.COMPLETE, species)
        self.pipeline_canvas.refresh(("metadata",))

    def _update_analysis_availability(self) -> None:
        path = self.image_view.image_path
        running = False
        if path is not None:
            running = str(path.resolve()).casefold() in self._active_tasks
        available = (
            path is not None
            and self.pipeline.node("identification").enabled
            and not running
        )
        self.analyze_button.setEnabled(available)
        self.analyze_action.setEnabled(available)

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
            "Seed Vision runtime",
            f"Seed Vision {__version__}\n"
            f"Application folder: {self._root}\n\n"
            "Run seed_vision.py --diagnostics for package and GPU details.",
        )


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
