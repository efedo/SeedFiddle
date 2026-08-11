"""Zoomable image canvas used for image review and future mask editing."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QLineF, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QDragEnterEvent,
    QDropEvent,
    QImage,
    QImageReader,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
    QTransform,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QToolButton,
)

from seedvision.ui.canvas_controls import style_canvas_control_bar
from seedvision.visualization import ADVANCED_OVERLAY_LABELS
from seedvision.annotation import (
    EdgeTraceOptions,
    ShapeSnapOptions,
    SmartFillOptions,
    SmartFillRegion,
    smart_fill_instance,
    smart_fill_region,
    snap_edge_point,
    snap_shape_polygon,
    trace_edge_path,
)


SUPPORTED_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
OVERLAY_MODES = {
    "raw_image",
    "deskew_colour",
    "calibrated_image",
    "colour_reference",
    "ruler_detection",
    "layout_detection",
    "perimeter_background_reference",
    "seed_scale_estimation",
    "foreground_feature",
    "foreground_mask",
    "foreground_binary_mask",
    "distance_transform",
    "distance_candidates",
    "circle_candidates",
    "proposals",
    "instance_masks",
    "background_likelihood",
    "refined_background_likelihood",
    "foreground_noise_likelihood",
    "edge_gradients",
    "surface_lightening_gradient",
    "surface_lightening_magnitude",
    "surface_darkening_gradient",
    "surface_darkening_magnitude",
    "weak_lightening_gradient",
    "weak_lightening_magnitude",
    "weak_darkening_gradient",
    "weak_darkening_magnitude",
    "darkness_noise_fine",
    "darkness_noise_medium",
    "darkness_noise_coarse",
    "colour_noise_fine",
    "colour_noise_medium",
    "colour_noise_coarse",
    "undirected_edges",
    "directed_edges",
    "edge_ridges",
    "edge_traces",
    "edge_trace_continuity",
    "edge_trace_gap_confidence",
    "edge_radius_confirmation",
    "edge_circle_fit",
    "edge_ellipse_fit",
    "edge_fit_residual",
    "edge_centre_votes",
    "edge_semantic_sides",
    "edge_rejections",
    "edge_fit_geometry",
    "seed_edge_curves",
    "none",
    *(mode for _, mode in ADVANCED_OVERLAY_LABELS),
}


class ImageView(QGraphicsView):
    """Display a laboratory image with smooth pan, zoom, fit, and file dropping."""

    image_dropped = Signal(str)
    reference_paint_finished = Signal()
    reference_mask_edited = Signal(str, object)
    instance_annotations_edited = Signal(object)
    instance_tool_status = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._image_item = None
        self._image_path: Path | None = None
        self._source_image: QImage | None = None
        self._overlay_items = []
        self._analysis_result = None
        self._source_image = None
        self._overlay_mode = "proposals"
        self._overlay_opacity = 0.68
        self._zoom_steps = 0
        self._fit_pending = False
        self._background_reference_mask: np.ndarray | None = None
        self._foreground_reference_mask: np.ndarray | None = None
        self._background_exclusion_mask: np.ndarray | None = None
        self._foreground_exclusion_mask: np.ndarray | None = None
        self._reference_annotations_visible = True
        self._instance_annotations: np.ndarray | None = None
        self._active_instance_id = 1
        self._instance_annotation_tool = "brush"
        self._edge_trace_options = EdgeTraceOptions()
        self._shape_snap_options = ShapeSnapOptions()
        self._smart_fill_options = SmartFillOptions()
        self._instance_tool_points: list[QPointF] = []
        self._annotation_evidence_cache: dict[str, np.ndarray] = {}
        self._instance_trace_anchor: QPointF | None = None
        self._instance_preview_geometry: np.ndarray | None = None
        self._instance_shape_reference_geometry: np.ndarray | None = None
        self._instance_preview_region: SmartFillRegion | None = None
        self._instance_preview_endpoint: QPointF | None = None
        self._instance_preview_point: QPointF | None = None
        self._pending_instance_preview_point: QPointF | None = None
        self._instance_preview_items = []
        self._instance_annotation_overlay_item = None
        self._instance_live_stroke_path: QPainterPath | None = None
        self._instance_live_stroke_item = None
        self._instance_preview_timer = QTimer(self)
        self._instance_preview_timer.setSingleShot(True)
        self._instance_preview_timer.setInterval(35)
        self._instance_preview_timer.timeout.connect(
            self._update_pending_instance_preview
        )
        self._reference_point_mode: str | None = None
        self._reference_brush_radius = 12.0
        self._reference_erase_enabled = False
        self._reference_paint_button: Qt.MouseButton | None = None
        self._reference_stroke_erases = False
        self._last_reference_paint_point: QPointF | None = None
        self._last_reference_hover_point: QPointF | None = None
        self._reference_brush_outline_item = None
        self._context_panel: QFrame | None = None

        self.setAcceptDrops(True)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.setBackgroundBrush(QColor("#20252b"))
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setViewportMargins(0, 38, 0, 0)
        self._build_zoom_controls()
        self._show_placeholder()

    def _build_zoom_controls(self) -> None:
        self.zoom_controls = QFrame(self)
        controls = QHBoxLayout(self.zoom_controls)
        controls.setContentsMargins(8, 4, 8, 4)
        controls.setSpacing(5)
        heading = QLabel("Image zoom", self.zoom_controls)
        self.zoom_out_button = QToolButton(self.zoom_controls)
        self.zoom_out_button.setText("−")
        self.zoom_out_button.setToolTip("Zoom out")
        self.zoom_out_button.clicked.connect(self.zoom_out)
        self.zoom_label = QLabel("100%", self.zoom_controls)
        self.zoom_label.setMinimumWidth(48)
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.zoom_in_button = QToolButton(self.zoom_controls)
        self.zoom_in_button.setText("+")
        self.zoom_in_button.setToolTip("Zoom in")
        self.zoom_in_button.clicked.connect(self.zoom_in)
        self.fit_button = QToolButton(self.zoom_controls)
        self.fit_button.setText("Fit")
        self.fit_button.clicked.connect(self.fit_image)
        self.actual_size_button = QToolButton(self.zoom_controls)
        self.actual_size_button.setText("100%")
        self.actual_size_button.setToolTip("Show one image pixel per screen pixel")
        self.actual_size_button.clicked.connect(self.actual_size)
        controls.addWidget(heading)
        controls.addStretch(1)
        controls.addWidget(self.zoom_out_button)
        controls.addWidget(self.zoom_label)
        controls.addWidget(self.zoom_in_button)
        controls.addWidget(self.fit_button)
        controls.addWidget(self.actual_size_button)
        style_canvas_control_bar(self.zoom_controls, "imageZoomControls")
        self.zoom_controls.raise_()

    def set_context_panel(self, panel: QFrame) -> None:
        """Attach a contextual control panel above the image viewport."""

        panel.setParent(self)
        self._context_panel = panel
        self._layout_context_panel()
        self.zoom_controls.raise_()

    def set_context_panel_visible(self, visible: bool) -> None:
        if self._context_panel is None:
            return
        self._context_panel.setVisible(visible)
        if visible:
            self._layout_context_panel()
            self._context_panel.raise_()
            self.zoom_controls.raise_()

    def _layout_context_panel(self) -> None:
        if self._context_panel is None:
            return
        hint = self._context_panel.sizeHint()
        available_width = max(240, self.width() - 24)
        available_height = max(140, self.height() - 58)
        width = min(max(360, hint.width()), available_width)
        height = min(hint.height(), available_height)
        self._context_panel.setGeometry(12, 46, width, height)

    @property
    def image_path(self) -> Path | None:
        return self._image_path

    @property
    def image_size(self) -> tuple[int, int] | None:
        if self._image_item is None:
            return None
        pixmap = self._image_item.pixmap()
        return pixmap.width(), pixmap.height()

    def _show_placeholder(self) -> None:
        self._scene.clear()
        self._image_item = None
        self._overlay_items = []
        self._analysis_result = None
        self._reference_brush_outline_item = None
        self._instance_annotation_overlay_item = None
        self._instance_preview_items = []
        self._instance_live_stroke_path = None
        self._instance_live_stroke_item = None
        message = QGraphicsTextItem(
            "Open or drop a calibrated laboratory image\n"
            "Supported formats: PNG, JPEG, TIFF, BMP"
        )
        message.setDefaultTextColor(QColor("#d8dee9"))
        message.setTextWidth(420)
        self._scene.addItem(message)
        bounds = message.boundingRect()
        message.setPos(-bounds.width() / 2, -bounds.height() / 2)
        self._scene.setSceneRect(QRectF(-300, -180, 600, 360))

    def load_image(self, path: Path) -> tuple[bool, str]:
        reader = QImageReader(str(path))
        reader.setAutoTransform(True)
        image = reader.read()
        if image.isNull():
            return False, reader.errorString() or "Qt could not decode the image."

        pixmap = QPixmap.fromImage(image)
        self._source_image = image.copy()
        self._scene.clear()
        self._overlay_items = []
        self._analysis_result = None
        self._background_reference_mask = None
        self._foreground_reference_mask = None
        self._background_exclusion_mask = None
        self._foreground_exclusion_mask = None
        self._instance_annotations = None
        self._instance_tool_points.clear()
        self._annotation_evidence_cache.clear()
        self._reference_brush_outline_item = None
        self._instance_annotation_overlay_item = None
        self._instance_preview_items = []
        self._instance_live_stroke_path = None
        self._instance_live_stroke_item = None
        self._instance_trace_anchor = None
        self._instance_preview_geometry = None
        self._instance_shape_reference_geometry = None
        self._instance_preview_region = None
        self._image_item = self._scene.addPixmap(pixmap)
        self._image_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self._scene.setSceneRect(self._image_item.boundingRect())
        self._image_path = path
        if self.isVisible():
            self.fit_image()
        else:
            self._fit_pending = True
        return True, ""

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().showEvent(event)
        if self._fit_pending:
            QTimer.singleShot(0, self._finish_pending_fit)

    def _finish_pending_fit(self) -> None:
        if not self._fit_pending:
            return
        self._fit_pending = False
        self.fit_image()

    def clear_analysis(self) -> None:
        """Remove analysis graphics without disturbing the loaded image."""

        self._clear_overlay_items()
        self._analysis_result = None
        self._annotation_evidence_cache.clear()
        self._restore_source_image()

    def _clear_overlay_items(self) -> None:
        for item in self._overlay_items:
            self._scene.removeItem(item)
        self._overlay_items.clear()
        self._instance_annotation_overlay_item = None

    def show_analysis(self, result) -> None:
        """Store an analysis result and render the selected overlay layer."""

        self._analysis_result = result
        self._annotation_evidence_cache.clear()
        calibration = getattr(result, "calibration", None)
        if calibration is not None:
            self._set_bgr_base_image(calibration.corrected_bgr)
        self._render_analysis()

    def set_background_point_editing(self, enabled: bool) -> None:
        self._set_reference_point_mode("background" if enabled else None)

    def set_reference_masks(
        self,
        background_mask: np.ndarray | None,
        foreground_mask: np.ndarray | None,
        background_exclusion_mask: np.ndarray | None = None,
        foreground_exclusion_mask: np.ndarray | None = None,
    ) -> None:
        """Display editable full-resolution binary masks in image coordinates."""

        self._background_reference_mask = self._normalized_reference_mask(
            background_mask
        )
        self._foreground_reference_mask = self._normalized_reference_mask(
            foreground_mask
        )
        self._background_exclusion_mask = self._normalized_reference_mask(
            background_exclusion_mask
        )
        self._foreground_exclusion_mask = self._normalized_reference_mask(
            foreground_exclusion_mask
        )
        self._render_analysis()

    def reference_mask(self, class_name: str) -> np.ndarray | None:
        masks = {
            "background": self._background_reference_mask,
            "foreground": self._foreground_reference_mask,
            "background_exclusion": self._background_exclusion_mask,
            "foreground_exclusion": self._foreground_exclusion_mask,
        }
        if class_name not in masks:
            raise ValueError(f"Unknown reference-mask class {class_name!r}.")
        mask = masks[class_name]
        return None if mask is None else mask.copy()

    def set_reference_annotations_visible(self, visible: bool) -> None:
        self._reference_annotations_visible = bool(visible)
        self._render_analysis()

    def _normalized_reference_mask(
        self, mask: np.ndarray | None
    ) -> np.ndarray | None:
        if mask is None:
            return None
        image_size = self.image_size
        if image_size is None:
            return np.asarray(mask, dtype=bool).copy()
        width, height = image_size
        values = np.asarray(mask, dtype=bool)
        if values.shape != (height, width):
            raise ValueError(
                f"Reference mask shape {values.shape} does not match image {(height, width)}."
            )
        return values.copy()

    def set_foreground_point_editing(self, enabled: bool) -> None:
        self._set_reference_point_mode("foreground" if enabled else None)

    def set_background_exclusion_editing(self, enabled: bool) -> None:
        self._set_reference_point_mode("background_exclusion" if enabled else None)

    def set_foreground_exclusion_editing(self, enabled: bool) -> None:
        self._set_reference_point_mode("foreground_exclusion" if enabled else None)

    def set_instance_annotations(self, annotations: np.ndarray | None) -> None:
        """Display an editable full-resolution seed-identity label map."""

        if annotations is None:
            self._instance_annotations = None
        else:
            values = np.asarray(annotations)
            if values.ndim != 2:
                raise ValueError("Instance annotations must be a two-dimensional label map.")
            if np.any(values < 0) or np.any(values > np.iinfo(np.uint16).max):
                raise ValueError("Instance annotation IDs must fit in an unsigned 16-bit label map.")
            image_size = self.image_size
            if image_size is not None:
                width, height = image_size
                if values.shape != (height, width):
                    raise ValueError(
                        f"Instance annotation shape {values.shape} does not match "
                        f"image {(height, width)}."
                    )
            self._instance_annotations = values.astype(np.uint16, copy=True)
        self._instance_trace_anchor = None
        self._clear_instance_preview()
        self._clear_instance_live_stroke()
        self._render_analysis()
        self._schedule_instance_preview(self._last_reference_hover_point)

    def instance_annotations(self) -> np.ndarray | None:
        labels = self._instance_annotations
        return None if labels is None else labels.copy()

    def set_instance_annotation_editing(self, enabled: bool) -> None:
        self._set_reference_point_mode("instance" if enabled else None)

    def set_instance_annotation_tool(self, tool: str) -> None:
        """Choose the brush, eraser, or image-aware instance tool."""

        if tool not in {"brush", "edge_trace", "shape_snap", "smart_fill", "eraser"}:
            raise ValueError(f"Unknown instance annotation tool {tool!r}.")
        self._instance_annotation_tool = tool
        self._instance_tool_points.clear()
        self._instance_trace_anchor = None
        self._clear_instance_preview()
        self._clear_instance_live_stroke()
        self.set_reference_erase_mode(tool == "eraser")
        self._schedule_instance_preview(self._last_reference_hover_point)

    def set_edge_trace_options(self, options: EdgeTraceOptions) -> None:
        self._edge_trace_options = options
        self._schedule_instance_preview(self._last_reference_hover_point)

    def set_shape_snap_options(self, options: ShapeSnapOptions) -> None:
        self._shape_snap_options = options
        self._schedule_instance_preview(self._last_reference_hover_point)

    def set_smart_fill_options(self, options: SmartFillOptions) -> None:
        self._smart_fill_options = options
        self._schedule_instance_preview(self._last_reference_hover_point)

    def set_active_instance_id(self, identifier: int) -> None:
        identifier = int(identifier)
        if not 1 <= identifier <= np.iinfo(np.uint16).max:
            raise ValueError("Seed instance IDs must be between 1 and 65,535.")
        self._active_instance_id = identifier
        self._instance_trace_anchor = None
        self._clear_instance_preview()
        self._clear_instance_live_stroke()
        if self._last_reference_hover_point is not None:
            self._update_reference_brush_outline(self._last_reference_hover_point)

    def set_reference_brush_radius(self, radius: float) -> None:
        """Set the corrected-image radius of the visibly painted sample area."""

        self._reference_brush_radius = max(2.0, float(radius))
        if self._last_reference_hover_point is not None:
            self._update_reference_brush_outline(
                self._last_reference_hover_point
            )
        # Changing a seed-annotation brush must not rebuild every analysis
        # overlay.  The cursor and the next committed stroke are sufficient.
        if self._reference_point_mode != "instance":
            self._render_analysis()

    def set_reference_erase_mode(self, enabled: bool) -> None:
        """Choose whether left-button reference strokes paint or erase."""

        self._reference_erase_enabled = bool(enabled)
        if self._last_reference_hover_point is not None:
            self._update_reference_brush_outline(
                self._last_reference_hover_point
            )

    def _set_reference_point_mode(self, mode: str | None) -> None:
        self._reference_point_mode = mode
        if mode is not None:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.viewport().setCursor(Qt.CursorShape.CrossCursor)
        else:
            self._reference_paint_button = None
            self._reference_stroke_erases = False
            self._last_reference_paint_point = None
            self._last_reference_hover_point = None
            self._hide_reference_brush_outline()
            self._instance_trace_anchor = None
            self._clear_instance_preview()
            self._clear_instance_live_stroke()
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.viewport().unsetCursor()

    def _update_reference_brush_outline(self, scene_point: QPointF) -> None:
        if self._reference_point_mode is None or self._image_item is None:
            self._hide_reference_brush_outline()
            return
        if not self._image_item.boundingRect().contains(scene_point):
            self._last_reference_hover_point = None
            self._hide_reference_brush_outline()
            return
        self._last_reference_hover_point = QPointF(scene_point)
        if self._instance_assisted_tool_active():
            self._hide_reference_brush_outline()
            self._schedule_instance_preview(scene_point)
            return
        radius = self._reference_brush_radius
        if self._reference_brush_outline_item is None:
            self._reference_brush_outline_item = self._scene.addEllipse(
                scene_point.x() - radius,
                scene_point.y() - radius,
                radius * 2.0,
                radius * 2.0,
            )
            self._reference_brush_outline_item.setZValue(1000.0)
            self._reference_brush_outline_item.setBrush(Qt.BrushStyle.NoBrush)
        else:
            self._reference_brush_outline_item.setRect(
                scene_point.x() - radius,
                scene_point.y() - radius,
                radius * 2.0,
                radius * 2.0,
            )
        erasing = (
            self._reference_stroke_erases
            if self._reference_paint_button is not None
            else self._reference_erase_enabled
        )
        colour = (
            QColor("#ff5f69")
            if erasing
            else QColor("#36f1b1")
            if self._reference_point_mode == "background"
            else QColor("#ff765f")
            if self._reference_point_mode == "foreground"
            else QColor("#be5cff")
            if self._reference_point_mode == "background_exclusion"
            else QColor("#ffce48")
            if self._reference_point_mode == "foreground_exclusion"
            else self.instance_colour(self._active_instance_id)
        )
        pen = QPen(colour, 2.0)
        pen.setCosmetic(True)
        pen.setStyle(
            Qt.PenStyle.DashLine if erasing else Qt.PenStyle.SolidLine
        )
        self._reference_brush_outline_item.setPen(pen)
        self._reference_brush_outline_item.show()

    def _hide_reference_brush_outline(self) -> None:
        if self._reference_brush_outline_item is not None:
            self._reference_brush_outline_item.hide()

    def set_overlay_mode(self, mode: str) -> None:
        if mode not in OVERLAY_MODES and not mode.startswith((
            "directional_background:",
            "colour_probability:",
            "pattern_probability:",
        )):
            raise ValueError(f"Unknown viewer overlay: {mode}")
        if mode == self._overlay_mode:
            return
        self._overlay_mode = mode
        self._render_analysis()

    def set_overlay_opacity(self, opacity: float) -> None:
        self._overlay_opacity = max(0.0, min(1.0, float(opacity)))
        for item in self._overlay_items:
            item.setOpacity(self._overlay_opacity)

    def _render_analysis(self) -> None:
        self._clear_overlay_items()
        result = self._analysis_result
        if result is None:
            return
        if self._overlay_mode == "raw_image":
            self._restore_source_image()
            return
        calibration = getattr(result, "calibration", None)
        if calibration is not None:
            self._set_bgr_base_image(calibration.corrected_bgr)
        if self._overlay_mode == "calibrated_image":
            self._render_context_annotations(result, include_scale=True)
            return
        if self._overlay_mode == "none":
            self._render_context_annotations(result)
            return
        if self._overlay_mode == "colour_reference":
            self._restore_source_image()
            self._render_colour_reference(result, corrected=False)
            return
        if self._overlay_mode == "deskew_colour":
            self._render_colour_reference(result, corrected=True)
            self._render_context_annotations(result)
            return
        if self._overlay_mode == "ruler_detection":
            self._render_ruler_reference(result)
            self._render_context_annotations(result)
            return
        if self._overlay_mode == "layout_detection":
            self._render_dish_edges(result, width=4, include_inner=True)
            self._render_context_annotations(result)
            return
        if self._overlay_mode == "perimeter_background_reference":
            self._render_dish_edges(result, width=3, include_inner=False)
            self._render_background_sampling_band(result)
            self._render_context_annotations(result)
            return
        if self._overlay_mode == "seed_scale_estimation":
            self._render_seed_scale(result)
            self._render_context_annotations(result)
            return
        if self._overlay_mode == "proposals":
            self._render_proposals(result)
            self._render_context_annotations(result)
            return

        if self._overlay_mode in {
            "foreground_feature",
            "foreground_mask",
            "foreground_binary_mask",
            "distance_transform",
        }:
            if self._overlay_mode == "foreground_mask":
                raster = result.foreground_probability
            elif self._overlay_mode == "foreground_binary_mask":
                raster = result.foreground_mask
            else:
                raster = getattr(result, self._overlay_mode)
            self._render_scalar_raster(
                raster,
                *result.crop_offset,
                valid_mask=result.layers.valid_mask,
            )
            self._render_context_annotations(result)
            return
        if self._overlay_mode in {"distance_candidates", "circle_candidates"}:
            geometry = getattr(result, f"{self._overlay_mode[:-1]}_geometry")
            self._render_candidate_geometry(result, geometry)
            self._render_context_annotations(result)
            return

        advanced = getattr(result, "advanced", None)
        if advanced is not None and (
            self._overlay_mode in advanced.rasters
            or self._overlay_mode in advanced.hue_rasters
            or self._overlay_mode.startswith((
                "colour_probability:", "pattern_probability:"
            ))
        ):
            self._render_rgba_overlay(
                advanced.rgba(self._overlay_mode),
                advanced.offset_x,
                advanced.offset_y,
            )
            if self._overlay_mode == "contact_graph":
                self._render_contact_graph(result)
            self._render_context_annotations(result)
            return

        layers = result.layers
        if (
            layers.background_mode == "disabled"
            and self._overlay_mode
            in {"background_likelihood", "refined_background_likelihood"}
        ):
            self._render_context_annotations(result)
            return
        if self._overlay_mode.startswith("directional_background:"):
            index = int(self._overlay_mode.partition(":")[2])
            rgba = layers.directional_background_rgba(index)
        elif self._overlay_mode == "instance_masks":
            rgba = layers.instance_rgba()
        elif self._overlay_mode == "background_likelihood":
            rgba = layers.background_rgba()
        elif self._overlay_mode == "refined_background_likelihood":
            rgba = layers.refined_background_rgba()
        elif self._overlay_mode == "foreground_noise_likelihood":
            rgba = layers.foreground_noise_rgba()
        elif self._overlay_mode == "edge_gradients":
            rgba = layers.edge_magnitude_rgba()
        elif self._overlay_mode in {
            "surface_lightening_gradient",
            "surface_darkening_gradient",
            "weak_lightening_gradient",
            "weak_darkening_gradient",
        }:
            filtered = self._overlay_mode.startswith("weak_")
            polarity = (
                "lightening"
                if "lightening" in self._overlay_mode
                else "darkening"
            )
            rgba = layers.surface_gradient_rgba(
                polarity, filtered=filtered
            )
        elif self._overlay_mode in {
            "surface_lightening_magnitude",
            "surface_darkening_magnitude",
            "weak_lightening_magnitude",
            "weak_darkening_magnitude",
        }:
            filtered = self._overlay_mode.startswith("weak_")
            polarity = (
                "lightening"
                if "lightening" in self._overlay_mode
                else "darkening"
            )
            rgba = layers.surface_gradient_magnitude_rgba(
                polarity, filtered=filtered
            )
        elif self._overlay_mode in {
            "darkness_noise_fine",
            "darkness_noise_medium",
            "darkness_noise_coarse",
            "colour_noise_fine",
            "colour_noise_medium",
            "colour_noise_coarse",
        }:
            channel, _, band = self._overlay_mode.partition("_noise_")
            band_index = {"fine": 0, "medium": 1, "coarse": 2}[band]
            rgba = layers.frequency_noise_rgba(channel, band_index)
        elif self._overlay_mode == "edge_ridges":
            rgba = layers.edge_ridges_rgba()
        elif self._overlay_mode == "edge_traces":
            rgba = layers.edge_traces_rgba()
        elif self._overlay_mode == "edge_trace_continuity":
            rgba = layers._heat_rgba(layers.edge_trace_continuity)
        elif self._overlay_mode == "edge_trace_gap_confidence":
            rgba = layers._heat_rgba(layers.edge_trace_gap_confidence)
        elif self._overlay_mode == "edge_radius_confirmation":
            rgba = layers.edge_radius_confirmation_rgba()
        elif self._overlay_mode == "edge_circle_fit":
            rgba = layers._heat_rgba(layers.edge_circle_confidence)
        elif self._overlay_mode == "edge_ellipse_fit":
            rgba = layers._heat_rgba(layers.edge_ellipse_confidence)
        elif self._overlay_mode == "edge_fit_residual":
            rgba = layers._heat_rgba(layers.edge_fit_residual)
        elif self._overlay_mode == "edge_centre_votes":
            rgba = layers._heat_rgba(layers.edge_centre_votes)
        elif self._overlay_mode == "edge_semantic_sides":
            rgba = layers._heat_rgba(layers.edge_semantic_sides)
        elif self._overlay_mode == "edge_rejections":
            rgba = layers.edge_rejection_rgba()
        elif self._overlay_mode == "edge_fit_geometry":
            self._render_edge_fit_geometry(result)
            self._render_context_annotations(result)
            return
        elif self._overlay_mode == "undirected_edges":
            rgba = layers.undirected_edge_rgba()
        elif self._overlay_mode == "seed_edge_curves":
            rgba = layers.seed_edge_curve_rgba()
        else:
            rgba = layers.directed_edge_rgba()
        self._render_rgba_overlay(rgba, layers.offset_x, layers.offset_y)
        if self._overlay_mode == "refined_background_likelihood":
            surrounding = layers.surrounding_noise_rgba()
            if surrounding is not None:
                self._render_rgba_overlay(
                    surrounding,
                    layers.surrounding_noise_offset_x,
                    layers.surrounding_noise_offset_y,
                )
        if self._overlay_mode == "background_likelihood":
            self._render_background_sampling_band(result)
        self._render_context_annotations(result)

    def _render_edge_fit_geometry(self, result) -> None:
        geometry = result.layers.edge_fit_geometry
        if geometry is None:
            return
        values = geometry.materialize()
        offset_x, offset_y = result.crop_offset
        vote_pen = QPen(QColor("#ffdc52"), 3)
        vote_pen.setCosmetic(True)
        for (x, y), confidence in zip(
            values["vote_centres_xy"], values["vote_confidence"], strict=True
        ):
            if confidence < 0.20:
                continue
            radius = 3.0 + 6.0 * float(confidence)
            item = self._scene.addEllipse(
                float(x + offset_x - radius),
                float(y + offset_y - radius),
                radius * 2.0,
                radius * 2.0,
                vote_pen,
            )
            item.setOpacity(self._overlay_opacity)
            item.setZValue(20)
            self._overlay_items.append(item)
        ellipse_pen = QPen(QColor("#42e5ff"), 4)
        ellipse_pen.setCosmetic(True)
        for centre, axes, angle, confidence in zip(
            values["instance_centres_xy"],
            values["ellipse_axes_xy"],
            values["ellipse_angle_radians"],
            values["ellipse_confidence"],
            strict=True,
        ):
            if confidence < 0.08:
                continue
            major, minor = float(axes[0]), float(axes[1])
            item = self._scene.addEllipse(
                -major,
                -minor,
                major * 2.0,
                minor * 2.0,
                ellipse_pen,
            )
            item.setPos(float(centre[0] + offset_x), float(centre[1] + offset_y))
            item.setRotation(float(np.rad2deg(angle)))
            item.setOpacity(self._overlay_opacity * min(1.0, float(confidence)))
            item.setZValue(19)
            self._overlay_items.append(item)

    def _render_rgba_overlay(
        self, rgba: np.ndarray, offset_x: int, offset_y: int
    ) -> None:
        height, width = rgba.shape[:2]
        image = QImage(
            rgba.data,
            width,
            height,
            int(rgba.strides[0]),
            QImage.Format.Format_RGBA8888,
        ).copy()
        item = self._scene.addPixmap(QPixmap.fromImage(image))
        item.setPos(offset_x, offset_y)
        item.setOpacity(self._overlay_opacity)
        item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        item.setZValue(10)
        self._overlay_items.append(item)

    def _render_contact_graph(self, result) -> None:
        proposals = {proposal.identifier: proposal for proposal in result.proposals}
        for first_id, second_id, confidence in result.advanced.contact_pairs:
            first = proposals.get(first_id)
            second = proposals.get(second_id)
            if first is None or second is None:
                continue
            colour = QColor.fromHsvF(
                max(0.0, min(0.33, confidence * 0.33)), 0.95, 1.0, 0.92
            )
            pen = QPen(colour, 3.0 + confidence * 4.0)
            pen.setCosmetic(True)
            line = self._scene.addLine(
                first.center_x,
                first.center_y,
                second.center_x,
                second.center_y,
                pen,
            )
            line.setOpacity(self._overlay_opacity)
            line.setZValue(14)
            self._overlay_items.append(line)

    def _render_scalar_raster(
        self,
        raster: np.ndarray,
        offset_x: int,
        offset_y: int,
        *,
        valid_mask: np.ndarray,
    ) -> None:
        values = np.asarray(raster)
        if values.dtype == np.uint8:
            gray = values
        else:
            valid_values = values[np.asarray(valid_mask) > 0]
            scale = float(np.percentile(valid_values, 99.0)) if valid_values.size else 1.0
            gray = np.uint8(np.clip(np.rint(values / max(scale, 1e-6) * 255), 0, 255))
        alpha = np.uint8(np.asarray(valid_mask) > 0) * 255
        rgba = np.dstack((gray, gray, gray, alpha))
        height, width = rgba.shape[:2]
        image = QImage(
            rgba.data, width, height, int(rgba.strides[0]), QImage.Format.Format_RGBA8888
        ).copy()
        item = self._scene.addPixmap(QPixmap.fromImage(image))
        item.setPos(offset_x, offset_y)
        item.setOpacity(self._overlay_opacity)
        item.setZValue(10)
        self._overlay_items.append(item)

    def _render_candidate_geometry(self, result, geometry) -> None:
        self._render_dish_edges(result, width=4)
        offset_x, offset_y = result.crop_offset
        candidate_pen = QPen(QColor("#ffd84a"), 3)
        candidate_pen.setCosmetic(True)
        for x, y, radius in np.asarray(geometry).reshape(-1, 3):
            circle = self._scene.addEllipse(
                float(x + offset_x - radius),
                float(y + offset_y - radius),
                float(radius * 2),
                float(radius * 2),
                candidate_pen,
            )
            circle.setOpacity(self._overlay_opacity)
            circle.setZValue(11)
            self._overlay_items.append(circle)

    def _render_dish_edges(
        self, result, *, width: int = 3, include_inner: bool = False
    ) -> None:
        """Draw the analysis rim, plus the lower rim on the layout node."""

        dish = result.dish
        legacy_radius = int(dish.radius)
        outer_radius = int(getattr(dish, "outer_radius", legacy_radius))
        inner_radius = int(getattr(dish, "inner_radius", legacy_radius))
        edges = [(outer_radius, QColor("#41d9ff"))]
        if include_inner:
            edges.append((inner_radius, QColor("#ff63d8")))
        rendered_radii: set[int] = set()
        for radius, colour in edges:
            radius = int(radius)
            if radius in rendered_radii:
                continue
            rendered_radii.add(radius)
            pen = QPen(colour)
            pen.setWidth(width)
            pen.setCosmetic(True)
            if not getattr(dish, "rim_pair_detected", False):
                pen.setStyle(Qt.PenStyle.DashLine)
            item = self._scene.addEllipse(
                dish.center_x - radius,
                dish.center_y - radius,
                radius * 2,
                radius * 2,
                pen,
            )
            item.setOpacity(self._overlay_opacity)
            item.setZValue(10)
            self._overlay_items.append(item)

    def _render_background_sampling_band(self, result) -> None:
        """Show the exact annulus sampled for the initial background prior."""

        band = getattr(result, "perimeter_background_band", None)
        if band is None:
            return
        inner_radius = max(0.0, float(band.inner_radius_px))
        outer_radius = max(inner_radius, float(band.outer_radius_px))
        if outer_radius <= inner_radius:
            return
        path = QPainterPath()
        path.setFillRule(Qt.FillRule.OddEvenFill)
        path.addEllipse(
            QRectF(
                result.dish.center_x - outer_radius,
                result.dish.center_y - outer_radius,
                outer_radius * 2.0,
                outer_radius * 2.0,
            )
        )
        path.addEllipse(
            QRectF(
                result.dish.center_x - inner_radius,
                result.dish.center_y - inner_radius,
                inner_radius * 2.0,
                inner_radius * 2.0,
            )
        )
        colour = QColor("#41d9ff") if band.outside_vessel else QColor("#ffb84a")
        fill = QColor(colour)
        fill.setAlpha(105)
        pen = QPen(colour, 3)
        pen.setCosmetic(True)
        item = self._scene.addPath(path, pen, QBrush(fill))
        item.setOpacity(self._overlay_opacity)
        item.setZValue(13)
        item.setToolTip(
            "Initial background sampling band: "
            f"{band.buffer_cm:.2f} cm buffer, "
            f"{band.thickness_cm:.2f} cm thickness, "
            f"{band.sample_count:,} pixels; "
            + ("outside upper rim" if band.outside_vessel else "inside-rim fallback")
        )
        self._overlay_items.append(item)

    def _render_seed_scale(self, result) -> None:
        x0, y0, x1, y1 = result.reference_roi
        roi_pen = QPen(QColor("#41d9ff"), 5)
        roi_pen.setCosmetic(True)
        rectangle = self._scene.addRect(x0, y0, x1 - x0, y1 - y0, roi_pen)
        rectangle.setOpacity(self._overlay_opacity)
        rectangle.setZValue(10)
        self._overlay_items.append(rectangle)
        diameter = float(result.estimated_seed_diameter_px)
        centre = QPointF(x0 + diameter * 0.7, y0 + diameter * 0.7)
        scale_pen = QPen(QColor("#ffd84a"), 5)
        scale_pen.setCosmetic(True)
        circle = self._scene.addEllipse(
            centre.x() - diameter * 0.5,
            centre.y() - diameter * 0.5,
            diameter,
            diameter,
            scale_pen,
        )
        circle.setOpacity(self._overlay_opacity)
        circle.setZValue(11)
        self._overlay_items.append(circle)

    def _render_context_annotations(
        self, result, *, include_scale: bool = False
    ) -> None:
        if include_scale:
            self._render_scale_bar(result)
        if self._reference_annotations_visible:
            self._render_background_reference_points()
            self._render_foreground_reference_points()
            self._render_exclusion_masks()
        self._render_instance_annotations()

    def _background_marker_radius(self) -> float:
        return self._reference_brush_radius

    def _render_background_reference_points(self) -> None:
        self._render_reference_mask(
            self._background_reference_mask,
            (54, 241, 177),
            30,
        )

    def _render_foreground_reference_points(self) -> None:
        self._render_reference_mask(
            self._foreground_reference_mask,
            (255, 118, 95),
            31,
        )

    def _render_exclusion_masks(self) -> None:
        self._render_reference_mask(
            self._background_exclusion_mask,
            (190, 92, 255),
            32,
        )
        self._render_reference_mask(
            self._foreground_exclusion_mask,
            (255, 206, 72),
            33,
        )

    @staticmethod
    def instance_colour(identifier: int) -> QColor:
        """Return a stable, high-contrast colour for a positive seed ID."""

        hue = (max(1, int(identifier)) * 0.61803398875) % 1.0
        return QColor.fromHsvF(hue, 0.78, 1.0)

    def _render_instance_annotations(self) -> None:
        if self._instance_annotation_overlay_item is not None:
            item = self._instance_annotation_overlay_item
            if item in self._overlay_items:
                self._overlay_items.remove(item)
            if item.scene() is self._scene:
                self._scene.removeItem(item)
            self._instance_annotation_overlay_item = None
        labels = self._instance_annotations
        if labels is None or not np.any(labels):
            return
        source_height, source_width = labels.shape
        maximum_display_dimension = 2048
        if max(source_height, source_width) > maximum_display_dimension:
            factor = maximum_display_dimension / max(source_height, source_width)
            display_width = max(1, round(source_width * factor))
            display_height = max(1, round(source_height * factor))
            columns = np.minimum(
                (np.arange(display_width) * source_width / display_width).astype(int),
                source_width - 1,
            )
            rows = np.minimum(
                (np.arange(display_height) * source_height / display_height).astype(int),
                source_height - 1,
            )
            display_labels = labels[np.ix_(rows, columns)]
        else:
            display_labels = labels
        rgba = np.zeros((*display_labels.shape, 4), np.uint8)
        for identifier in np.unique(display_labels):
            if identifier == 0:
                continue
            colour = self.instance_colour(int(identifier))
            selected = display_labels == identifier
            rgba[selected, :3] = (colour.red(), colour.green(), colour.blue())
            rgba[selected, 3] = 156
        height, width = display_labels.shape
        image = QImage(
            rgba.data,
            width,
            height,
            int(rgba.strides[0]),
            QImage.Format.Format_RGBA8888,
        ).copy()
        item = self._scene.addPixmap(QPixmap.fromImage(image))
        item.setTransformationMode(Qt.TransformationMode.FastTransformation)
        item.setTransform(
            QTransform.fromScale(source_width / width, source_height / height)
        )
        item.setZValue(32)
        self._overlay_items.append(item)
        self._instance_annotation_overlay_item = item

    def _refresh_instance_annotation_overlay(self) -> None:
        self._render_instance_annotations()
        self._raise_instance_preview_items()

    def _start_instance_live_stroke(
        self, scene_point: QPointF, *, erase: bool
    ) -> None:
        self._clear_instance_live_stroke()
        colour = (
            QColor(255, 70, 85, 180)
            if erase
            else self.instance_colour(self._active_instance_id)
        )
        if not erase:
            colour.setAlpha(170)
        pen = QPen(colour, max(2.0, self._reference_brush_radius * 2.0))
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        path = QPainterPath(scene_point)
        path.lineTo(scene_point.x() + 0.01, scene_point.y())
        self._instance_live_stroke_path = path
        self._instance_live_stroke_item = self._scene.addPath(path, pen)
        self._instance_live_stroke_item.setZValue(1002.0)

    def _extend_instance_live_stroke(self, scene_point: QPointF) -> None:
        if (
            self._instance_live_stroke_path is None
            or self._instance_live_stroke_item is None
        ):
            return
        self._instance_live_stroke_path.lineTo(scene_point)
        self._instance_live_stroke_item.setPath(self._instance_live_stroke_path)

    def _clear_instance_live_stroke(self) -> None:
        item = self._instance_live_stroke_item
        if item is not None and item.scene() is self._scene:
            self._scene.removeItem(item)
        self._instance_live_stroke_item = None
        self._instance_live_stroke_path = None

    def _render_reference_mask(
        self,
        mask: np.ndarray | None,
        colour: tuple[int, int, int],
        z_value: float,
    ) -> None:
        if mask is None or not np.any(mask):
            return
        values = np.asarray(mask, dtype=bool)
        source_height, source_width = values.shape
        maximum_display_dimension = 2048
        if max(source_height, source_width) > maximum_display_dimension:
            factor = maximum_display_dimension / max(source_height, source_width)
            display_width = max(1, round(source_width * factor))
            display_height = max(1, round(source_height * factor))
            columns = np.minimum(
                (np.arange(display_width) * source_width / display_width).astype(int),
                source_width - 1,
            )
            rows = np.minimum(
                (np.arange(display_height) * source_height / display_height).astype(int),
                source_height - 1,
            )
            display_values = values[np.ix_(rows, columns)]
        else:
            display_values = values
        rgba = np.zeros((*display_values.shape, 4), np.uint8)
        rgba[:, :, :3] = colour
        rgba[:, :, 3] = np.uint8(display_values) * 92
        height, width = display_values.shape
        image = QImage(
            rgba.data,
            width,
            height,
            int(rgba.strides[0]),
            QImage.Format.Format_RGBA8888,
        ).copy()
        item = self._scene.addPixmap(QPixmap.fromImage(image))
        item.setTransformationMode(Qt.TransformationMode.FastTransformation)
        item.setTransform(
            QTransform.fromScale(source_width / width, source_height / height)
        )
        item.setZValue(z_value)
        self._overlay_items.append(item)

    def _set_bgr_base_image(self, bgr) -> None:
        if self._image_item is None:
            return
        contiguous = bgr if bgr.flags.c_contiguous else bgr.copy()
        height, width = contiguous.shape[:2]
        image = QImage(
            contiguous.data,
            width,
            height,
            int(contiguous.strides[0]),
            QImage.Format.Format_BGR888,
        ).copy()
        self._image_item.setPixmap(QPixmap.fromImage(image))
        self._scene.setSceneRect(self._image_item.boundingRect())

    def _restore_source_image(self) -> None:
        if self._image_item is None or self._source_image is None:
            return
        self._image_item.setPixmap(QPixmap.fromImage(self._source_image))
        self._scene.setSceneRect(self._image_item.boundingRect())

    def _render_colour_reference(self, result, *, corrected: bool) -> None:
        calibration = getattr(result, "calibration", None)
        card = None if calibration is None else calibration.colour_card
        if card is None:
            return
        bounds = (
            calibration.transform_points(card.bounds)
            if corrected
            else np.asarray(card.bounds, dtype=np.float32)
        )
        bounds_item = self._scene.addPolygon(
            QPolygonF([QPointF(float(x), float(y)) for x, y in bounds]),
            QPen(QColor("#41d9ff"), 9),
        )
        bounds_item.setZValue(12)
        bounds_item.setOpacity(self._overlay_opacity)
        self._overlay_items.append(bounds_item)
        swatch_pen = QPen(QColor("#ffdb55"), 6)
        for swatch in card.swatches:
            corners = (
                calibration.transform_points(swatch.corners)
                if corrected
                else np.asarray(swatch.corners, dtype=np.float32)
            )
            item = self._scene.addPolygon(
                QPolygonF([QPointF(float(x), float(y)) for x, y in corners]),
                swatch_pen,
            )
            item.setZValue(13)
            item.setOpacity(self._overlay_opacity)
            self._overlay_items.append(item)

    def _render_ruler_reference(self, result) -> None:
        calibration = getattr(result, "calibration", None)
        ruler = None if calibration is None else calibration.ruler
        if ruler is None:
            return
        endpoints = calibration.ruler_endpoints_corrected()
        point_a = QPointF(float(endpoints[0, 0]), float(endpoints[0, 1]))
        point_b = QPointF(float(endpoints[1, 0]), float(endpoints[1, 1]))
        line_pen = QPen(QColor("#ff58cc"), 11)
        line = self._scene.addLine(QLineF(point_a, point_b), line_pen)
        line.setZValue(13)
        line.setOpacity(self._overlay_opacity)
        self._overlay_items.append(line)
        endpoint_pen = QPen(QColor("#ffffff"), 7)
        radius = max(7.0, ruler.width_px * 0.04)
        for point in (point_a, point_b):
            marker = self._scene.addEllipse(
                point.x() - radius,
                point.y() - radius,
                radius * 2,
                radius * 2,
                endpoint_pen,
            )
            marker.setZValue(14)
            marker.setOpacity(self._overlay_opacity)
            self._overlay_items.append(marker)

    def _render_scale_bar(self, result) -> None:
        calibration = getattr(result, "calibration", None)
        if calibration is None or calibration.pixels_per_mm is None:
            return
        image_size = self.image_size
        if image_size is None:
            return
        width, height = image_size
        bar_mm = calibration.scale_bar_mm
        bar_pixels = bar_mm * calibration.pixels_per_mm
        margin = max(24.0, min(width, height) * 0.035)
        x1 = width - margin
        x0 = x1 - bar_pixels
        y = height - margin
        outline_pen = QPen(QColor("#101418"), max(8.0, height * 0.005))
        bar_pen = QPen(QColor("#ffffff"), max(3.0, height * 0.002))
        for pen in (outline_pen, bar_pen):
            line = self._scene.addLine(x0, y, x1, y, pen)
            line.setZValue(20)
            line.setOpacity(self._overlay_opacity)
            self._overlay_items.append(line)
        tick_height = max(12.0, height * 0.012)
        for x in (x0, x1):
            tick = self._scene.addLine(x, y - tick_height * 0.5, x, y + tick_height * 0.5, bar_pen)
            tick.setZValue(20)
            tick.setOpacity(self._overlay_opacity)
            self._overlay_items.append(tick)
        label_position = QPointF(x0, y - tick_height * 3.0)
        for offset, colour, z_value in (
            (QPointF(2.0, 2.0), QColor("#101418"), 20.0),
            (QPointF(0.0, 0.0), QColor("#ffffff"), 21.0),
        ):
            label = self._scene.addText("5 cm")
            label.setDefaultTextColor(colour)
            label.setPos(label_position + offset)
            label.setZValue(z_value)
            label.setOpacity(self._overlay_opacity)
            self._overlay_items.append(label)

    def _render_proposals(self, result) -> None:
        self._render_dish_edges(result)

        proposal_pen = QPen(QColor("#ffd84a"))
        proposal_pen.setWidth(2)
        proposal_pen.setCosmetic(True)
        center_pen = QPen(QColor("#ff4b4b"))
        center_pen.setWidth(2)
        center_pen.setCosmetic(True)
        for proposal in result.proposals:
            radius = proposal.radius
            ellipse = self._scene.addEllipse(
                proposal.center_x - radius,
                proposal.center_y - radius,
                radius * 2,
                radius * 2,
                proposal_pen,
            )
            ellipse.setZValue(11)
            ellipse.setOpacity(self._overlay_opacity)
            dot_radius = max(2.5, radius * 0.10)
            dot = self._scene.addEllipse(
                proposal.center_x - dot_radius,
                proposal.center_y - dot_radius,
                dot_radius * 2,
                dot_radius * 2,
                center_pen,
            )
            dot.setZValue(12)
            dot.setOpacity(self._overlay_opacity)
            self._overlay_items.extend((ellipse, dot))

    def fit_image(self) -> None:
        if self._image_item is None:
            return
        self.resetTransform()
        self.fitInView(
            self._image_item.boundingRect(),
            Qt.AspectRatioMode.KeepAspectRatio,
        )
        self._zoom_steps = 0
        self._update_zoom_indicator()

    def actual_size(self) -> None:
        if self._image_item is None:
            return
        self.resetTransform()
        self._zoom_steps = 0
        self._update_zoom_indicator()

    def zoom_in(self) -> None:
        self._zoom_by(1.2)

    def zoom_out(self) -> None:
        self._zoom_by(1.0 / 1.2)

    def _zoom_by(self, factor: float) -> None:
        if self._image_item is None:
            return
        current = float(self.transform().m11())
        target = current * float(factor)
        if not 0.03 <= target <= 32.0:
            return
        self.scale(float(factor), float(factor))
        self._update_zoom_indicator()

    def _update_zoom_indicator(self) -> None:
        self.zoom_label.setText(f"{self.transform().m11() * 100.0:.0f}%")

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 - Qt override
        if self._image_item is None:
            super().wheelEvent(event)
            return
        direction = 1 if event.angleDelta().y() > 0 else -1
        self._zoom_by(1.2 if direction > 0 else 1 / 1.2)
        self._zoom_steps += direction

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self.zoom_controls.setGeometry(0, 0, max(120, self.width()), 38)
        self._layout_context_panel()
        if self._context_panel is not None:
            self._context_panel.raise_()
        self.zoom_controls.raise_()

    def _instance_assisted_tool_active(self) -> bool:
        return (
            self._reference_point_mode == "instance"
            and not self._reference_erase_enabled
            and self._instance_annotation_tool
            in {"edge_trace", "shape_snap", "smart_fill"}
        )

    def _remove_instance_preview_items(self) -> None:
        for item in self._instance_preview_items:
            if item.scene() is self._scene:
                self._scene.removeItem(item)
        self._instance_preview_items.clear()

    def _clear_instance_preview(self) -> None:
        self._instance_preview_timer.stop()
        self._pending_instance_preview_point = None
        self._instance_preview_geometry = None
        self._instance_shape_reference_geometry = None
        self._instance_preview_region = None
        self._instance_preview_endpoint = None
        self._instance_preview_point = None
        self._remove_instance_preview_items()

    def _raise_instance_preview_items(self) -> None:
        for item in self._instance_preview_items:
            item.setZValue(1001.0)

    def _schedule_instance_preview(self, scene_point: QPointF | None) -> None:
        if (
            scene_point is None
            or not self._instance_assisted_tool_active()
            or self._image_item is None
            or not self._image_item.boundingRect().contains(scene_point)
        ):
            return
        self._pending_instance_preview_point = QPointF(scene_point)
        if not self._instance_preview_timer.isActive():
            self._instance_preview_timer.start()

    def _update_pending_instance_preview(self) -> None:
        point = self._pending_instance_preview_point
        self._pending_instance_preview_point = None
        if point is not None:
            self._update_instance_assisted_preview(point)

    @staticmethod
    def _preview_path(geometry: np.ndarray, *, closed: bool) -> QPainterPath:
        points = np.asarray(geometry, dtype=np.float32).reshape(-1, 2)
        path = QPainterPath()
        if not len(points):
            return path
        path.moveTo(float(points[0, 0]), float(points[0, 1]))
        for x, y in points[1:]:
            path.lineTo(float(x), float(y))
        if closed:
            path.closeSubpath()
        return path

    def _shape_reference_polygon(self, centre: QPointF, radius: float) -> np.ndarray:
        options = self._shape_snap_options
        angles = np.linspace(
            0.0,
            2.0 * np.pi,
            options.angular_samples,
            endpoint=False,
            dtype=np.float32,
        )
        radius_x = max(3.0, float(radius))
        radius_y = radius_x if options.shape == "circle" else radius_x * 0.72
        rotation = np.deg2rad(options.rotation_degrees)
        cosine, sine = np.cos(angles), np.sin(angles)
        local_x, local_y = radius_x * cosine, radius_y * sine
        cos_r, sin_r = float(np.cos(rotation)), float(np.sin(rotation))
        return np.column_stack(
            (
                centre.x() + local_x * cos_r - local_y * sin_r,
                centre.y() + local_x * sin_r + local_y * cos_r,
            )
        ).astype(np.float32)

    def _render_instance_assisted_preview(self) -> None:
        self._remove_instance_preview_items()
        colour = self.instance_colour(self._active_instance_id)
        pen = QPen(colour, 3.0)
        pen.setCosmetic(True)
        pen.setStyle(Qt.PenStyle.DashLine)
        if self._instance_annotation_tool == "smart_fill":
            region = self._instance_preview_region
            if region is not None and region.added_count:
                rgba = np.zeros((*region.mask.shape, 4), np.uint8)
                rgba[region.mask, :3] = (
                    colour.red(),
                    colour.green(),
                    colour.blue(),
                )
                rgba[region.mask, 3] = 108
                height, width = region.mask.shape
                image = QImage(
                    rgba.data,
                    width,
                    height,
                    int(rgba.strides[0]),
                    QImage.Format.Format_RGBA8888,
                ).copy()
                item = self._scene.addPixmap(QPixmap.fromImage(image))
                item.setPos(float(region.x), float(region.y))
                self._instance_preview_items.append(item)
        else:
            geometry = self._instance_preview_geometry
            if (
                self._instance_annotation_tool == "shape_snap"
                and self._instance_shape_reference_geometry is not None
            ):
                reference_pen = QPen(QColor("#ffffff"), 1.5)
                reference_pen.setCosmetic(True)
                reference_pen.setStyle(Qt.PenStyle.DotLine)
                reference_item = self._scene.addPath(
                    self._preview_path(
                        self._instance_shape_reference_geometry,
                        closed=True,
                    ),
                    reference_pen,
                    QBrush(Qt.BrushStyle.NoBrush),
                )
                self._instance_preview_items.append(reference_item)
            if geometry is not None and len(geometry) > 1:
                closed = self._instance_annotation_tool == "shape_snap"
                fill = QColor(colour)
                fill.setAlpha(55 if closed else 0)
                item = self._scene.addPath(
                    self._preview_path(geometry, closed=closed),
                    pen,
                    QBrush(fill) if closed else QBrush(Qt.BrushStyle.NoBrush),
                )
                self._instance_preview_items.append(item)
            endpoint = self._instance_preview_endpoint
            if endpoint is not None:
                radius = max(3.0, self._edge_trace_options.search_radius_px * 0.18)
                marker = self._scene.addEllipse(
                    endpoint.x() - radius,
                    endpoint.y() - radius,
                    radius * 2.0,
                    radius * 2.0,
                    pen,
                    QBrush(QColor(colour.red(), colour.green(), colour.blue(), 70)),
                )
                self._instance_preview_items.append(marker)
        if self._instance_trace_anchor is not None:
            anchor = self._instance_trace_anchor
            anchor_pen = QPen(QColor("#ffffff"), 2.0)
            anchor_pen.setCosmetic(True)
            marker = self._scene.addEllipse(
                anchor.x() - 4.0,
                anchor.y() - 4.0,
                8.0,
                8.0,
                anchor_pen,
                QBrush(colour),
            )
            self._instance_preview_items.append(marker)
        self._raise_instance_preview_items()

    def _update_instance_assisted_preview(self, scene_point: QPointF) -> None:
        if not self._instance_assisted_tool_active():
            self._clear_instance_preview()
            return
        try:
            edge = self._full_annotation_evidence("edge_likelihood")
            self._instance_preview_point = QPointF(scene_point)
            self._instance_preview_geometry = None
            self._instance_shape_reference_geometry = None
            self._instance_preview_region = None
            self._instance_preview_endpoint = QPointF(scene_point)
            if self._instance_annotation_tool == "edge_trace":
                snapped = snap_edge_point(
                    (scene_point.x(), scene_point.y()),
                    edge,
                    self._edge_trace_options.search_radius_px,
                )
                endpoint = QPointF(float(snapped[0]), float(snapped[1]))
                self._instance_preview_endpoint = endpoint
                if self._instance_trace_anchor is None:
                    self._instance_preview_geometry = np.asarray(
                        ((snapped[0], snapped[1]),), dtype=np.int32
                    )
                else:
                    tangent = self._annotation_tangent(
                        self._edge_trace_options.tangent_mode
                    )
                    self._instance_preview_geometry = trace_edge_path(
                        (
                            self._instance_trace_anchor.x(),
                            self._instance_trace_anchor.y(),
                        ),
                        (endpoint.x(), endpoint.y()),
                        edge,
                        self._edge_trace_options,
                        tangent,
                    )
            elif self._instance_annotation_tool == "shape_snap":
                tangent = self._annotation_tangent(
                    self._shape_snap_options.tangent_mode
                )
                fallback = float(
                    getattr(
                        self._analysis_result,
                        "estimated_seed_diameter_px",
                        60.0,
                    )
                ) * 0.5
                self._instance_shape_reference_geometry = (
                    self._shape_reference_polygon(scene_point, fallback)
                )
                self._instance_preview_geometry = snap_shape_polygon(
                    (scene_point.x(), scene_point.y()),
                    (scene_point.x(), scene_point.y()),
                    edge,
                    self._shape_snap_options,
                    tangent,
                    fallback_radius_px=fallback,
                )
                self._instance_preview_endpoint = None
            else:
                labels = self._ensure_instance_labels()
                corrected = np.asarray(self._analysis_result.calibration.corrected_bgr)
                self._instance_preview_region = smart_fill_region(
                    labels,
                    corrected,
                    edge,
                    (scene_point.x(), scene_point.y()),
                    self._active_instance_id,
                    self._smart_fill_options,
                )
                self._instance_preview_endpoint = None
            self._render_instance_assisted_preview()
        except (RuntimeError, ValueError) as error:
            self._clear_instance_preview()
            self.instance_tool_status.emit(str(error))

    def _preview_matches(self, scene_point: QPointF) -> bool:
        return (
            self._instance_preview_point is not None
            and QLineF(self._instance_preview_point, scene_point).length() <= 0.75
        )

    def _commit_instance_assisted_preview(self, scene_point: QPointF) -> bool:
        if not self._preview_matches(scene_point):
            self._update_instance_assisted_preview(scene_point)
        if self._instance_annotation_tool == "edge_trace":
            endpoint = self._instance_preview_endpoint
            if endpoint is None:
                return False
            if self._instance_trace_anchor is None:
                self._instance_trace_anchor = QPointF(endpoint)
                self.instance_tool_status.emit(
                    "Edge-trace anchor set. Move the cursor to preview a snapped "
                    "segment, then click to apply it."
                )
                self._render_instance_assisted_preview()
                return False
            geometry = self._instance_preview_geometry
            if geometry is None or len(geometry) < 2:
                return False
            changed = self._paint_instance_geometry(geometry, filled=False)
            self._instance_trace_anchor = QPointF(endpoint)
            self.instance_tool_status.emit(
                f"Edge trace added {changed:,} pixels to seed "
                f"{self._active_instance_id}; the endpoint is the next anchor."
            )
        elif self._instance_annotation_tool == "shape_snap":
            geometry = self._instance_preview_geometry
            if geometry is None or len(geometry) < 3:
                return False
            changed = self._paint_instance_geometry(geometry, filled=True)
            self.instance_tool_status.emit(
                f"Shape snap added {changed:,} pixels to seed "
                f"{self._active_instance_id}."
            )
        else:
            region = self._instance_preview_region
            if region is None or not region.added_count:
                self.instance_tool_status.emit(
                    "Smart fill found no reachable pixels at this cursor position."
                )
                return False
            labels = self._ensure_instance_labels()
            height, width = region.mask.shape
            roi = labels[
                region.y : region.y + height,
                region.x : region.x + width,
            ]
            writable = region.mask & (
                (roi == 0) | (roi == self._active_instance_id)
            )
            changed = int(
                np.count_nonzero(writable & (roi != self._active_instance_id))
            )
            roi[writable] = np.uint16(self._active_instance_id)
            self.instance_tool_status.emit(
                f"Smart fill added {changed:,} locally matched pixels to seed "
                f"{self._active_instance_id}."
            )
        if changed:
            self._refresh_instance_annotation_overlay()
            # The main window owns this draft from here. Avoid a second
            # full-resolution label-map copy on every assisted click.
            self.instance_annotations_edited.emit(self._instance_annotations)
            self.reference_paint_finished.emit()
        self._schedule_instance_preview(scene_point)
        return changed > 0

    def _instance_uses_assisted_tool(self, button: Qt.MouseButton) -> bool:
        return button == Qt.MouseButton.LeftButton and self._instance_assisted_tool_active()

    def _full_annotation_evidence(self, attribute: str) -> np.ndarray:
        """Materialize one compact analysis raster in full-image coordinates."""

        cached = self._annotation_evidence_cache.get(attribute)
        if cached is not None:
            return cached
        result = self._analysis_result
        image_size = self.image_size
        if result is None or image_size is None:
            raise RuntimeError("Run the image analysis before using assisted tools.")
        source = getattr(result.layers, attribute, None)
        if source is None:
            raise RuntimeError(f"The analysis did not produce {attribute.replace('_', ' ')}.")
        values = np.asarray(source)
        if values.ndim != 2:
            values = np.squeeze(values)
        if values.ndim != 2:
            raise RuntimeError(f"{attribute.replace('_', ' ').capitalize()} is not a raster.")
        width, height = image_size
        if values.shape == (height, width):
            full = values
        else:
            full = np.zeros((height, width), dtype=values.dtype)
            offset_x = int(getattr(result.layers, "offset_x", result.crop_offset[0]))
            offset_y = int(getattr(result.layers, "offset_y", result.crop_offset[1]))
            target_x0, target_y0 = max(0, offset_x), max(0, offset_y)
            source_x0, source_y0 = max(0, -offset_x), max(0, -offset_y)
            copy_width = min(values.shape[1] - source_x0, width - target_x0)
            copy_height = min(values.shape[0] - source_y0, height - target_y0)
            if copy_width > 0 and copy_height > 0:
                full[
                    target_y0 : target_y0 + copy_height,
                    target_x0 : target_x0 + copy_width,
                ] = values[
                    source_y0 : source_y0 + copy_height,
                    source_x0 : source_x0 + copy_width,
                ]
        self._annotation_evidence_cache[attribute] = full
        return full

    def _annotation_tangent(self, mode: str) -> np.ndarray | None:
        if mode == "off":
            return None
        attribute = (
            "undirected_edge_hue" if mode == "undirected" else "directed_edge_hue"
        )
        return self._full_annotation_evidence(attribute)

    def _ensure_instance_labels(self) -> np.ndarray:
        image_size = self.image_size
        if image_size is None:
            raise RuntimeError("No image is loaded.")
        width, height = image_size
        if self._instance_annotations is None or self._instance_annotations.shape != (height, width):
            self._instance_annotations = np.zeros((height, width), dtype=np.uint16)
        return self._instance_annotations

    def _paint_instance_geometry(self, geometry: np.ndarray, *, filled: bool) -> int:
        labels = self._ensure_instance_labels()
        paint = np.zeros(labels.shape, dtype=np.uint8)
        points = np.asarray(geometry, dtype=np.int32).reshape((-1, 1, 2))
        if filled:
            cv2.fillPoly(paint, [points], 255, lineType=cv2.LINE_AA)
        else:
            thickness = max(1, int(round(self._reference_brush_radius * 0.7)))
            cv2.polylines(
                paint,
                [points],
                False,
                255,
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )
        writable = (paint > 0) & (
            (labels == 0) | (labels == self._active_instance_id)
        )
        changed = int(np.count_nonzero(writable & (labels != self._active_instance_id)))
        labels[writable] = np.uint16(self._active_instance_id)
        return changed

    def _apply_instance_assisted_tool(self, end_point: QPointF) -> bool:
        """Apply the current image-aware tool and report whether labels changed."""

        try:
            edge = self._full_annotation_evidence("edge_likelihood")
            if self._instance_annotation_tool == "edge_trace":
                points = [*self._instance_tool_points]
                if not points or QLineF(points[-1], end_point).length() > 0.5:
                    points.append(QPointF(end_point))
                if len(points) < 2:
                    return False
                tangent = self._annotation_tangent(
                    self._edge_trace_options.tangent_mode
                )
                changed = 0
                for start, end in zip(points, points[1:]):
                    path = trace_edge_path(
                        (start.x(), start.y()),
                        (end.x(), end.y()),
                        edge,
                        self._edge_trace_options,
                        tangent,
                    )
                    changed += self._paint_instance_geometry(path, filled=False)
                self.instance_tool_status.emit(
                    f"Edge trace added {changed:,} pixels to seed {self._active_instance_id}."
                )
                return changed > 0
            if self._instance_annotation_tool == "shape_snap":
                if not self._instance_tool_points:
                    return False
                tangent = self._annotation_tangent(
                    self._shape_snap_options.tangent_mode
                )
                start = self._instance_tool_points[0]
                fallback = float(
                    getattr(self._analysis_result, "estimated_seed_diameter_px", 60.0)
                ) * 0.5
                polygon = snap_shape_polygon(
                    (start.x(), start.y()),
                    (end_point.x(), end_point.y()),
                    edge,
                    self._shape_snap_options,
                    tangent,
                    fallback_radius_px=fallback,
                )
                changed = self._paint_instance_geometry(polygon, filled=True)
                self.instance_tool_status.emit(
                    f"Shape snap added {changed:,} pixels to seed {self._active_instance_id}."
                )
                return changed > 0
            if self._instance_annotation_tool == "smart_fill":
                labels = self._ensure_instance_labels()
                corrected = np.asarray(self._analysis_result.calibration.corrected_bgr)
                filled, changed = smart_fill_instance(
                    labels,
                    corrected,
                    edge,
                    (end_point.x(), end_point.y()),
                    self._active_instance_id,
                    self._smart_fill_options,
                )
                if changed:
                    self._instance_annotations = filled
                    self.instance_tool_status.emit(
                        f"Smart fill added {changed:,} locally matched pixels to seed "
                        f"{self._active_instance_id}."
                    )
                    return True
                self.instance_tool_status.emit(
                    "Smart fill found no reachable pixels. Move inside the seed or "
                    "adjust its edge/tolerance settings."
                )
        except (RuntimeError, ValueError) as error:
            self.instance_tool_status.emit(str(error))
        return False

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._reference_point_mode is None or self._image_item is None:
            super().mousePressEvent(event)
            return
        scene_point = self.mapToScene(event.position().toPoint())
        if not self._image_item.boundingRect().contains(scene_point):
            event.ignore()
            return
        if event.button() in {
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.RightButton,
        }:
            if self._instance_uses_assisted_tool(event.button()):
                self._update_reference_brush_outline(scene_point)
                self._commit_instance_assisted_preview(scene_point)
                event.accept()
                return
            self._reference_paint_button = event.button()
            self._reference_stroke_erases = (
                self._reference_erase_enabled
                or event.button() == Qt.MouseButton.RightButton
            )
            self._last_reference_paint_point = scene_point
            self._update_reference_brush_outline(scene_point)
            self._emit_reference_brush_dab(
                scene_point,
                erase=self._reference_stroke_erases,
            )
            if self._reference_point_mode == "instance":
                self._start_instance_live_stroke(
                    scene_point,
                    erase=self._reference_stroke_erases,
                )
            else:
                self._render_analysis()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._reference_point_mode is not None and self._image_item is not None:
            self._update_reference_brush_outline(
                self.mapToScene(event.position().toPoint())
            )
        if (
            self._reference_point_mode is None
            or self._reference_paint_button is None
            or self._image_item is None
        ):
            super().mouseMoveEvent(event)
            return
        scene_point = self.mapToScene(event.position().toPoint())
        if not self._image_item.boundingRect().contains(scene_point):
            event.accept()
            return
        previous = self._last_reference_paint_point or scene_point
        dx = scene_point.x() - previous.x()
        dy = scene_point.y() - previous.y()
        distance = (dx * dx + dy * dy) ** 0.5
        spacing = max(1.0, self._reference_brush_radius * 0.45)
        step_count = max(1, int(np.ceil(distance / spacing)))
        for step in range(1, step_count + 1):
            fraction = step / step_count
            interpolated = QPointF(
                previous.x() + dx * fraction,
                previous.y() + dy * fraction,
            )
            self._emit_reference_brush_dab(
                interpolated,
                erase=self._reference_stroke_erases,
            )
        # Interpolation may add many mask circles, but only one viewer redraw is
        # required for this mouse event.
        if self._reference_point_mode == "instance":
            self._extend_instance_live_stroke(scene_point)
        else:
            self._render_analysis()
        self._last_reference_paint_point = scene_point
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if (
            self._reference_point_mode is not None
            and self._reference_paint_button == event.button()
        ):
            assisted = self._instance_uses_assisted_tool(event.button())
            changed = False
            if assisted:
                scene_point = self.mapToScene(event.position().toPoint())
                if self._image_item is not None and self._image_item.boundingRect().contains(scene_point):
                    changed = self._apply_instance_assisted_tool(scene_point)
            self._reference_paint_button = None
            self._reference_stroke_erases = False
            self._last_reference_paint_point = None
            self._instance_tool_points.clear()
            if self._last_reference_hover_point is not None:
                self._update_reference_brush_outline(
                    self._last_reference_hover_point
                )
            if self._reference_point_mode == "instance":
                if changed or not assisted:
                    # Emit the mutable draft itself; applied annotations are
                    # copied only when the user commits them.
                    self.instance_annotations_edited.emit(self._instance_annotations)
                self._refresh_instance_annotation_overlay()
                self._clear_instance_live_stroke()
            else:
                mask = self.reference_mask(self._reference_point_mode)
                self.reference_mask_edited.emit(self._reference_point_mode, mask)
            if changed and self._reference_point_mode != "instance":
                self._render_analysis()
            self.reference_paint_finished.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._last_reference_hover_point = None
        self._hide_reference_brush_outline()
        self._instance_preview_point = None
        self._instance_preview_geometry = None
        self._instance_shape_reference_geometry = None
        self._instance_preview_region = None
        self._instance_preview_endpoint = None
        self._remove_instance_preview_items()
        super().leaveEvent(event)

    def _emit_reference_brush_dab(
        self, scene_point: QPointF, *, erase: bool
    ) -> None:
        if self._reference_point_mode is None:
            return
        image_size = self.image_size
        if image_size is None:
            return
        width, height = image_size
        if self._reference_point_mode == "instance":
            attribute = "_instance_annotations"
            mask = getattr(self, attribute)
            if mask is None or mask.shape != (height, width):
                mask = np.zeros((height, width), dtype=np.uint16)
                setattr(self, attribute, mask)
            value = 0 if erase else self._active_instance_id
        else:
            attributes = {
                "background": "_background_reference_mask",
                "foreground": "_foreground_reference_mask",
                "background_exclusion": "_background_exclusion_mask",
                "foreground_exclusion": "_foreground_exclusion_mask",
            }
            attribute = attributes[self._reference_point_mode]
            mask = getattr(self, attribute)
            if mask is None or mask.shape != (height, width):
                mask = np.zeros((height, width), dtype=bool)
                setattr(self, attribute, mask)
            value = not erase
        self._paint_mask_circle(
            mask,
            scene_point.x(),
            scene_point.y(),
            value,
        )

    def _paint_mask_circle(
        self,
        mask: np.ndarray,
        point_x: float,
        point_y: float,
        value: bool | int,
    ) -> None:
        radius = max(1, int(round(self._reference_brush_radius)))
        center_x = int(round(point_x))
        center_y = int(round(point_y))
        x0 = max(0, center_x - radius)
        x1 = min(mask.shape[1], center_x + radius + 1)
        y0 = max(0, center_y - radius)
        y1 = min(mask.shape[0], center_y + radius + 1)
        if x0 >= x1 or y0 >= y1:
            return
        yy, xx = np.ogrid[y0:y1, x0:x1]
        circle = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius**2
        region = mask[y0:y1, x0:x1]
        region[circle] = value

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        urls = event.mimeData().urls()
        if any(
            url.isLocalFile()
            and Path(url.toLocalFile()).suffix.lower() in SUPPORTED_SUFFIXES
            for url in urls
        ):
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        for url in event.mimeData().urls():
            if not url.isLocalFile():
                continue
            path = Path(url.toLocalFile())
            if path.suffix.lower() in SUPPORTED_SUFFIXES:
                self.image_dropped.emit(str(path))
                event.acceptProposedAction()
                return
        super().dropEvent(event)
