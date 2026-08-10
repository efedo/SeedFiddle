"""Zoomable image canvas used for image review and future mask editing."""

from __future__ import annotations

from pathlib import Path

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
from PySide6.QtWidgets import QGraphicsScene, QGraphicsTextItem, QGraphicsView

from seedvision.visualization import ADVANCED_OVERLAY_LABELS


SUPPORTED_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
OVERLAY_MODES = {
    "raw_image",
    "deskew_colour",
    "calibrated_image",
    "colour_reference",
    "ruler_detection",
    "layout_detection",
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
    "edge_gradients",
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
        self._reference_point_mode: str | None = None
        self._reference_brush_radius = 12.0
        self._reference_erase_enabled = False
        self._reference_paint_button: Qt.MouseButton | None = None
        self._reference_stroke_erases = False
        self._last_reference_paint_point: QPointF | None = None
        self._last_reference_hover_point: QPointF | None = None
        self._reference_brush_outline_item = None

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
        self._show_placeholder()

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
        self._reference_brush_outline_item = None
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
        self._restore_source_image()

    def _clear_overlay_items(self) -> None:
        for item in self._overlay_items:
            self._scene.removeItem(item)
        self._overlay_items.clear()

    def show_analysis(self, result) -> None:
        """Store an analysis result and render the selected overlay layer."""

        self._analysis_result = result
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
    ) -> None:
        """Display editable full-resolution binary masks in image coordinates."""

        self._background_reference_mask = self._normalized_reference_mask(
            background_mask
        )
        self._foreground_reference_mask = self._normalized_reference_mask(
            foreground_mask
        )
        self._render_analysis()

    def reference_mask(self, class_name: str) -> np.ndarray | None:
        mask = (
            self._background_reference_mask
            if class_name == "background"
            else self._foreground_reference_mask
        )
        return None if mask is None else mask.copy()

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

    def set_reference_brush_radius(self, radius: float) -> None:
        """Set the corrected-image radius of the visibly painted sample area."""

        self._reference_brush_radius = max(2.0, float(radius))
        if self._last_reference_hover_point is not None:
            self._update_reference_brush_outline(
                self._last_reference_hover_point
            )
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
        elif self._overlay_mode == "edge_gradients":
            rgba = layers.edge_magnitude_rgba()
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
            f"Initial background sampling band: {band.sample_count:,} pixels; "
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
        self._render_background_reference_points()
        self._render_foreground_reference_points()

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

    def actual_size(self) -> None:
        if self._image_item is None:
            return
        self.resetTransform()
        self._zoom_steps = 0

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 - Qt override
        if self._image_item is None:
            super().wheelEvent(event)
            return
        direction = 1 if event.angleDelta().y() > 0 else -1
        if direction < 0 and self._zoom_steps <= -8:
            return
        if direction > 0 and self._zoom_steps >= 30:
            return
        factor = 1.2 if direction > 0 else 1 / 1.2
        self.scale(factor, factor)
        self._zoom_steps += direction

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
        self._render_analysis()
        self._last_reference_paint_point = scene_point
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if (
            self._reference_point_mode is not None
            and self._reference_paint_button == event.button()
        ):
            self._reference_paint_button = None
            self._reference_stroke_erases = False
            self._last_reference_paint_point = None
            if self._last_reference_hover_point is not None:
                self._update_reference_brush_outline(
                    self._last_reference_hover_point
                )
            mask = self.reference_mask(self._reference_point_mode)
            self.reference_mask_edited.emit(self._reference_point_mode, mask)
            self.reference_paint_finished.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._last_reference_hover_point = None
        self._hide_reference_brush_outline()
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
        attribute = (
            "_background_reference_mask"
            if self._reference_point_mode == "background"
            else "_foreground_reference_mask"
        )
        mask = getattr(self, attribute)
        if mask is None or mask.shape != (height, width):
            mask = np.zeros((height, width), dtype=bool)
            setattr(self, attribute, mask)
        self._paint_mask_circle(
            mask,
            scene_point.x(),
            scene_point.y(),
            not erase,
        )

    def _paint_mask_circle(
        self,
        mask: np.ndarray,
        point_x: float,
        point_y: float,
        value: bool,
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
