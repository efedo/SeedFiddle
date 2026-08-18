"""Zoomable image canvas used for image review and future mask editing."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QEvent, QLineF, QPoint, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QDragEnterEvent,
    QDropEvent,
    QFont,
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
    QGraphicsItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QToolButton,
)

from seedvision.ui.canvas_controls import style_canvas_control_bar
from seedvision.visualization import ADVANCED_OVERLAY_LABELS
from seedvision.resources import release_host_caches
from seedvision.annotation import (
    EdgeTraceOptions,
    ShapeGuidedFillOptions,
    ShapeGuidedFillRegion,
    SmartFillOptions,
    SmartFillRegion,
    smart_fill_instance,
    smart_fill_region,
    shape_guided_fill_region,
    snap_edge_point,
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
    "foreground_colour_gamut",
    "foreground_binary_mask",
    "distance_transform",
    "distance_candidates",
    "circle_candidates",
    "proposals",
    "instance_masks",
    "background_likelihood",
    "background_colour_gamut",
    "refined_background_likelihood",
    "foreground_noise_likelihood",
    "reference_texture_prototypes",
    "reference_seed_surface_probability",
    "reference_background_texture_probability",
    "reference_other_texture_probability",
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
    "physical_edge_probability",
    "non_edge_probability",
    "reference_edge_comparison",
    "net_physical_edge_probability",
    "reference_edge_ridges",
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
    "procedural_seed_material",
    "procedural_seed_mask",
    "procedural_boundary_cost",
    "procedural_centres",
    "procedural_instances",
    "procedural_confidence",
    "unet_interior",
    "unet_physical_boundary",
    "unet_pattern_boundary",
    "unet_centres",
    "unet_distance",
    "unet_uncertainty",
    "unet_instances",
    "unet_confidence",
    "stardist_object_probability",
    "stardist_radial_uncertainty",
    "stardist_instances",
    "stardist_confidence",
    "none",
    *(mode for _, mode in ADVANCED_OVERLAY_LABELS),
}


def _edge_strip_annotation_lines(
    patch_rect: QRectF,
    patch_shape: tuple[int, int],
    *,
    normal_offset_px: float,
    tangent_half_length_px: float,
) -> tuple[QLineF, tuple[QLineF, ...]]:
    """Map the edge descriptor's source-pixel strips onto a thumbnail.

    Edge thumbnails are tangent-aligned before display, so the pooled strips
    are horizontal.  The descriptor evaluates both normal polarities; the two
    returned side lines are therefore deliberately unordered rather than
    labelled as interior and exterior.  A side strip outside the displayed
    context is omitted instead of being misleadingly clamped onto its border.
    """

    patch_height, patch_width = (int(value) for value in patch_shape)
    if patch_height <= 0 or patch_width <= 0:
        raise ValueError("Prototype patch dimensions must be positive.")
    scale_x = float(patch_rect.width()) / float(patch_width)
    scale_y = float(patch_rect.height()) / float(patch_height)
    centre_x = float(patch_rect.center().x())
    centre_y = float(patch_rect.center().y())
    half_length = max(0.0, float(tangent_half_length_px)) * scale_x
    left = max(float(patch_rect.left()), centre_x - half_length)
    right = min(float(patch_rect.right()), centre_x + half_length)
    centre_line = QLineF(left, centre_y, right, centre_y)
    normal_offset = max(0.0, float(normal_offset_px)) * scale_y
    side_lines = tuple(
        QLineF(left, side_y, right, side_y)
        for side_y in (centre_y - normal_offset, centre_y + normal_offset)
        if float(patch_rect.top()) <= side_y <= float(patch_rect.bottom())
    )
    return centre_line, side_lines


class _InstanceAnnotationTileItem(QGraphicsItem):
    """Paint an integer instance map from bounded, full-resolution tiles.

    The label map remains authoritative at corrected-image resolution.  Qt asks
    this one item to paint only its exposed scene rectangle, which lets us
    materialize 512-pixel colour-indexed tiles as the viewport encounters them
    rather than creating one scaled-down pixmap for the complete photograph.
    The LRU bound is deliberately independent of image dimensions.
    """

    TILE_SIZE = 512
    CACHE_CAPACITY = 512
    CACHE_BYTE_CAPACITY = 48 * 1024 * 1024

    def __init__(
        self,
        labels: np.ndarray,
        *,
        colour_for_identifier,
        selected_identifier: int | None = None,
        render_bounds: tuple[int, int, int, int] | None = None,
    ) -> None:
        super().__init__()
        self._labels = np.asarray(labels, dtype=np.uint16)
        self._colour_for_identifier = colour_for_identifier
        self._selected_identifier = (
            None if selected_identifier is None else int(selected_identifier)
        )
        self._render_bounds = self._normalized_bounds(render_bounds)
        self._tile_cache: OrderedDict[
            tuple[int, int, int, int], QImage
        ] = OrderedDict()
        self._tile_cache_bytes = 0
        self._generation = 0
        self.setFlag(
            QGraphicsItem.GraphicsItemFlag.ItemUsesExtendedStyleOption, True
        )
        self.setCacheMode(QGraphicsItem.CacheMode.NoCache)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.setAcceptHoverEvents(False)

    @property
    def tile_size(self) -> int:
        return self.TILE_SIZE

    @property
    def cache_capacity(self) -> int:
        return self.CACHE_CAPACITY

    @property
    def cache_size(self) -> int:
        return len(self._tile_cache)

    @property
    def cache_keys(self) -> tuple[tuple[int, int], ...]:
        """Return cached global tile coordinates (generation stays internal)."""

        return tuple((key[2], key[3]) for key in self._tile_cache)

    @property
    def cache_bytes(self) -> int:
        return self._tile_cache_bytes

    @property
    def cache_byte_capacity(self) -> int:
        return self.CACHE_BYTE_CAPACITY

    @property
    def generation(self) -> int:
        return self._generation

    def _normalized_bounds(
        self, bounds: tuple[int, int, int, int] | None
    ) -> tuple[int, int, int, int]:
        height, width = self._labels.shape
        if bounds is None:
            return (0, 0, width, height)
        x0, y0, x1, y1 = (int(value) for value in bounds)
        x0 = max(0, min(width, x0))
        y0 = max(0, min(height, y0))
        x1 = max(x0, min(width, x1))
        y1 = max(y0, min(height, y1))
        return (x0, y0, x1, y1)

    def boundingRect(self) -> QRectF:  # noqa: N802 - Qt override
        x0, y0, x1, y1 = self._render_bounds
        return QRectF(float(x0), float(y0), float(x1 - x0), float(y1 - y0))

    def replace_annotations(
        self,
        labels: np.ndarray,
        *,
        selected_identifier: int | None,
        render_bounds: tuple[int, int, int, int] | None,
    ) -> None:
        """Install a new label generation and discard every cached tile."""

        values = np.asarray(labels, dtype=np.uint16)
        old_bounds = self._render_bounds
        self._labels = values
        new_bounds = self._normalized_bounds(render_bounds)
        if new_bounds != old_bounds:
            self.prepareGeometryChange()
            self._render_bounds = new_bounds
        self._selected_identifier = (
            None if selected_identifier is None else int(selected_identifier)
        )
        self.invalidate_tiles()

    def invalidate_tiles(self) -> None:
        """Invalidate cached imagery after any authoritative label edit."""

        self._generation += 1
        self._tile_cache.clear()
        self._tile_cache_bytes = 0
        self.update()

    def _tile_image(self, tile_x: int, tile_y: int) -> QImage:
        """Return one exact-resolution global tile, cached by tile index."""

        tile_x = int(tile_x)
        tile_y = int(tile_y)
        selected_key = self._selected_identifier or 0
        key = (self._generation, selected_key, tile_x, tile_y)
        if key in self._tile_cache:
            cached = self._tile_cache[key]
            self._tile_cache.move_to_end(key)
            return cached

        height, width = self._labels.shape
        x0 = tile_x * self.TILE_SIZE
        y0 = tile_y * self.TILE_SIZE
        x1 = min(width, x0 + self.TILE_SIZE)
        y1 = min(height, y0 + self.TILE_SIZE)
        if x0 < 0 or y0 < 0 or x0 >= width or y0 >= height:
            image = QImage()
        else:
            tile_labels = self._labels[y0:y1, x0:x1]
            if self._selected_identifier is not None:
                selected = tile_labels == self._selected_identifier
                if not np.any(selected):
                    image = QImage()
                else:
                    colour = self._colour_for_identifier(self._selected_identifier)
                    image = self._owned_indexed_image(
                        np.asarray(selected, dtype=np.uint8),
                        (
                            QColor(0, 0, 0, 0).rgba(),
                            QColor(
                                colour.red(),
                                colour.green(),
                                colour.blue(),
                                156,
                            ).rgba(),
                        ),
                    )
            else:
                identifiers = np.unique(tile_labels)
                identifiers = identifiers[identifiers != 0]
                if not len(identifiers):
                    image = QImage()
                elif len(identifiers) <= 255:
                    # Indexed colour retains one exact label-map pixel per
                    # image byte. A normal 512-pixel seed tile contains far
                    # fewer than 255 identities, so a complete 6240 x 4160
                    # viewport fits inside the fixed 48 MiB cache without a
                    # lower-resolution display pyramid.
                    local_lut = np.zeros(1 << 16, dtype=np.uint8)
                    colour_table = [QColor(0, 0, 0, 0).rgba()]
                    for palette_index, identifier_value in enumerate(
                        identifiers, start=1
                    ):
                        identifier = int(identifier_value)
                        local_lut[identifier] = palette_index
                        colour = self._colour_for_identifier(identifier)
                        colour_table.append(
                            QColor(
                                colour.red(),
                                colour.green(),
                                colour.blue(),
                                156,
                            ).rgba()
                        )
                    image = self._owned_indexed_image(
                        local_lut[tile_labels], colour_table
                    )
                else:
                    # Pathological tiles with more than 255 distinct identities
                    # cannot be represented by an eight-bit local palette.
                    colour_lut = np.zeros((1 << 16, 4), dtype=np.uint8)
                    for identifier_value in identifiers:
                        identifier = int(identifier_value)
                        colour = self._colour_for_identifier(identifier)
                        colour_lut[identifier] = (
                            colour.red(),
                            colour.green(),
                            colour.blue(),
                            156,
                        )
                    image = self._owned_rgba_image(colour_lut[tile_labels])

        image_cost = int(image.sizeInBytes()) if not image.isNull() else 0
        self._tile_cache[key] = image
        self._tile_cache_bytes += image_cost
        while (
            len(self._tile_cache) > self.CACHE_CAPACITY
            or self._tile_cache_bytes > self.cache_byte_capacity
        ):
            _, evicted = self._tile_cache.popitem(last=False)
            if not evicted.isNull():
                self._tile_cache_bytes -= int(evicted.sizeInBytes())
        return image

    @staticmethod
    def _owned_indexed_image(
        indexes: np.ndarray, colour_table
    ) -> QImage:
        contiguous = np.ascontiguousarray(indexes, dtype=np.uint8)
        height, width = contiguous.shape
        image = QImage(
            contiguous.data,
            width,
            height,
            int(contiguous.strides[0]),
            QImage.Format.Format_Indexed8,
        ).copy()
        image.setColorTable(list(colour_table))
        return image

    @staticmethod
    def _owned_rgba_image(rgba: np.ndarray) -> QImage:
        height, width = rgba.shape[:2]
        return QImage(
            rgba.data,
            width,
            height,
            int(rgba.strides[0]),
            QImage.Format.Format_RGBA8888,
        ).copy()

    def _tile_pixmap(self, tile_x: int, tile_y: int) -> QPixmap:
        """Compatibility diagnostic; painting itself retains bounded QImages."""

        return QPixmap.fromImage(self.render_tile(tile_x, tile_y))

    def render_tile(self, tile_x: int, tile_y: int) -> QImage:
        """Expose an owned tile image for exact rendering diagnostics/tests."""

        image = self._tile_image(tile_x, tile_y)
        if not image.isNull():
            return image.copy()
        height, width = self._labels.shape
        x0 = int(tile_x) * self.TILE_SIZE
        y0 = int(tile_y) * self.TILE_SIZE
        tile_width = max(0, min(width, x0 + self.TILE_SIZE) - max(0, x0))
        tile_height = max(0, min(height, y0 + self.TILE_SIZE) - max(0, y0))
        if not tile_width or not tile_height:
            return QImage()
        transparent = QImage(
            tile_width,
            tile_height,
            QImage.Format.Format_RGBA8888,
        )
        transparent.fill(Qt.GlobalColor.transparent)
        return transparent

    def paint(self, painter, option, widget=None) -> None:  # noqa: ARG002
        visible = option.exposedRect.intersected(self.boundingRect())
        if visible.isEmpty():
            return
        left = max(0.0, visible.left())
        top = max(0.0, visible.top())
        right = min(float(self._labels.shape[1]), visible.right())
        bottom = min(float(self._labels.shape[0]), visible.bottom())
        if right <= left or bottom <= top:
            return
        first_x = max(0, int(np.floor(left / self.TILE_SIZE)))
        first_y = max(0, int(np.floor(top / self.TILE_SIZE)))
        last_x = int(np.ceil(right / self.TILE_SIZE))
        last_y = int(np.ceil(bottom / self.TILE_SIZE))

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        for tile_y in range(first_y, last_y):
            for tile_x in range(first_x, last_x):
                image = self._tile_image(tile_x, tile_y)
                if image.isNull():
                    continue
                painter.drawImage(
                    QPointF(
                        float(tile_x * self.TILE_SIZE),
                        float(tile_y * self.TILE_SIZE),
                    ),
                    image,
                )
        painter.restore()


class ImageView(QGraphicsView):
    """Display a laboratory image with smooth pan, zoom, fit, and file dropping."""

    image_dropped = Signal(str)
    reference_paint_finished = Signal()
    reference_mask_edited = Signal(str, object)
    instance_annotations_edited = Signal(object)
    instance_tool_status = Signal(str)
    shape_fill_size_preference_changed = Signal(float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._image_item = None
        self._image_path: Path | None = None
        self._source_pixmap: QPixmap | None = None
        self._corrected_pixmap: QPixmap | None = None
        self._corrected_base_key: tuple[int, tuple[int, ...], tuple[int, ...]] | None = None
        self._gamut_pixmap: QPixmap | None = None
        self._gamut_base_key = None
        self._prototype_collage_pixmap: QPixmap | None = None
        self._prototype_collage_key = None
        self._hsv_gamut_value = 0.75
        self._colour_gamut_parameters: dict[str, dict[str, object]] = {
            "background": {},
            "foreground": {},
        }
        self._displayed_base = "source"
        self._overlay_items = []
        self._analysis_result = None
        self._overlay_mode = "proposals"
        self._overlay_opacity = 0.68
        self._zoom_steps = 0
        self._fit_pending = False
        self._background_reference_mask: np.ndarray | None = None
        self._foreground_reference_mask: np.ndarray | None = None
        self._background_exclusion_mask: np.ndarray | None = None
        self._foreground_exclusion_mask: np.ndarray | None = None
        self._physical_edge_reference_mask: np.ndarray | None = None
        self._non_edge_reference_mask: np.ndarray | None = None
        self._material_reference_annotations_visible = True
        # Retired manual Physical-edge/Non-edge masks can still be restored
        # from an older reference archive, but they are deliberately never
        # presented or edited.  Boundary supervision now comes from complete
        # labelled seed instances.
        self._boundary_reference_annotations_visible = False
        self._instance_annotations_visible = True
        # Compatibility summary for callers that still treat the two binary
        # reference groups as one layer. Instance annotations have always been
        # independent of this flag.
        self._reference_annotations_visible = True
        self._instance_annotations: np.ndarray | None = None
        self._instance_bounds_cache: dict[
            int, tuple[int, int, int, int] | None
        ] = {}
        self._active_instance_id = 1
        self._show_selected_instance_only = False
        self._instance_annotation_tool = "brush"
        self._edge_trace_options = EdgeTraceOptions()
        self._smart_fill_options = SmartFillOptions()
        self._smart_fill_edge_source = "adaptive"
        self._shape_guided_fill_options = ShapeGuidedFillOptions()
        self._shape_guided_edge_source = "adaptive"
        self._instance_tool_points: list[QPointF] = []
        self._annotation_evidence_cache: dict[str, np.ndarray] = {}
        self._instance_trace_anchor: QPointF | None = None
        self._instance_trace_geometry: np.ndarray | None = None
        self._instance_preview_geometry: np.ndarray | None = None
        self._instance_preview_region: SmartFillRegion | None = None
        self._instance_shape_guided_region: ShapeGuidedFillRegion | None = None
        self._instance_preview_endpoint: QPointF | None = None
        self._instance_preview_point: QPointF | None = None
        self._pending_instance_preview_point: QPointF | None = None
        self._instance_preview_items = []
        self._instance_annotation_overlay_item = None
        self._instance_live_stroke_path: QPainterPath | None = None
        self._instance_live_stroke_item = None
        self._reference_live_stroke_path: QPainterPath | None = None
        self._reference_live_stroke_item = None
        self._instance_preview_timer = QTimer(self)
        self._instance_preview_timer.setSingleShot(True)
        self._instance_preview_timer.setInterval(35)
        self._instance_preview_timer.timeout.connect(
            self._update_pending_instance_preview
        )
        self._gamut_render_timer = QTimer(self)
        self._gamut_render_timer.setSingleShot(True)
        self._gamut_render_timer.setInterval(90)
        self._gamut_render_timer.timeout.connect(self._render_analysis)
        self._reference_point_mode: str | None = None
        self._reference_brush_radius = 12.0
        self._reference_erase_enabled = False
        self._edge_reference_snap_enabled = True
        self._edge_reference_snap_strength = 0.70
        self._reference_paint_button: Qt.MouseButton | None = None
        self._reference_stroke_erases = False
        self._last_reference_paint_point: QPointF | None = None
        self._last_reference_hover_point: QPointF | None = None
        self._reference_brush_outline_item = None
        self._context_panel: QFrame | None = None
        self._context_panel_drag_handle = None
        self._context_panel_user_position: QPoint | None = None
        self._context_panel_drag_global: QPointF | None = None
        self._context_panel_drag_origin: QPoint | None = None

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
        self._build_instance_continuity_warning()
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

    def _build_instance_continuity_warning(self) -> None:
        """Create the non-scrolling warning shown over the image pane."""

        self.instance_continuity_warning_banner = QLabel(self)
        self.instance_continuity_warning_banner.setObjectName(
            "instanceContinuityWarningBanner"
        )
        self.instance_continuity_warning_banner.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )
        self.instance_continuity_warning_banner.setWordWrap(True)
        self.instance_continuity_warning_banner.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents
        )
        self.instance_continuity_warning_banner.setStyleSheet(
            "QLabel#instanceContinuityWarningBanner {"
            " color: #fffafa;"
            " background-color: rgba(176, 24, 24, 238);"
            " border: 2px solid #ff7777;"
            " border-radius: 6px;"
            " padding: 6px 10px;"
            " font-weight: 700;"
            " font-size: 13px;"
            "}"
        )
        self.instance_continuity_warning_banner.hide()

    def set_instance_continuity_warning(
        self, text: str, *, tool_tip: str = ""
    ) -> None:
        """Synchronize the prominent image-pane continuity warning."""

        message = str(text).strip()
        banner = self.instance_continuity_warning_banner
        changed = (
            banner.text() != message
            or banner.toolTip() != str(tool_tip)
            or banner.isHidden() == bool(message)
        )
        banner.setText(message)
        banner.setToolTip(str(tool_tip))
        banner.setVisible(bool(message))
        if not changed:
            return
        self._layout_instance_continuity_warning()
        self._layout_context_panel()
        if self._context_panel is not None:
            self._context_panel.raise_()
        if message:
            banner.raise_()
        self.zoom_controls.raise_()

    def _layout_instance_continuity_warning(self) -> None:
        banner = self.instance_continuity_warning_banner
        if banner.isHidden():
            return
        available_width = max(80, self.width() - 24)
        width = min(960, available_width)
        flags = Qt.TextFlag.TextWordWrap | Qt.AlignmentFlag.AlignCenter
        text_bounds = banner.fontMetrics().boundingRect(
            0,
            0,
            max(40, width - 24),
            160,
            flags,
            banner.text(),
        )
        height = max(36, min(96, text_bounds.height() + 18))
        x = max(0, (self.width() - width) // 2)
        y = self.zoom_controls.height() + 6
        banner.setGeometry(x, y, width, height)

    def _context_panel_minimum_y(self) -> int:
        minimum_y = self.zoom_controls.height()
        if not self.instance_continuity_warning_banner.isHidden():
            minimum_y = max(
                minimum_y,
                self.instance_continuity_warning_banner.geometry().bottom() + 8,
            )
        return minimum_y

    def set_context_panel(self, panel: QFrame) -> None:
        """Attach a contextual control panel above the image viewport."""

        panel.setParent(self)
        self._context_panel = panel
        self._layout_context_panel()
        self.instance_continuity_warning_banner.raise_()
        self.zoom_controls.raise_()

    def set_context_panel_drag_handle(self, handle) -> None:
        """Let a dedicated handle reposition the contextual panel."""

        if self._context_panel_drag_handle is not None:
            self._context_panel_drag_handle.removeEventFilter(self)
        self._context_panel_drag_handle = handle
        handle.installEventFilter(self)
        handle.setCursor(Qt.CursorShape.OpenHandCursor)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt override
        if watched is self._context_panel_drag_handle and self._context_panel is not None:
            if (
                event.type() == QEvent.Type.MouseButtonPress
                and event.button() == Qt.MouseButton.LeftButton
            ):
                self._context_panel_drag_global = event.globalPosition()
                self._context_panel_drag_origin = self._context_panel.pos()
                watched.setCursor(Qt.CursorShape.ClosedHandCursor)
                event.accept()
                return True
            if (
                event.type() == QEvent.Type.MouseMove
                and self._context_panel_drag_global is not None
                and event.buttons() & Qt.MouseButton.LeftButton
            ):
                delta = (event.globalPosition() - self._context_panel_drag_global).toPoint()
                self._move_context_panel_to(
                    self._context_panel_drag_origin + delta,
                    remember=True,
                )
                event.accept()
                return True
            if event.type() == QEvent.Type.MouseButtonRelease:
                self._context_panel_drag_global = None
                self._context_panel_drag_origin = None
                watched.setCursor(Qt.CursorShape.OpenHandCursor)
                event.accept()
                return True
        return super().eventFilter(watched, event)

    def set_context_panel_visible(self, visible: bool) -> None:
        if self._context_panel is None:
            return
        self._context_panel.setVisible(visible)
        if visible:
            self._layout_context_panel()
            self._context_panel.raise_()
            self.instance_continuity_warning_banner.raise_()
            self.zoom_controls.raise_()

    def _layout_context_panel(self) -> None:
        if self._context_panel is None:
            return
        hint = self._context_panel.sizeHint()
        available_width = max(240, self.width() - 24)
        minimum_y = self._context_panel_minimum_y()
        available_height = max(140, self.height() - minimum_y - 12)
        width = min(max(360, hint.width()), available_width)
        height = min(hint.height(), available_height)
        self._context_panel.resize(width, height)
        requested = self._context_panel_user_position or QPoint(12, 46)
        self._move_context_panel_to(requested, remember=False)

    def _move_context_panel_to(
        self, position: QPoint, *, remember: bool
    ) -> None:
        if self._context_panel is None:
            return
        minimum_y = self._context_panel_minimum_y()
        maximum_x = max(0, self.width() - self._context_panel.width())
        maximum_y = max(minimum_y, self.height() - self._context_panel.height())
        clamped = QPoint(
            max(0, min(int(position.x()), maximum_x)),
            max(minimum_y, min(int(position.y()), maximum_y)),
        )
        self._context_panel.move(clamped)
        if remember:
            self._context_panel_user_position = clamped

    @property
    def image_path(self) -> Path | None:
        return self._image_path

    @property
    def image_size(self) -> tuple[int, int] | None:
        if self._image_item is None:
            return None
        # A diagnostic chart temporarily replaces the displayed pixmap, but
        # editable masks remain in corrected-image coordinates.
        pixmap = (
            self._corrected_pixmap or self._source_pixmap
            if self._displayed_base in {"gamut", "prototype_collage"}
            else self._image_item.pixmap()
        )
        if pixmap is None:
            return None
        return pixmap.width(), pixmap.height()

    def _show_placeholder(self) -> None:
        self.set_instance_continuity_warning("")
        self._scene.clear()
        self._image_item = None
        self._overlay_items = []
        self._analysis_result = None
        self._source_pixmap = None
        self._corrected_pixmap = None
        self._corrected_base_key = None
        self._gamut_pixmap = None
        self._gamut_base_key = None
        self._prototype_collage_pixmap = None
        self._prototype_collage_key = None
        self._displayed_base = "source"
        self._reference_brush_outline_item = None
        self._instance_annotation_overlay_item = None
        self._instance_preview_items = []
        self._instance_live_stroke_path = None
        self._instance_live_stroke_item = None
        self._reference_live_stroke_path = None
        self._reference_live_stroke_item = None
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
        self.set_instance_continuity_warning("")
        self._source_pixmap = pixmap
        self._corrected_pixmap = None
        self._corrected_base_key = None
        self._gamut_pixmap = None
        self._gamut_base_key = None
        self._prototype_collage_pixmap = None
        self._prototype_collage_key = None
        self._displayed_base = "source"
        release_host_caches(self._analysis_result)
        self._scene.clear()
        self._overlay_items = []
        self._analysis_result = None
        self._background_reference_mask = None
        self._foreground_reference_mask = None
        self._background_exclusion_mask = None
        self._foreground_exclusion_mask = None
        self._physical_edge_reference_mask = None
        self._non_edge_reference_mask = None
        self._instance_annotations = None
        self._instance_bounds_cache.clear()
        self._instance_tool_points.clear()
        self._annotation_evidence_cache.clear()
        self._reference_brush_outline_item = None
        self._instance_annotation_overlay_item = None
        self._instance_preview_items = []
        self._instance_live_stroke_path = None
        self._instance_live_stroke_item = None
        self._reference_live_stroke_path = None
        self._reference_live_stroke_item = None
        self._instance_trace_anchor = None
        self._instance_trace_geometry = None
        self._instance_preview_geometry = None
        self._instance_preview_region = None
        self._instance_shape_guided_region = None
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
        release_host_caches(self._analysis_result)
        self._analysis_result = None
        self._annotation_evidence_cache.clear()
        self._corrected_pixmap = None
        self._corrected_base_key = None
        self._restore_source_image()

    def _clear_overlay_items(self) -> None:
        for item in self._overlay_items:
            self._scene.removeItem(item)
        self._overlay_items.clear()
        self._instance_annotation_overlay_item = None

    def show_analysis(self, result, *, render: bool = True) -> None:
        """Store an analysis result and render the selected overlay layer."""

        if self._analysis_result is not result:
            release_host_caches(self._analysis_result)
            self._corrected_pixmap = None
            self._corrected_base_key = None
            self._gamut_pixmap = None
            self._gamut_base_key = None
            self._prototype_collage_pixmap = None
            self._prototype_collage_key = None
            self._displayed_base = "stale"
        self._analysis_result = result
        self._annotation_evidence_cache.clear()
        if render:
            self._render_analysis()

    def refresh_analysis(self) -> None:
        """Render the current base, overlay, and annotations once."""

        self._render_analysis()

    def set_background_point_editing(self, enabled: bool) -> None:
        self._set_reference_point_mode("background" if enabled else None)

    def set_reference_masks(
        self,
        background_mask: np.ndarray | None,
        foreground_mask: np.ndarray | None,
        background_exclusion_mask: np.ndarray | None = None,
        foreground_exclusion_mask: np.ndarray | None = None,
        *,
        physical_edge_mask: np.ndarray | None = None,
        non_edge_mask: np.ndarray | None = None,
        copy: bool = True,
        render: bool = True,
        normalize_material: bool = True,
    ) -> None:
        """Display editable full-resolution binary masks in image coordinates."""

        background = self._normalized_reference_mask(
            background_mask, copy=copy
        )
        foreground = self._normalized_reference_mask(
            foreground_mask, copy=copy
        )
        background_other = self._normalized_reference_mask(
            background_exclusion_mask, copy=copy
        )
        foreground_other = self._normalized_reference_mask(
            foreground_exclusion_mask, copy=copy
        )
        present = next(
            (
                value
                for value in (background, foreground, background_other, foreground_other)
                if value is not None
            ),
            None,
        )
        if present is None:
            self._background_reference_mask = None
            self._foreground_reference_mask = None
            self._background_exclusion_mask = None
            self._foreground_exclusion_mask = None
        else:
            empty = np.zeros(present.shape, dtype=bool)
            background_values = empty if background is None else background
            foreground_values = empty if foreground is None else foreground
            if background_other is None:
                other = empty if foreground_other is None else foreground_other
            elif foreground_other is None or foreground_other is background_other:
                other = background_other
            else:
                other = background_other | foreground_other
            # MainWindow normally supplies an already-exclusive categorical
            # layer. Preserve those arrays rather than allocating three new
            # full-resolution masks every time the display is refreshed. The
            # slower normalization path also accepts old four-mask sessions.
            conflicts = normalize_material and bool(
                np.any(background_values & foreground_values)
                or np.any(background_values & other)
                or np.any(foreground_values & other)
            )
            if conflicts:
                foreground_values = foreground_values & ~other
                background_values = background_values & ~other & ~foreground_values
            self._background_reference_mask = background_values
            self._foreground_reference_mask = foreground_values
            self._background_exclusion_mask = other
            self._foreground_exclusion_mask = other
        self._physical_edge_reference_mask = self._normalized_reference_mask(
            physical_edge_mask, copy=copy
        )
        self._non_edge_reference_mask = self._normalized_reference_mask(
            non_edge_mask, copy=copy
        )
        if render:
            self._render_analysis()

    def reference_mask(
        self, class_name: str, *, copy: bool = True
    ) -> np.ndarray | None:
        masks = {
            "background": self._background_reference_mask,
            "foreground": self._foreground_reference_mask,
            "background_exclusion": self._background_exclusion_mask,
            "foreground_exclusion": self._foreground_exclusion_mask,
            "physical_edge": self._physical_edge_reference_mask,
            "non_edge": self._non_edge_reference_mask,
        }
        if class_name == "other":
            mask = self._background_exclusion_mask
            return None if mask is None else mask.copy() if copy else mask
        if class_name not in masks:
            raise ValueError(f"Unknown reference-mask class {class_name!r}.")
        mask = masks[class_name]
        return None if mask is None else mask.copy() if copy else mask

    def set_reference_annotations_visible(self, visible: bool) -> None:
        """Show or hide the user-facing material-reference annotations.

        Retained for compatibility with the former single ``Show marks``
        control. Retired legacy edge/non-edge masks remain hidden.
        """

        visible = bool(visible)
        self._material_reference_annotations_visible = visible
        self._reference_annotations_visible = visible
        self._render_analysis()

    def set_material_reference_annotations_visible(self, visible: bool) -> None:
        """Change display of Background/Foreground/Other marks only."""

        self._material_reference_annotations_visible = bool(visible)
        self._sync_combined_reference_visibility()
        self._render_analysis()

    def set_boundary_reference_annotations_visible(self, visible: bool) -> None:
        """Keep retired Physical-edge/Non-edge masks hidden.

        This compatibility entry point intentionally ignores ``visible`` so
        old callers cannot resurrect the removed painting layer.
        """

        del visible
        self._boundary_reference_annotations_visible = False
        self._sync_combined_reference_visibility()
        self._render_analysis()

    def set_instance_annotations_visible(self, visible: bool) -> None:
        """Change display of labelled seed instances without editing them."""

        visible = bool(visible)
        if visible == self._instance_annotations_visible:
            return
        self._instance_annotations_visible = visible
        # Unlike the two binary groups, instance painting is one tracked
        # pixmap, so this avoids rebuilding the selected analysis overlay.
        self._refresh_instance_annotation_overlay()

    def _sync_combined_reference_visibility(self) -> None:
        self._reference_annotations_visible = bool(
            self._material_reference_annotations_visible
        )

    def _normalized_reference_mask(
        self, mask: np.ndarray | None, *, copy: bool
    ) -> np.ndarray | None:
        if mask is None:
            return None
        image_size = self.image_size
        if image_size is None:
            values = np.asarray(mask, dtype=bool)
            return values.copy() if copy else values
        width, height = image_size
        values = np.asarray(mask, dtype=bool)
        if values.shape != (height, width):
            raise ValueError(
                f"Reference mask shape {values.shape} does not match image {(height, width)}."
            )
        return values.copy() if copy else values

    def set_foreground_point_editing(self, enabled: bool) -> None:
        self._set_reference_point_mode("foreground" if enabled else None)

    def set_background_exclusion_editing(self, enabled: bool) -> None:
        self._set_reference_point_mode("other" if enabled else None)

    def set_foreground_exclusion_editing(self, enabled: bool) -> None:
        self._set_reference_point_mode("other" if enabled else None)

    def set_other_reference_editing(self, enabled: bool) -> None:
        self._set_reference_point_mode("other" if enabled else None)

    def set_physical_edge_reference_editing(self, enabled: bool) -> None:
        self._set_reference_point_mode("physical_edge" if enabled else None)

    def set_non_edge_reference_editing(self, enabled: bool) -> None:
        self._set_reference_point_mode("non_edge" if enabled else None)

    def set_edge_reference_snap(self, enabled: bool) -> None:
        self._edge_reference_snap_enabled = bool(enabled)

    def set_edge_reference_snap_strength(self, strength: float) -> None:
        self._edge_reference_snap_strength = max(0.0, min(1.0, float(strength)))
        if self._last_reference_hover_point is not None:
            self._update_reference_brush_outline(self._last_reference_hover_point)

    def set_instance_annotations(
        self,
        annotations: np.ndarray | None,
        *,
        copy: bool = True,
        render: bool = True,
    ) -> None:
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
            self._instance_annotations = values.astype(np.uint16, copy=copy)
        self._instance_bounds_cache.clear()
        self._instance_trace_anchor = None
        self._instance_trace_geometry = None
        self._clear_instance_preview()
        self._clear_instance_live_stroke()
        if render:
            self._render_analysis()
        self._schedule_instance_preview(self._last_reference_hover_point)

    def instance_annotations(self) -> np.ndarray | None:
        labels = self._instance_annotations
        return None if labels is None else labels.copy()

    def set_instance_annotation_editing(self, enabled: bool) -> None:
        self._set_reference_point_mode("instance" if enabled else None)

    def set_instance_annotation_tool(self, tool: str) -> None:
        """Choose the brush, eraser, or image-aware instance tool."""

        if tool not in {
            "brush",
            "edge_trace",
            "smart_fill",
            "shape_guided_fill",
            "eraser",
        }:
            raise ValueError(f"Unknown instance annotation tool {tool!r}.")
        self._instance_annotation_tool = tool
        self._instance_tool_points.clear()
        self._instance_trace_anchor = None
        self._instance_trace_geometry = None
        self._clear_instance_preview()
        self._clear_instance_live_stroke()
        self.set_reference_erase_mode(tool == "eraser")

    def set_edge_trace_options(self, options: EdgeTraceOptions) -> None:
        self._edge_trace_options = options
        if self._instance_preview_point is not None:
            self._schedule_instance_preview(self._last_reference_hover_point)

    def set_smart_fill_options(
        self,
        options: SmartFillOptions,
        *,
        edge_source: str | None = None,
    ) -> None:
        selected_source = options.edge_source if edge_source is None else str(edge_source)
        allowed_sources = {
            "adaptive",
            "ridges",
            "reference_ridges",
            "traces",
            "magnitude",
            "physical",
            "net_physical",
        }
        if selected_source not in allowed_sources:
            raise ValueError(f"Unknown annotation edge source {selected_source!r}.")
        core_source = (
            "physical" if selected_source == "net_physical" else selected_source
        )
        if options.edge_source != core_source:
            options = replace(options, edge_source=core_source)
        self._smart_fill_options = options
        self._smart_fill_edge_source = selected_source
        if self._instance_preview_point is not None:
            self._schedule_instance_preview(self._last_reference_hover_point)

    def set_shape_guided_fill_options(
        self,
        options: ShapeGuidedFillOptions,
        *,
        edge_source: str = "adaptive",
    ) -> None:
        if edge_source not in {
            "adaptive",
            "ridges",
            "reference_ridges",
            "traces",
            "magnitude",
            "physical",
            "net_physical",
        }:
            raise ValueError(f"Unknown annotation edge source {edge_source!r}.")
        self._shape_guided_fill_options = options
        self._shape_guided_edge_source = edge_source
        if self._instance_preview_point is not None:
            self._schedule_instance_preview(self._last_reference_hover_point)

    def set_active_instance_id(self, identifier: int) -> None:
        identifier = int(identifier)
        if not 1 <= identifier <= np.iinfo(np.uint16).max:
            raise ValueError("Seed instance IDs must be between 1 and 65,535.")
        changed = identifier != self._active_instance_id
        self._active_instance_id = identifier
        self._instance_trace_anchor = None
        self._instance_trace_geometry = None
        self._clear_instance_preview()
        self._clear_instance_live_stroke()
        if changed:
            if self._show_selected_instance_only:
                self._refresh_instance_annotation_overlay()
            self.focus_instance(identifier)
        if self._last_reference_hover_point is not None:
            self._update_reference_brush_outline(self._last_reference_hover_point)

    def set_show_selected_instance_only(self, enabled: bool) -> None:
        """Filter the annotation overlay to the active seed identity."""

        enabled = bool(enabled)
        if enabled == self._show_selected_instance_only:
            return
        self._show_selected_instance_only = enabled
        self._refresh_instance_annotation_overlay()
        if enabled:
            self.focus_instance(self._active_instance_id)

    def focus_instance(self, identifier: int | None = None) -> bool:
        """Centre the existing view on one annotated seed without changing zoom."""

        labels = self._instance_annotations
        target = self._active_instance_id if identifier is None else int(identifier)
        if labels is None:
            return False
        bounds = self._instance_bounds(target)
        if bounds is None:
            return False
        x0, y0, x1, y1 = bounds
        centre_x = (float(x0) + float(x1 - 1)) * 0.5
        centre_y = (float(y0) + float(y1 - 1)) * 0.5
        self.centerOn(centre_x, centre_y)
        return True

    def _instance_bounds(
        self, identifier: int
    ) -> tuple[int, int, int, int] | None:
        """Return a cached exclusive bbox without allocating full-image indices."""

        identifier = int(identifier)
        if identifier in self._instance_bounds_cache:
            return self._instance_bounds_cache[identifier]
        labels = self._instance_annotations
        if labels is None:
            self._instance_bounds_cache[identifier] = None
            return None
        minimum_x = labels.shape[1]
        minimum_y = labels.shape[0]
        maximum_x = -1
        maximum_y = -1
        for row_start in range(0, labels.shape[0], 512):
            row_end = min(labels.shape[0], row_start + 512)
            rows, columns = np.nonzero(labels[row_start:row_end] == identifier)
            if not len(rows):
                continue
            minimum_x = min(minimum_x, int(columns.min()))
            maximum_x = max(maximum_x, int(columns.max()))
            minimum_y = min(minimum_y, row_start + int(rows.min()))
            maximum_y = max(maximum_y, row_start + int(rows.max()))
        bounds = (
            None
            if maximum_x < 0
            else (minimum_x, minimum_y, maximum_x + 1, maximum_y + 1)
        )
        self._instance_bounds_cache[identifier] = bounds
        return bounds

    def set_reference_brush_radius(self, radius: float) -> None:
        """Set the corrected-image radius of the visibly painted sample area."""

        self._reference_brush_radius = max(2.0, float(radius))
        if self._last_reference_hover_point is not None:
            self._update_reference_brush_outline(
                self._last_reference_hover_point
            )
        # The brush cursor and next committed stroke are sufficient; mask
        # overlays do not depend on the configured brush radius.

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
            self._instance_trace_geometry = None
            self._clear_instance_preview()
            self._clear_instance_live_stroke()
            self._clear_reference_live_stroke()
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
        display_point = scene_point
        if (
            self._reference_point_mode in {"physical_edge", "non_edge"}
            and self._edge_reference_snap_enabled
        ):
            # This is also the live preview of the position that the next
            # physical-edge dab will commit.
            display_point = self._snap_reference_edge_point(scene_point)
        radius = self._reference_brush_radius
        if self._reference_brush_outline_item is None:
            self._reference_brush_outline_item = self._scene.addEllipse(
                display_point.x() - radius,
                display_point.y() - radius,
                radius * 2.0,
                radius * 2.0,
            )
            self._reference_brush_outline_item.setZValue(1000.0)
            self._reference_brush_outline_item.setBrush(Qt.BrushStyle.NoBrush)
        else:
            self._reference_brush_outline_item.setRect(
                display_point.x() - radius,
                display_point.y() - radius,
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
            else QColor("#96969c")
            if self._reference_point_mode == "other"
            else QColor("#ffd137")
            if self._reference_point_mode == "physical_edge"
            else QColor("#5ca0ff")
            if self._reference_point_mode == "non_edge"
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
            "colour_probability:",
            "pattern_probability:",
        )):
            raise ValueError(f"Unknown viewer overlay: {mode}")
        if mode == self._overlay_mode:
            return
        full_pane_modes = {
            "background_colour_gamut",
            "foreground_colour_gamut",
            "reference_texture_prototypes",
        }
        was_full_pane = self._overlay_mode in full_pane_modes
        self._overlay_mode = mode
        self._render_analysis()
        if was_full_pane or mode in full_pane_modes:
            QTimer.singleShot(0, self.fit_image)

    def set_colour_gamut_parameters(
        self, class_name: str, parameters: dict[str, object]
    ) -> None:
        if class_name not in self._colour_gamut_parameters:
            raise ValueError(f"Unknown colour-gamut class {class_name!r}.")
        values = dict(parameters)
        if values == self._colour_gamut_parameters[class_name]:
            return
        self._colour_gamut_parameters[class_name] = values
        self._gamut_base_key = None
        if self._overlay_mode == f"{class_name}_colour_gamut":
            self._render_analysis()

    def set_hsv_gamut_value(self, value: float) -> None:
        value = max(0.0, min(1.0, float(value)))
        if abs(value - self._hsv_gamut_value) < 1e-6:
            return
        self._hsv_gamut_value = value
        self._gamut_base_key = None
        if self._overlay_mode in {
            "background_colour_gamut",
            "foreground_colour_gamut",
        }:
            # Slider drags can emit dozens of values. Coalesce them so a dense
            # multimodal foreground fit renders only the final requested slice.
            self._gamut_render_timer.start()

    def dominant_hsv_gamut_value(self, class_name: str) -> float | None:
        """Return the brightness of the strongest fitted colour mode."""

        if class_name not in self._colour_gamut_parameters:
            raise ValueError(f"Unknown colour-gamut class {class_name!r}.")
        if self._analysis_result is None:
            return None
        profile = getattr(
            self._analysis_result.layers,
            f"{class_name}_colour_profile",
            None,
        )
        from seedvision.ui.pipeline_inspector import BackgroundColourGamut

        return BackgroundColourGamut.dominant_hsv_value(profile)

    def set_overlay_opacity(self, opacity: float) -> None:
        self._overlay_opacity = max(0.0, min(1.0, float(opacity)))
        for item in self._overlay_items:
            if item.data(0) != "fixed-opacity-overlay-annotation":
                item.setOpacity(self._overlay_opacity)

    def _render_analysis(self) -> None:
        self._clear_overlay_items()
        result = self._analysis_result
        if result is None:
            return
        # Preserve at most the CPU mirrors needed by the currently displayed
        # overlay. Previous overlay downloads have already been copied into Qt.
        release_host_caches(result)
        if self._overlay_mode == "raw_image":
            self._restore_source_image()
            return
        calibration = getattr(result, "calibration", None)
        if calibration is not None:
            self._set_bgr_base_image(calibration.corrected_bgr)
        else:
            self._restore_source_image()
        if self._overlay_mode in {
            "background_colour_gamut",
            "foreground_colour_gamut",
        }:
            self._render_colour_gamut(result)
            return
        if self._overlay_mode == "reference_texture_prototypes":
            self._render_reference_texture_collage(result)
            return
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
            self._render_background_starting_colour(result)
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

        if self._overlay_mode.startswith("procedural_"):
            procedural = getattr(result, "procedural_instances", None)
            if procedural is not None:
                if self._overlay_mode == "procedural_instances":
                    self._render_rgba_overlay(
                        procedural.instance_rgba(),
                        *result.crop_offset,
                        source_shape=procedural.source_shape,
                    )
                    self._render_procedural_centres(result)
                else:
                    raster = {
                        "procedural_seed_material": procedural.occupancy_likelihood,
                        "procedural_seed_mask": procedural.occupancy_mask,
                        "procedural_boundary_cost": procedural.boundary_cost,
                        "procedural_centres": procedural.centre_likelihood,
                        "procedural_confidence": procedural.confidence_raster(),
                    }[self._overlay_mode]
                    self._render_scalar_raster(
                        raster,
                        *result.crop_offset,
                        valid_mask=result.layers.valid_mask,
                        source_shape=procedural.source_shape,
                    )
                    if self._overlay_mode == "procedural_centres":
                        self._render_procedural_centres(result)
            self._render_context_annotations(result)
            return

        learned_name = (
            "unet_instances"
            if self._overlay_mode.startswith("unet_")
            else "stardist_instances"
            if self._overlay_mode.startswith("stardist_")
            else None
        )
        if learned_name is not None:
            learned = getattr(result, learned_name, None)
            if learned is not None:
                identity_mode = self._overlay_mode == learned_name
                confidence_mode = self._overlay_mode == learned_name.replace(
                    "instances", "confidence"
                )
                if identity_mode:
                    self._render_rgba_overlay(
                        learned.instance_rgba(), *result.crop_offset
                    )
                    self._render_learned_centres(result, learned)
                elif confidence_mode:
                    self._render_scalar_raster(
                        learned.confidence_raster(),
                        *result.crop_offset,
                        valid_mask=result.layers.valid_mask,
                    )
                else:
                    raster_key = {
                        "unet_interior": "interior",
                        "unet_physical_boundary": "physical_boundary",
                        "unet_pattern_boundary": "pattern_boundary",
                        "unet_centres": "centre",
                        "unet_distance": "distance",
                        "unet_uncertainty": "uncertainty",
                        "stardist_object_probability": "object_probability",
                        "stardist_radial_uncertainty": "radial_uncertainty",
                    }[self._overlay_mode]
                    self._render_scalar_raster(
                        learned.rasters[raster_key],
                        *result.crop_offset,
                        valid_mask=result.layers.valid_mask,
                    )
                    if self._overlay_mode == "unet_centres":
                        self._render_learned_centres(result, learned)
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
        if self._overlay_mode == "instance_masks":
            rgba = layers.instance_rgba()
        elif self._overlay_mode == "background_likelihood":
            rgba = layers.background_rgba()
        elif self._overlay_mode == "refined_background_likelihood":
            rgba = layers.refined_background_rgba()
        elif self._overlay_mode == "foreground_noise_likelihood":
            rgba = layers.foreground_noise_rgba()
        elif self._overlay_mode == "reference_seed_surface_probability":
            rgba = layers.reference_seed_surface_rgba()
        elif self._overlay_mode == "reference_background_texture_probability":
            rgba = layers.reference_material_probability_rgba("background")
        elif self._overlay_mode == "reference_other_texture_probability":
            rgba = layers.reference_material_probability_rgba("other")
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
        elif self._overlay_mode == "physical_edge_probability":
            rgba = layers.reference_edge_probability_rgba(True)
        elif self._overlay_mode == "non_edge_probability":
            rgba = layers.reference_edge_probability_rgba(False)
        elif self._overlay_mode == "reference_edge_comparison":
            rgba = layers.reference_edge_comparison_rgba()
        elif self._overlay_mode == "net_physical_edge_probability":
            rgba = layers.net_physical_edge_probability_rgba()
        elif self._overlay_mode == "reference_edge_ridges":
            rgba = layers.reference_edge_ridges_rgba()
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
            surrounding = layers.surrounding_background_rgba()
            if surrounding is not None:
                self._render_rgba_overlay(
                    surrounding,
                    layers.surrounding_noise_offset_x,
                    layers.surrounding_noise_offset_y,
                )
            self._render_background_sampling_band(result, fill=False)
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
        self,
        rgba: np.ndarray,
        offset_x: int,
        offset_y: int,
        *,
        source_shape: tuple[int, int] | None = None,
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
        if source_shape is not None and source_shape != (height, width):
            source_height, source_width = source_shape
            item.setTransform(
                QTransform.fromScale(source_width / width, source_height / height)
            )
        item.setOpacity(self._overlay_opacity)
        item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        item.setZValue(10)
        self._overlay_items.append(item)

    def _render_procedural_centres(self, result) -> None:
        procedural = getattr(result, "procedural_instances", None)
        if procedural is None:
            return
        offset_x, offset_y = result.crop_offset
        for index, (x, y) in enumerate(procedural.centres_xy):
            confidence = (
                float(procedural.instance_confidences[index])
                if index < len(procedural.instance_confidences)
                else 0.0
            )
            colour = QColor.fromHsvF(0.33 * confidence, 0.95, 1.0)
            pen = QPen(colour, 3.0)
            pen.setCosmetic(True)
            radius = 3.0 + 3.0 * confidence
            item = self._scene.addEllipse(
                float(x + offset_x - radius),
                float(y + offset_y - radius),
                radius * 2.0,
                radius * 2.0,
                pen,
            )
            item.setOpacity(self._overlay_opacity)
            item.setZValue(15)
            self._overlay_items.append(item)

    def _render_learned_centres(self, result, learned) -> None:
        offset_x, offset_y = result.crop_offset
        for index, (x, y) in enumerate(learned.centres_xy):
            confidence = float(learned.instance_confidences[index])
            colour = QColor.fromHsvF(0.33 * confidence, 0.95, 1.0)
            pen = QPen(colour, 3.0)
            pen.setCosmetic(True)
            radius = 3.0 + 3.0 * confidence
            item = self._scene.addEllipse(
                float(x + offset_x - radius),
                float(y + offset_y - radius),
                radius * 2.0,
                radius * 2.0,
                pen,
            )
            item.setOpacity(self._overlay_opacity)
            item.setZValue(15)
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
        source_shape: tuple[int, int] | None = None,
    ) -> None:
        values = np.asarray(raster)
        height, width = values.shape[:2]
        valid_values_u8 = np.asarray(valid_mask)
        if valid_values_u8.shape != (height, width):
            valid_values_u8 = cv2.resize(
                valid_values_u8,
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
        if values.dtype == np.uint8:
            gray = values
        else:
            valid_values = values[valid_values_u8 > 0]
            scale = float(np.percentile(valid_values, 99.0)) if valid_values.size else 1.0
            gray = np.uint8(np.clip(np.rint(values / max(scale, 1e-6) * 255), 0, 255))
        alpha = np.uint8(valid_values_u8 > 0) * 255
        rgba = np.dstack((gray, gray, gray, alpha))
        height, width = rgba.shape[:2]
        image = QImage(
            rgba.data, width, height, int(rgba.strides[0]), QImage.Format.Format_RGBA8888
        ).copy()
        item = self._scene.addPixmap(QPixmap.fromImage(image))
        item.setPos(offset_x, offset_y)
        if source_shape is not None and source_shape != (height, width):
            source_height, source_width = source_shape
            item.setTransform(
                QTransform.fromScale(source_width / width, source_height / height)
            )
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

    def _render_background_sampling_band(self, result, *, fill: bool = True) -> None:
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
        fill_colour = QColor(colour)
        fill_colour.setAlpha(105)
        pen = QPen(colour, 3)
        pen.setCosmetic(True)
        brush = QBrush(fill_colour) if fill else QBrush(Qt.BrushStyle.NoBrush)
        item = self._scene.addPath(path, pen, brush)
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
        scale_pen = QPen(QColor("#ffd84a"), 1)
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

        diameter_text = f"{diameter:.1f}".rstrip("0").rstrip(".") + " px"
        label = self._scene.addText(diameter_text)
        label_font = QFont(label.font())
        label_font.setPixelSize(max(12, min(36, round(diameter * 0.16))))
        label_font.setBold(True)
        label.setFont(label_font)
        label.setDefaultTextColor(QColor("#ffd84a"))
        label_bounds = label.boundingRect()
        label.setPos(
            centre.x() - label_bounds.width() * 0.5,
            circle.rect().bottom() + max(2.0, diameter * 0.03),
        )
        label.setOpacity(self._overlay_opacity)
        label.setZValue(11)
        label.setToolTip(
            f"Estimated reference seed diameter: {diameter:.1f} pixels."
        )
        self._overlay_items.append(label)

    def _render_context_annotations(
        self, result, *, include_scale: bool = False
    ) -> None:
        if include_scale:
            self._render_scale_bar(result)
        if self._material_reference_annotations_visible:
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
            (150, 150, 156),
            32,
        )

    def _render_edge_reference_masks(self) -> None:
        self._render_reference_mask(
            self._physical_edge_reference_mask,
            (255, 209, 55),
            34,
        )
        self._render_reference_mask(
            self._non_edge_reference_mask,
            (92, 160, 255),
            35,
        )

    @staticmethod
    def instance_colour(identifier: int) -> QColor:
        """Return a stable, high-contrast colour for a positive seed ID."""

        hue = (max(1, int(identifier)) * 0.61803398875) % 1.0
        return QColor.fromHsvF(hue, 0.78, 1.0)

    def _render_instance_annotations(self) -> None:
        existing = self._instance_annotation_overlay_item

        def remove_existing() -> None:
            if existing is None:
                return
            if isinstance(existing, _InstanceAnnotationTileItem):
                existing.invalidate_tiles()
            if existing in self._overlay_items:
                self._overlay_items.remove(existing)
            if existing.scene() is self._scene:
                self._scene.removeItem(existing)
            self._instance_annotation_overlay_item = None

        if not self._instance_annotations_visible:
            remove_existing()
            return
        labels = self._instance_annotations
        if labels is None:
            remove_existing()
            return
        selected_identifier: int | None = None
        render_bounds: tuple[int, int, int, int] | None = None
        if self._show_selected_instance_only:
            bounds = self._instance_bounds(self._active_instance_id)
            if bounds is None:
                remove_existing()
                return
            selected_x0, selected_y0, selected_x1, selected_y1 = bounds
            render_bounds = (
                max(0, selected_x0 - 1),
                max(0, selected_y0 - 1),
                min(labels.shape[1], selected_x1 + 1),
                min(labels.shape[0], selected_y1 + 1),
            )
            selected_identifier = self._active_instance_id

        if isinstance(existing, _InstanceAnnotationTileItem) and existing.scene() is self._scene:
            existing.replace_annotations(
                labels,
                selected_identifier=selected_identifier,
                render_bounds=render_bounds,
            )
            existing.setZValue(32)
            return

        remove_existing()
        item = _InstanceAnnotationTileItem(
            labels,
            colour_for_identifier=self.instance_colour,
            selected_identifier=selected_identifier,
            render_bounds=render_bounds,
        )
        item.setOpacity(self._overlay_opacity)
        item.setZValue(32)
        self._scene.addItem(item)
        self._overlay_items.append(item)
        self._instance_annotation_overlay_item = item

    def _render_background_starting_colour(self, result) -> None:
        """Show the exact median colour selected from the perimeter band."""

        median_lab = getattr(result, "perimeter_background_lab", None)
        image_size = self.image_size
        if median_lab is None or image_size is None:
            return
        lab_values = np.asarray(median_lab, dtype=np.float32)
        if lab_values.shape != (3,) or not np.isfinite(lab_values).all():
            return
        lab_u8 = np.clip(np.rint(lab_values), 0, 255).astype(np.uint8)[None, None]
        blue, green, red = (
            int(value)
            for value in cv2.cvtColor(lab_u8, cv2.COLOR_LAB2BGR)[0, 0]
        )
        colour = QColor(red, green, blue)
        colour_hex = colour.name(QColor.NameFormat.HexRgb).upper()
        width, height = image_size
        shortest = float(min(width, height))
        font_pixels = max(16, min(64, round(shortest * 0.014)))
        swatch_size = max(44.0, min(150.0, shortest * 0.055))
        padding = max(8.0, font_pixels * 0.38)
        gap = max(8.0, font_pixels * 0.34)
        margin = max(18.0, shortest * 0.025)
        label_text = f"Median starting background\n{colour_hex}"
        tooltip = (
            "Selected median starting background colour: "
            f"{colour_hex}; OpenCV Lab "
            f"{lab_values[0]:.1f}, {lab_values[1]:.1f}, {lab_values[2]:.1f}."
        )

        label = self._scene.addText(label_text)
        font = QFont(label.font())
        font.setPixelSize(font_pixels)
        font.setBold(True)
        label.setFont(font)
        label.setDefaultTextColor(QColor("#ffffff"))
        text_bounds = label.boundingRect()
        content_height = max(swatch_size, text_bounds.height())
        panel_width = padding * 2.0 + swatch_size + gap + text_bounds.width()
        panel_height = padding * 2.0 + content_height
        panel = self._scene.addRect(
            margin,
            margin,
            panel_width,
            panel_height,
            QPen(QColor(255, 255, 255, 185), 2.0),
            QBrush(QColor(16, 20, 24, 220)),
        )
        swatch = self._scene.addRect(
            margin + padding,
            margin + padding + (content_height - swatch_size) * 0.5,
            swatch_size,
            swatch_size,
            QPen(QColor("#ffffff"), 3.0),
            QBrush(colour),
        )
        label.setPos(
            margin + padding + swatch_size + gap,
            margin + padding + (content_height - text_bounds.height()) * 0.5,
        )
        for item, z_value in ((panel, 40.0), (swatch, 41.0), (label, 41.0)):
            # The swatch communicates an absolute colour. It must not blend
            # with the photograph when the raster-overlay opacity is changed.
            item.setData(0, "fixed-opacity-overlay-annotation")
            item.setOpacity(1.0)
            item.setZValue(z_value)
            item.setToolTip(tooltip)
            self._overlay_items.append(item)

    def _refresh_instance_annotation_overlay(self) -> None:
        self._instance_bounds_cache.clear()
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

    def _start_reference_live_stroke(
        self, scene_point: QPointF, *, erase: bool
    ) -> None:
        self._clear_reference_live_stroke()
        if (
            self._reference_point_mode in {"physical_edge", "non_edge"}
            and self._edge_reference_snap_enabled
        ):
            scene_point = self._snap_reference_edge_point(scene_point)
        colours = {
            "background": QColor(54, 241, 177, 185),
            "foreground": QColor(255, 118, 95, 185),
            "other": QColor(150, 150, 156, 185),
            "physical_edge": QColor(255, 209, 55, 210),
            "non_edge": QColor(92, 160, 255, 185),
        }
        colour = (
            QColor(255, 70, 85, 190)
            if erase
            else colours.get(self._reference_point_mode, QColor(255, 255, 255, 180))
        )
        pen = QPen(colour, max(2.0, self._reference_brush_radius * 2.0))
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        if erase:
            pen.setStyle(Qt.PenStyle.DashLine)
        path = QPainterPath(scene_point)
        path.lineTo(scene_point.x() + 0.01, scene_point.y())
        self._reference_live_stroke_path = path
        self._reference_live_stroke_item = self._scene.addPath(path, pen)
        self._reference_live_stroke_item.setZValue(1002.0)

    def _extend_reference_live_stroke(self, scene_point: QPointF) -> None:
        if (
            self._reference_live_stroke_path is None
            or self._reference_live_stroke_item is None
        ):
            return
        if (
            self._reference_point_mode in {"physical_edge", "non_edge"}
            and self._edge_reference_snap_enabled
        ):
            scene_point = self._snap_reference_edge_point(scene_point)
        self._reference_live_stroke_path.lineTo(scene_point)
        self._reference_live_stroke_item.setPath(self._reference_live_stroke_path)

    def _clear_reference_live_stroke(self) -> None:
        item = self._reference_live_stroke_item
        if item is not None and item.scene() is self._scene:
            self._scene.removeItem(item)
        self._reference_live_stroke_item = None
        self._reference_live_stroke_path = None

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
        self._image_item.setToolTip("")
        key = (id(bgr), tuple(bgr.shape), tuple(bgr.strides))
        if self._corrected_pixmap is None or self._corrected_base_key != key:
            contiguous = bgr if bgr.flags.c_contiguous else bgr.copy()
            height, width = contiguous.shape[:2]
            image = QImage(
                contiguous.data,
                width,
                height,
                int(contiguous.strides[0]),
                QImage.Format.Format_BGR888,
            ).copy()
            self._corrected_pixmap = QPixmap.fromImage(image)
            self._corrected_base_key = key
        if self._displayed_base != "corrected":
            self._image_item.setPixmap(self._corrected_pixmap)
            self._scene.setSceneRect(self._image_item.boundingRect())
            self._displayed_base = "corrected"

    def _render_colour_gamut(self, result) -> None:
        """Replace the image temporarily with a full-size HSV value slice."""

        from seedvision.ui.pipeline_inspector import BackgroundColourGamut

        if self._image_item is None:
            return
        class_name = (
            "foreground"
            if self._overlay_mode == "foreground_colour_gamut"
            else "background"
        )
        profile = getattr(
            result.layers, f"{class_name}_colour_profile", None
        )
        if profile is None:
            return
        parameters = self._colour_gamut_parameters[class_name]
        key = (
            id(profile),
            class_name,
            round(self._hsv_gamut_value, 3),
            tuple(sorted((str(key), repr(value)) for key, value in parameters.items())),
        )
        if self._gamut_pixmap is None or self._gamut_base_key != key:
            image, _probability, _centres = (
                BackgroundColourGamut.render_hsv_value_slice(
                    profile,
                    parameters,
                    class_name=class_name,
                    value=self._hsv_gamut_value,
                )
            )
            self._gamut_pixmap = QPixmap.fromImage(image)
            self._gamut_base_key = key
        self._image_item.setPixmap(self._gamut_pixmap)
        self._scene.setSceneRect(self._image_item.boundingRect())
        self._displayed_base = "gamut"

    def _render_reference_texture_collage(self, result) -> None:
        """Replace the image with every retained material and edge medoid."""

        if self._image_item is None:
            return
        profile = getattr(result.layers, "reference_texture_profile", None)
        if profile is None:
            return
        key = (
            id(profile),
            len(profile.prototypes),
            profile.patch_size_px,
            round(float(profile.edge_strip_normal_offset_px), 4),
            round(float(profile.edge_strip_tangent_half_length_px), 4),
        )
        if (
            self._prototype_collage_pixmap is None
            or self._prototype_collage_key != key
        ):
            class_details = (
                ("background", "Background", QColor("#5ab7ff")),
                ("foreground", "Foreground / seed surface", QColor("#49e6a7")),
                ("other", "Other material", QColor("#ffad55")),
                ("physical_edge", "Physical edge", QColor("#ffe25c")),
                (
                    "non_edge",
                    "Non-physical / internal edge",
                    QColor("#ba8cff"),
                ),
            )
            groups = {
                name: [
                    prototype
                    for prototype in profile.prototypes
                    if prototype.class_name == name
                ]
                for name, _label, _colour in class_details
            }
            columns = 14
            tile_width = 82
            tile_height = 98
            left = 30
            top = 104
            canvas_width = max(960, left * 2 + columns * tile_width)
            canvas_height = top + 30
            for name, _label, _colour in class_details:
                count = len(groups[name])
                rows = max(1, int(np.ceil(count / columns)))
                canvas_height += 34 + rows * tile_height + 18
            canvas_height = max(640, canvas_height)
            image = QImage(
                canvas_width, canvas_height, QImage.Format.Format_RGB32
            )
            image.fill(QColor("#20262d"))
            painter = QPainter(image)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            title_font = QFont("Segoe UI", 18)
            title_font.setBold(True)
            painter.setFont(title_font)
            painter.setPen(QColor("#f3f6f9"))
            painter.drawText(
                QRectF(20, 16, canvas_width - 40, 30),
                Qt.AlignmentFlag.AlignCenter,
                "Reference texture prototypes",
            )
            painter.setFont(QFont("Segoe UI", 10))
            painter.setPen(QColor("#c8d1da"))
            painter.drawText(
                QRectF(40, 47, canvas_width - 80, 43),
                Qt.AlignmentFlag.AlignHCenter
                | Qt.AlignmentFlag.AlignTop
                | Qt.TextFlag.TextWordWrap,
                "Thumbnails are visual context only; matching does not treat the "
                "whole square as a prototype. Edge descriptors sample the yellow "
                "centre line and two polarity-neutral cyan side strips; both side "
                "assignments are tested.",
            )
            sample_counts = dict(profile.class_sample_counts)
            y = top
            for class_name, label, colour in class_details:
                prototypes = groups[class_name]
                painter.setPen(colour)
                heading_font = QFont("Segoe UI", 12)
                heading_font.setBold(True)
                painter.setFont(heading_font)
                painter.drawText(
                    QRectF(left, y, canvas_width - left * 2, 26),
                    Qt.AlignmentFlag.AlignVCenter,
                    f"{label} — {len(prototypes)} prototypes from "
                    f"{sample_counts.get(class_name, 0):,} "
                    f"{profile.sample_count_unit_for(class_name)}",
                )
                y += 32
                painter.setFont(QFont("Segoe UI", 8))
                rows = max(1, int(np.ceil(len(prototypes) / columns)))
                for index, prototype in enumerate(prototypes):
                    column = index % columns
                    row = index // columns
                    x = left + column * tile_width
                    tile_y = y + row * tile_height
                    patch = prototype.patch_bgr
                    patch_image = QImage(
                        patch.data,
                        patch.shape[1],
                        patch.shape[0],
                        int(patch.strides[0]),
                        QImage.Format.Format_BGR888,
                    ).copy()
                    patch_rect = QRectF(x + 5, tile_y + 3, 68, 68)
                    painter.drawImage(patch_rect, patch_image)
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.setPen(QPen(colour, 2.0))
                    painter.drawRect(patch_rect)
                    if class_name in {"physical_edge", "non_edge"}:
                        centre_line, side_lines = _edge_strip_annotation_lines(
                            patch_rect,
                            patch.shape[:2],
                            normal_offset_px=(
                                profile.edge_strip_normal_offset_px
                            ),
                            tangent_half_length_px=(
                                profile.edge_strip_tangent_half_length_px
                            ),
                        )
                        painter.setPen(QPen(QColor("#38ddff"), 1.5))
                        for side_line in side_lines:
                            painter.drawLine(side_line)
                        painter.setPen(QPen(QColor("#fff06a"), 1.8))
                        painter.drawLine(centre_line)
                    painter.setPen(QColor("#e8edf2"))
                    painter.drawText(
                        QRectF(x, tile_y + 73, tile_width - 4, 18),
                        Qt.AlignmentFlag.AlignCenter,
                        f"{index + 1} · {prototype.weight:.0%}",
                    )
                if not prototypes:
                    painter.setPen(QColor("#7f8a95"))
                    painter.drawText(
                        QRectF(left + 5, y + 10, canvas_width - 80, 24),
                        "No applied references for this class.",
                    )
                y += rows * tile_height + 18
            painter.end()
            self._prototype_collage_pixmap = QPixmap.fromImage(image)
            self._prototype_collage_key = key
        self._image_item.setPixmap(self._prototype_collage_pixmap)
        self._image_item.setToolTip(
            "Prototype squares provide source-image context only; they are not "
            "matched as image patches. On tangent-aligned edge thumbnails, the "
            "yellow line marks the centre edge sample and the two cyan lines "
            "mark the polarity-neutral side samples. Matching evaluates both "
            "possible side assignments."
        )
        self._scene.setSceneRect(self._image_item.boundingRect())
        self._displayed_base = "prototype_collage"

    def _restore_source_image(self) -> None:
        if self._image_item is None or self._source_pixmap is None:
            return
        self._image_item.setToolTip("")
        if self._displayed_base != "source":
            self._image_item.setPixmap(self._source_pixmap)
            self._scene.setSceneRect(self._image_item.boundingRect())
            self._displayed_base = "source"

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
        if (
            self._reference_point_mode == "instance"
            and not self._reference_erase_enabled
            and self._instance_annotation_tool == "shape_guided_fill"
        ):
            wheel_delta = int(event.angleDelta().y())
            if wheel_delta:
                direction = 1 if wheel_delta > 0 else -1
                notch_count = max(1, abs(wheel_delta) // 120)
                preferred_scale = float(
                    np.clip(
                        self._shape_guided_fill_options.preferred_scale
                        + direction * 0.05 * notch_count,
                        0.40,
                        2.0,
                    )
                )
                # Quantization prevents accumulated floating-point noise from
                # making the displayed percentage drift after repeated notches.
                preferred_scale = round(preferred_scale * 20.0) / 20.0
                self._shape_guided_fill_options = replace(
                    self._shape_guided_fill_options,
                    preferred_scale=preferred_scale,
                )
                self.shape_fill_size_preference_changed.emit(preferred_scale)
                self.instance_tool_status.emit(
                    f"Shape fill oval size preference: {preferred_scale:.0%}."
                )
                self._schedule_instance_preview(self._last_reference_hover_point)
            event.accept()
            return
        direction = 1 if event.angleDelta().y() > 0 else -1
        self._zoom_by(1.2 if direction > 0 else 1 / 1.2)
        self._zoom_steps += direction

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self.zoom_controls.setGeometry(0, 0, max(120, self.width()), 38)
        self._layout_instance_continuity_warning()
        self._layout_context_panel()
        if self._context_panel is not None:
            self._context_panel.raise_()
        self.instance_continuity_warning_banner.raise_()
        self.zoom_controls.raise_()

    def _instance_assisted_tool_active(self) -> bool:
        return (
            self._reference_point_mode == "instance"
            and not self._reference_erase_enabled
            and self._instance_annotation_tool
            in {"edge_trace", "smart_fill", "shape_guided_fill"}
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
        self._instance_preview_region = None
        self._instance_shape_guided_region = None
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
        # Produce the first hover result immediately so entering an assisted
        # tool never shows a blank cursor while the debounce interval elapses.
        # Subsequent moves remain coalesced by the timer for interactive speed.
        if (
            self._instance_preview_point is None
            and self._pending_instance_preview_point is None
        ):
            self._update_instance_assisted_preview(QPointF(scene_point))
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

    def _trace_closure_radius(self) -> float:
        """Return a small full-resolution snap radius for the first anchor."""

        return max(
            3.0,
            min(10.0, float(self._edge_trace_options.search_radius_px) * 0.40),
        )

    def _combined_instance_trace_geometry(
        self, segment: np.ndarray
    ) -> np.ndarray:
        """Append one preview segment to the ordered trace-session contour."""

        current = np.asarray(segment, dtype=np.int32).reshape(-1, 2)
        accumulated = self._instance_trace_geometry
        if accumulated is None or not len(accumulated):
            return current.copy()
        previous = np.asarray(accumulated, dtype=np.int32).reshape(-1, 2)
        if not len(current):
            return previous.copy()
        start = 1 if np.array_equal(previous[-1], current[0]) else 0
        return np.concatenate((previous, current[start:]), axis=0)

    def _closed_trace_geometry(
        self, geometry: np.ndarray
    ) -> np.ndarray | None:
        """Close a non-trivial contour returned near its first anchor."""

        points = np.asarray(geometry, dtype=np.int32).reshape(-1, 2)
        if len(points) < 4:
            return None
        if float(np.linalg.norm(points[-1].astype(float) - points[0])) > (
            self._trace_closure_radius()
        ):
            return None
        closed = points.copy()
        closed[-1] = closed[0]
        span_x = int(np.ptp(closed[:, 0]))
        span_y = int(np.ptp(closed[:, 1]))
        minimum_area = max(9.0, self._trace_closure_radius() ** 2)
        if min(span_x, span_y) < 3 or abs(cv2.contourArea(closed)) < minimum_area:
            return None
        return closed

    def _render_instance_assisted_preview(self) -> None:
        self._remove_instance_preview_items()
        colour = self.instance_colour(self._active_instance_id)
        pen = QPen(colour, 3.0)
        pen.setCosmetic(True)
        pen.setStyle(Qt.PenStyle.DashLine)
        if self._instance_annotation_tool in {"smart_fill", "shape_guided_fill"}:
            region = (
                self._instance_preview_region
                if self._instance_annotation_tool == "smart_fill"
                else self._instance_shape_guided_region
            )
            if region is not None and region.added_count and np.any(region.mask):
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
            if (
                self._instance_annotation_tool == "shape_guided_fill"
                and self._instance_shape_guided_region is not None
            ):
                guided = self._instance_shape_guided_region
                if guided.prior_polygon is not None:
                    prior_pen = QPen(QColor("#ffffff"), 1.5)
                    prior_pen.setCosmetic(True)
                    prior_pen.setStyle(Qt.PenStyle.DotLine)
                    prior_item = self._scene.addPath(
                        self._preview_path(guided.prior_polygon, closed=True),
                        prior_pen,
                        QBrush(Qt.BrushStyle.NoBrush),
                    )
                    prior_item.setToolTip(
                        "Automatically rotated ellipse prior; this outline is not "
                        "the committed boundary."
                    )
                    self._instance_preview_items.append(prior_item)
                if guided.boundary_polygon is not None:
                    boundary_colour = colour if guided.accepted else QColor("#ffb347")
                    boundary_pen = QPen(boundary_colour, 2.5)
                    boundary_pen.setCosmetic(True)
                    boundary_pen.setStyle(
                        Qt.PenStyle.SolidLine
                        if guided.accepted
                        else Qt.PenStyle.DashLine
                    )
                    boundary_item = self._scene.addPath(
                        self._preview_path(guided.boundary_polygon, closed=True),
                        boundary_pen,
                        QBrush(Qt.BrushStyle.NoBrush),
                    )
                    boundary_item.setToolTip(
                        (
                            f"Refined boundary: {guided.edge_coverage:.0%} edge "
                            f"coverage; confidence {guided.confidence:.0%}."
                        )
                        if guided.accepted
                        else guided.reason
                    )
                    self._instance_preview_items.append(boundary_item)
        else:
            geometry = self._instance_preview_geometry
            if geometry is not None and len(geometry) > 1:
                display_geometry = geometry
                closed_trace = None
                if self._instance_annotation_tool == "edge_trace":
                    closed_trace = self._closed_trace_geometry(
                        self._combined_instance_trace_geometry(geometry)
                    )
                    if closed_trace is not None:
                        display_geometry = closed_trace
                closed = closed_trace is not None
                filled_preview = (
                    closed_trace is not None
                    and self._edge_trace_options.fill_closed_loops
                )
                fill = QColor(colour)
                fill.setAlpha(55 if filled_preview else 0)
                item = self._scene.addPath(
                    self._preview_path(display_geometry, closed=closed),
                    pen,
                    (
                        QBrush(fill)
                        if filled_preview
                        else QBrush(Qt.BrushStyle.NoBrush)
                    ),
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
        if (
            self._instance_trace_geometry is not None
            and len(self._instance_trace_geometry) > 1
        ):
            start_x, start_y = self._instance_trace_geometry[0]
            start_pen = QPen(QColor("#71f5ff"), 2.0)
            start_pen.setCosmetic(True)
            start_pen.setStyle(Qt.PenStyle.DashLine)
            start_marker = self._scene.addEllipse(
                float(start_x) - 5.0,
                float(start_y) - 5.0,
                10.0,
                10.0,
                start_pen,
                QBrush(Qt.BrushStyle.NoBrush),
            )
            self._instance_preview_items.append(start_marker)
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
            self._instance_preview_point = QPointF(scene_point)
            self._instance_preview_geometry = None
            self._instance_preview_region = None
            self._instance_shape_guided_region = None
            self._instance_preview_endpoint = QPointF(scene_point)
            if self._instance_annotation_tool == "edge_trace":
                edge = self._annotation_edge_evidence(
                    self._edge_trace_options.edge_source
                )
                snapped = snap_edge_point(
                    (scene_point.x(), scene_point.y()),
                    edge,
                    self._edge_trace_options.search_radius_px,
                )
                if (
                    self._instance_trace_geometry is not None
                    and len(self._instance_trace_geometry) > 1
                ):
                    start_x, start_y = self._instance_trace_geometry[0]
                    if np.hypot(
                        float(snapped[0] - start_x),
                        float(snapped[1] - start_y),
                    ) <= self._trace_closure_radius():
                        snapped = (int(start_x), int(start_y))
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
            elif self._instance_annotation_tool == "shape_guided_fill":
                edge = self._annotation_edge_evidence(
                    self._shape_guided_edge_source
                )
                labels = self._ensure_instance_labels()
                corrected = np.asarray(
                    self._analysis_result.calibration.corrected_bgr
                )
                diameter = float(
                    getattr(
                        self._analysis_result,
                        "estimated_seed_diameter_px",
                        60.0,
                    )
                )
                self._instance_shape_guided_region = shape_guided_fill_region(
                    labels,
                    corrected,
                    edge,
                    (scene_point.x(), scene_point.y()),
                    self._active_instance_id,
                    diameter,
                    self._shape_guided_fill_options,
                )
                guided = self._instance_shape_guided_region
                self.instance_tool_status.emit(
                    (
                        f"Shape-guided preview: {guided.edge_coverage:.0%} edge "
                        f"coverage, {guided.confidence:.0%} confidence; click to apply."
                    )
                    if guided.accepted
                    else guided.reason
                )
                self._instance_preview_endpoint = None
            else:
                edge = self._annotation_edge_evidence(
                    self._smart_fill_edge_source
                )
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
                self._instance_trace_geometry = np.asarray(
                    ((round(endpoint.x()), round(endpoint.y())),),
                    dtype=np.int32,
                )
                self._instance_preview_point = None
                self._pending_instance_preview_point = None
                self._instance_preview_timer.stop()
                self.instance_tool_status.emit(
                    "Edge-trace anchor set. Move the cursor to preview a snapped "
                    "segment, then click to apply it."
                )
                self._render_instance_assisted_preview()
                return False
            geometry = self._instance_preview_geometry
            if geometry is None or len(geometry) < 2:
                return False
            combined = self._combined_instance_trace_geometry(geometry)
            closed_geometry = self._closed_trace_geometry(combined)
            self._detach_active_reference_buffers()
            fill_closed = (
                closed_geometry is not None
                and self._edge_trace_options.fill_closed_loops
            )
            changed = self._paint_instance_geometry(
                closed_geometry if fill_closed else geometry,
                filled=fill_closed,
            )
            if closed_geometry is None:
                self._instance_trace_geometry = combined
                self._instance_trace_anchor = QPointF(endpoint)
            else:
                self._instance_trace_geometry = None
                self._instance_trace_anchor = None
            self._instance_preview_point = None
            if fill_closed:
                message = (
                    f"Closed edge trace filled {changed:,} interior pixels for seed "
                    f"{self._active_instance_id}."
                )
            elif closed_geometry is not None:
                message = (
                    f"Closed edge trace added {changed:,} outline pixels to seed "
                    f"{self._active_instance_id}."
                )
            else:
                message = (
                    f"Edge trace added {changed:,} one-pixel edge pixels to seed "
                    f"{self._active_instance_id}; the endpoint is the next anchor."
                )
            self.instance_tool_status.emit(message)
        elif self._instance_annotation_tool == "shape_guided_fill":
            region = self._instance_shape_guided_region
            if region is None:
                self.instance_tool_status.emit(
                    "Shape-guided fill has no preview at this cursor position."
                )
                return False
            if not region.accepted:
                self.instance_tool_status.emit(region.reason)
                return False
            if not region.added_count:
                self.instance_tool_status.emit(
                    "The accepted shape-guided boundary adds no new pixels."
                )
                return False
            self._detach_active_reference_buffers()
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
                f"Shape-guided fill added {changed:,} pixels to seed "
                f"{self._active_instance_id} along a refined boundary with "
                f"{region.edge_coverage:.0%} edge coverage."
            )
        else:
            region = self._instance_preview_region
            if region is None or not region.added_count:
                self.instance_tool_status.emit(
                    "Smart fill found no reachable pixels at this cursor position."
                )
                return False
            self._detach_active_reference_buffers()
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

    def _annotation_edge_evidence(self, source: str) -> np.ndarray:
        """Return the exact user-selected edge raster for assisted tools."""

        source = str(source)
        net_scale = (
            float(
                getattr(
                    getattr(self._analysis_result, "layers", None),
                    "net_physical_edge_internal_scale",
                    0.50,
                )
            )
            if source == "net_physical"
            else None
        )
        cache_key = (
            f"edge_source:{source}:{net_scale:.6f}"
            if net_scale is not None
            else f"edge_source:{source}"
        )
        cached = self._annotation_evidence_cache.get(cache_key)
        if cached is not None:
            return cached

        attributes = {
            "magnitude": "edge_likelihood",
            "ridges": "edge_ridges",
            "reference_ridges": "reference_edge_ridges",
            "traces": "edge_trace_labels",
            "physical": "physical_edge_probability",
        }

        def display_u8(attribute: str, *, binary: bool = False) -> np.ndarray:
            values = np.asarray(self._full_annotation_evidence(attribute))
            if binary:
                return np.where(values > 0, 255, 0).astype(np.uint8)
            if values.dtype == np.uint8:
                return values
            result = np.asarray(values, dtype=np.float32)
            maximum = float(np.nanmax(result)) if result.size else 0.0
            if maximum <= 1.0:
                result = result * 255.0
            return np.nan_to_num(
                result, nan=0.0, posinf=255.0, neginf=0.0
            ).clip(0.0, 255.0).astype(np.uint8)

        if source in attributes:
            result = display_u8(
                attributes[source], binary=source == "traces"
            )
        elif source == "net_physical":
            physical = display_u8("physical_edge_probability").astype(
                np.float32
            )
            try:
                non_physical = display_u8("non_edge_probability").astype(
                    np.float32
                )
            except RuntimeError:
                non_physical = np.zeros_like(physical)
            result = np.rint(
                np.clip(physical - float(net_scale) * non_physical, 0.0, 255.0)
            ).astype(np.uint8)
        elif source == "adaptive":
            precise = []
            for attribute, binary in (
                ("edge_ridges", False),
                ("edge_trace_labels", True),
            ):
                try:
                    precise.append(display_u8(attribute, binary=binary))
                except RuntimeError:
                    pass
            broad = []
            for attribute in ("physical_edge_probability", "edge_likelihood"):
                try:
                    broad.append(display_u8(attribute))
                except RuntimeError:
                    pass
            available = precise or broad
            if not available:
                raise RuntimeError(
                    "The analysis did not produce usable annotation edge evidence."
                )
            result = np.maximum.reduce(available)
            if precise and broad:
                result = np.maximum(
                    np.maximum.reduce(precise),
                    np.rint(np.maximum.reduce(broad) * 0.45).astype(np.uint8),
                )
        else:
            raise ValueError(f"Unknown annotation edge source {source!r}.")
        self._annotation_evidence_cache[cache_key] = result
        return result

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
            self._instance_bounds_cache.clear()
        return self._instance_annotations

    def _detach_active_reference_buffers(self) -> None:
        """Copy the active categorical group once before an atomic edit.

        MainWindow retains the previous arrays until the completed stroke or
        assisted click is emitted. This copy-on-write boundary lets it build a
        compact undo delta without taking full-resolution snapshots per dab.
        """

        image_size = self.image_size
        if image_size is None:
            return
        width, height = image_size
        mode = self._reference_point_mode
        if mode == "instance":
            self._instance_annotations = self._ensure_instance_labels().copy()
            return
        if mode in {"background", "foreground", "other"}:
            background, foreground, other = self._ensure_material_reference_masks(
                height, width
            )
            self._background_reference_mask = background.copy()
            self._foreground_reference_mask = foreground.copy()
            self._background_exclusion_mask = other.copy()
            self._foreground_exclusion_mask = self._background_exclusion_mask
            return
        if mode in {"physical_edge", "non_edge"}:
            physical, non_edge = self._ensure_edge_reference_masks(height, width)
            self._physical_edge_reference_mask = physical.copy()
            self._non_edge_reference_mask = non_edge.copy()

    def _paint_instance_geometry(self, geometry: np.ndarray, *, filled: bool) -> int:
        labels = self._ensure_instance_labels()
        paint = np.zeros(labels.shape, dtype=np.uint8)
        points = np.asarray(geometry, dtype=np.int32).reshape((-1, 1, 2))
        if filled:
            cv2.fillPoly(paint, [points], 255, lineType=cv2.LINE_8)
        else:
            cv2.polylines(
                paint,
                [points],
                False,
                255,
                thickness=1,
                lineType=cv2.LINE_8,
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
            if self._instance_annotation_tool == "edge_trace":
                edge = self._annotation_edge_evidence(
                    self._edge_trace_options.edge_source
                )
                points = [*self._instance_tool_points]
                if not points or QLineF(points[-1], end_point).length() > 0.5:
                    points.append(QPointF(end_point))
                if len(points) < 2:
                    return False
                self._detach_active_reference_buffers()
                tangent = self._annotation_tangent(
                    self._edge_trace_options.tangent_mode
                )
                traced_segments = []
                for start, end in zip(points, points[1:]):
                    path = trace_edge_path(
                        (start.x(), start.y()),
                        (end.x(), end.y()),
                        edge,
                        self._edge_trace_options,
                        tangent,
                    )
                    if traced_segments and np.array_equal(
                        traced_segments[-1][-1], path[0]
                    ):
                        path = path[1:]
                    if len(path):
                        traced_segments.append(path)
                if not traced_segments:
                    return False
                geometry = np.concatenate(traced_segments, axis=0)
                closed_geometry = self._closed_trace_geometry(geometry)
                fill_closed = (
                    closed_geometry is not None
                    and self._edge_trace_options.fill_closed_loops
                )
                changed = self._paint_instance_geometry(
                    closed_geometry if fill_closed else geometry,
                    filled=fill_closed,
                )
                self.instance_tool_status.emit(
                    (
                        f"Closed edge trace filled {changed:,} interior pixels for seed "
                        if fill_closed
                        else f"Edge trace added {changed:,} one-pixel edge pixels to seed "
                    )
                    + f"{self._active_instance_id}."
                )
                return changed > 0
            if self._instance_annotation_tool == "shape_guided_fill":
                edge = self._annotation_edge_evidence(
                    self._shape_guided_edge_source
                )
                labels = self._ensure_instance_labels()
                corrected = np.asarray(
                    self._analysis_result.calibration.corrected_bgr
                )
                diameter = float(
                    getattr(
                        self._analysis_result,
                        "estimated_seed_diameter_px",
                        60.0,
                    )
                )
                region = shape_guided_fill_region(
                    labels,
                    corrected,
                    edge,
                    (end_point.x(), end_point.y()),
                    self._active_instance_id,
                    diameter,
                    self._shape_guided_fill_options,
                )
                if not region.accepted:
                    self.instance_tool_status.emit(region.reason)
                    return False
                self._detach_active_reference_buffers()
                labels = self._ensure_instance_labels()
                roi = labels[
                    region.y : region.y + region.mask.shape[0],
                    region.x : region.x + region.mask.shape[1],
                ]
                writable = region.mask & (
                    (roi == 0) | (roi == self._active_instance_id)
                )
                changed = int(
                    np.count_nonzero(
                        writable & (roi != self._active_instance_id)
                    )
                )
                roi[writable] = np.uint16(self._active_instance_id)
                self.instance_tool_status.emit(
                    f"Shape-guided fill added {changed:,} pixels to seed "
                    f"{self._active_instance_id}."
                )
                return changed > 0
            if self._instance_annotation_tool == "smart_fill":
                edge = self._annotation_edge_evidence(
                    self._smart_fill_edge_source
                )
                self._detach_active_reference_buffers()
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
            self._detach_active_reference_buffers()
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
                self._start_reference_live_stroke(
                    scene_point,
                    erase=self._reference_stroke_erases,
                )
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
        if self._reference_point_mode == "instance":
            self._extend_instance_live_stroke(scene_point)
        else:
            self._extend_reference_live_stroke(scene_point)
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
                # Signal payloads are immutable snapshots for external listeners.
                # The main window stores this sole copy as the next draft.
                mask = self.reference_mask(self._reference_point_mode)
                self.reference_mask_edited.emit(self._reference_point_mode, mask)
                self._render_analysis()
                self._clear_reference_live_stroke()
            self.reference_paint_finished.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._last_reference_hover_point = None
        self._hide_reference_brush_outline()
        self._instance_preview_point = None
        self._instance_preview_geometry = None
        self._instance_preview_region = None
        self._instance_shape_guided_region = None
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
            mode = self._reference_point_mode
            attributes = {
                "background": "_background_reference_mask",
                "foreground": "_foreground_reference_mask",
                "other": "_background_exclusion_mask",
                "physical_edge": "_physical_edge_reference_mask",
                "non_edge": "_non_edge_reference_mask",
            }
            attribute = attributes[mode]
            mask = getattr(self, attribute)
            if mask is None or mask.shape != (height, width):
                mask = np.zeros((height, width), dtype=bool)
                setattr(self, attribute, mask)
            value = not erase
            if mode == "other":
                self._foreground_exclusion_mask = mask
            if (
                not erase
                and mode in {"physical_edge", "non_edge"}
                and self._edge_reference_snap_enabled
            ):
                scene_point = self._snap_reference_edge_point(scene_point)
            if mode in {"background", "foreground", "other"}:
                exclusive_masks = self._ensure_material_reference_masks(height, width)
            else:
                exclusive_masks = self._ensure_edge_reference_masks(height, width)
            # The helper may have lazily copied an immutable applied array.
            mask = getattr(self, attribute)
        if (
            self._reference_point_mode == "instance"
            and self._show_selected_instance_only
        ):
            self._paint_selected_instance_circle(
                mask, scene_point.x(), scene_point.y(), erase=erase
            )
        else:
            self._paint_mask_circle(mask, scene_point.x(), scene_point.y(), value)
        if self._reference_point_mode != "instance" and not erase:
            for other_mask in exclusive_masks:
                if other_mask is mask:
                    continue
                self._paint_mask_circle(
                    other_mask, scene_point.x(), scene_point.y(), False
                )

    def _ensure_material_reference_masks(
        self, height: int, width: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        attributes = (
            "_background_reference_mask",
            "_foreground_reference_mask",
            "_background_exclusion_mask",
        )
        masks = []
        for attribute in attributes:
            mask = getattr(self, attribute)
            if mask is None or mask.shape != (height, width):
                mask = np.zeros((height, width), dtype=bool)
                setattr(self, attribute, mask)
            elif not mask.flags.writeable:
                mask = mask.copy()
                setattr(self, attribute, mask)
            masks.append(mask)
        # Other is represented to the existing analysis API as negative
        # evidence for both material models; both names share one array.
        self._foreground_exclusion_mask = masks[2]
        return tuple(masks)

    def _ensure_edge_reference_masks(
        self, height: int, width: int
    ) -> tuple[np.ndarray, np.ndarray]:
        masks = []
        for attribute in (
            "_physical_edge_reference_mask",
            "_non_edge_reference_mask",
        ):
            mask = getattr(self, attribute)
            if mask is None or mask.shape != (height, width):
                mask = np.zeros((height, width), dtype=bool)
                setattr(self, attribute, mask)
            elif not mask.flags.writeable:
                mask = mask.copy()
                setattr(self, attribute, mask)
            masks.append(mask)
        return tuple(masks)

    def _snap_reference_edge_point(self, scene_point: QPointF) -> QPointF:
        """Snap boundary-review dabs to the strongest nearby analysed edge."""

        try:
            edge = self._full_annotation_evidence("edge_likelihood")
            snapped_x, snapped_y = snap_edge_point(
                (scene_point.x(), scene_point.y()),
                edge,
                max(3.0, self._reference_brush_radius * 1.5),
            )
            strength = self._edge_reference_snap_strength
            return QPointF(
                float(scene_point.x() + strength * (snapped_x - scene_point.x())),
                float(scene_point.y() + strength * (snapped_y - scene_point.y())),
            )
        except (RuntimeError, ValueError):
            return scene_point

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

    def _paint_selected_instance_circle(
        self,
        labels: np.ndarray,
        point_x: float,
        point_y: float,
        *,
        erase: bool,
    ) -> None:
        """Edit the active ID without damaging hidden neighbouring seeds."""

        radius = max(1, int(round(self._reference_brush_radius)))
        center_x = int(round(point_x))
        center_y = int(round(point_y))
        x0 = max(0, center_x - radius)
        x1 = min(labels.shape[1], center_x + radius + 1)
        y0 = max(0, center_y - radius)
        y1 = min(labels.shape[0], center_y + radius + 1)
        if x0 >= x1 or y0 >= y1:
            return
        yy, xx = np.ogrid[y0:y1, x0:x1]
        circle = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius**2
        region = labels[y0:y1, x0:x1]
        if erase:
            writable = circle & (region == self._active_instance_id)
            region[writable] = 0
        else:
            writable = circle & (
                (region == 0) | (region == self._active_instance_id)
            )
            region[writable] = np.uint16(self._active_instance_id)

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
