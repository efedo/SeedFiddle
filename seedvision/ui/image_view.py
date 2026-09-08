"""Zoomable image canvas used for image review and future mask editing."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
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
    ANNOTATION_EDGE_SOURCES,
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
    "ruler_evidence",
    "ruler_detection",
    "layout_detection",
    "hue_only",
    "wavelet_detail_1",
    "wavelet_detail_2",
    "wavelet_detail_3",
    "wavelet_detail_4",
    "wavelet_residual",
    "perimeter_background_reference",
    "seed_scale_estimation",
    "seed_size_ovality_distribution",
    "seed_pose_shape_distributions",
    "seed_mean_shape_atlas",
    "seed_shape_uncertainty",
    "seed_boundary_curvature_distribution",
    "foreground_mask",
    "foreground_colour_gamut",
    "distance_transform",
    "distance_candidates",
    "circle_candidates",
    "proposals",
    "instance_masks",
    "background_likelihood",
    "other_colour_probability",
    "foreground_colour_excess",
    "background_colour_gamut",
    "refined_background_likelihood",
    "other_noise_probability",
    "foreground_noise_likelihood",
    "foreground_noise_excess",
    "material_seed_support",
    "material_background_support",
    "material_other_support",
    "material_nonseed_support",
    "material_seed_probability",
    "material_nonseed_probability",
    "material_ambiguity_probability",
    "material_unknown_probability",
    "material_background_subtype",
    "material_other_subtype",
    "material_subtype_ambiguity",
    "material_subtype_unknown",
    "seed_coat_white_probability",
    "seed_coat_banded_light_probability",
    "seed_coat_banded_dark_probability",
    "seed_coat_other_probability",
    "seed_condition_immature_probability",
    "seed_condition_split_probability",
    "seed_condition_wrinkled_probability",
    "seed_condition_stained_probability",
    "reference_texture_prototypes",
    "reference_prototype_footprints",
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
    "reference_edge_excess",
    "physical_edge_interior_direction",
    "net_physical_edge_probability",
    "reference_edge_probability",
    "conservative_net_physical_edge_evidence",
    "reference_edge_ridges",
    "locally_normalized_net_physical_edge",
    "net_reference_edge_ridges",
    "normalized_net_reference_edge_ridges",
    "edge_traces",
    "edge_trace_continuity",
    "edge_trace_gap_confidence",
    "edge_radius_confirmation",
    "edge_circle_fit",
    "edge_ellipse_fit",
    "edge_fit_residual",
    "edge_centre_votes",
    "oval_centre_probability",
    "edge_semantic_sides",
    "edge_rejections",
    "edge_fit_geometry",
    "edge_oval_hypotheses",
    "seed_edge_curves",
    "procedural_seed_material",
    "procedural_seed_mask",
    "procedural_boundary_cost",
    "procedural_centres",
    "procedural_instances",
    "procedural_confidence",
    "procedural_concavity",
    "procedural_alternative_candidates",
    "procedural_reference_error",
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
    manual_seed_centres_edited = Signal(object, str, str)
    manual_seed_centre_editing_cancelled = Signal()
    procedural_instance_selected = Signal(int)
    hilum_landmark_edited = Signal(object, object)
    hilum_editing_cancelled = Signal()

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
        self._automatic_background_reference_visible = True
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
        self._manual_seed_centres = np.empty((0, 2), dtype=np.float64)
        self._manual_seed_centre_mode = "augment"
        self._manual_seed_centre_editing = False
        self._manual_seed_centre_selected: dict[str, object] | None = None
        self._manual_seed_centre_drag: dict[str, object] | None = None
        self._manual_seed_centre_drag_items: list[QGraphicsItem] = []
        self._selected_procedural_label = 0
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
        self._hilum_editing = False
        self._hilum_point = None
        self._hilum_direction = None
        self._hilum_drag = None
        self._hilum_items = []
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
        self._build_calculating_banner()
        self._build_instance_continuity_warning()
        self._show_placeholder()

    def _build_zoom_controls(self) -> None:
        self.zoom_controls = QFrame(self)
        controls = QHBoxLayout(self.zoom_controls)
        controls.setContentsMargins(8, 4, 8, 4)
        controls.setSpacing(5)
        heading = QLabel("Image zoom", self.zoom_controls)
        self.current_file_label = QLabel("No image selected", self.zoom_controls)
        self.current_file_label.setObjectName("currentImageFileLabel")
        self.current_file_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
        )
        self.current_file_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.current_file_label.setStyleSheet(
            "QLabel#currentImageFileLabel { color: #e8eef4; font-weight: 600; }"
        )
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
        controls.addWidget(self.current_file_label, 2)
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

    def _build_calculating_banner(self) -> None:
        """Create the unmistakable selected-overlay calculation notice."""

        self.calculating_banner = QLabel(self)
        self.calculating_banner.setObjectName("overlayCalculatingBanner")
        self.calculating_banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.calculating_banner.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents
        )
        self.calculating_banner.setStyleSheet(
            "QLabel#overlayCalculatingBanner {"
            " color: #ff3434;"
            " background-color: rgba(15, 18, 22, 210);"
            " border: 3px solid #d71920;"
            " border-radius: 8px;"
            " padding: 12px 24px;"
            " font-weight: 900;"
            " font-size: 28px;"
            "}"
        )
        self.calculating_banner.hide()

    def set_overlay_calculating(self, text: str | None) -> None:
        message = "" if text is None else str(text).strip()
        self.calculating_banner.setText(message)
        self.calculating_banner.setVisible(bool(message))
        self._layout_calculating_banner()
        if message:
            self.calculating_banner.raise_()
        self.zoom_controls.raise_()

    def _layout_calculating_banner(self) -> None:
        if self.calculating_banner.isHidden():
            return
        width = min(max(320, self.calculating_banner.sizeHint().width()), max(320, self.width() - 40))
        height = max(64, self.calculating_banner.sizeHint().height())
        self.calculating_banner.setGeometry(
            max(0, (self.width() - width) // 2),
            max(self.zoom_controls.height() + 18, (self.height() - height) // 5),
            width,
            height,
        )

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

    def _hilum_arrow_length(self):
        return max(30., float(getattr(self._analysis_result, "estimated_seed_diameter_px", 80.)) * 0.45)

    def set_hilum_editing(self, enabled: bool) -> None:
        self._hilum_editing = bool(enabled)
        self._hilum_drag = None
        if enabled:
            self._clear_instance_preview()
            self._hide_reference_brush_outline()
            self.setFocus()
        self.viewport().setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)
        self._draw_hilum_landmark()

    def set_hilum_landmark(self, point, direction) -> None:
        self._hilum_point, self._hilum_direction = point, direction
        self._draw_hilum_landmark()

    def _draw_hilum_landmark(self):
        for item in self._hilum_items:
            self._scene.removeItem(item)
        self._hilum_items.clear()
        if not self._hilum_editing or self._hilum_point is None:
            return
        start = QPointF(*self._hilum_point)
        radius = 6 / max(self.transform().m11(), .05)
        path = QPainterPath()
        path.addEllipse(start, radius, radius)
        path.moveTo(start + QPointF(-radius*1.5, 0))
        path.lineTo(start + QPointF(radius*1.5, 0))
        path.moveTo(start + QPointF(0, -radius*1.5))
        path.lineTo(start + QPointF(0, radius*1.5))
        if self._hilum_direction is not None:
            direction = QPointF(*self._hilum_direction)
            tip = start + direction*self._hilum_arrow_length()
            normal = QPointF(-direction.y(), direction.x())
            path.moveTo(start)
            path.lineTo(tip)
            path.moveTo(tip - direction*radius*2 + normal*radius)
            path.lineTo(tip)
            path.lineTo(tip - direction*radius*2 - normal*radius)
            path.addEllipse(tip, radius*.65, radius*.65)
        for colour, width, z in (("#171b24", 6, 100), ("#ff66dc", 2.5, 101)):
            pen = QPen(QColor(colour), width)
            pen.setCosmetic(True)
            item = self._scene.addPath(path, pen)
            item.setZValue(z)
            self._hilum_items.append(item)

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
        user_size = getattr(self._context_panel, "user_size", None)
        hint = user_size or self._context_panel.sizeHint()
        available_width = max(240, self.width() - 24)
        minimum_y = self._context_panel_minimum_y()
        available_height = max(140, self.height() - minimum_y - 12)
        width = min(max(300, hint.width()) if user_size else 430, available_width)
        height = min(hint.height() if user_size else 680, available_height)
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

        self.set_hilum_editing(False)
        self.hilum_editing_cancelled.emit()
        self.set_hilum_landmark(None, None)
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
        self._clear_manual_seed_centre_drag_items()
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
        self._manual_seed_centres = np.empty((0, 2), dtype=np.float64)
        self._manual_seed_centre_mode = "augment"
        self._manual_seed_centre_editing = False
        self._manual_seed_centre_selected = None
        self._manual_seed_centre_drag = None
        self._selected_procedural_label = 0
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
        self.current_file_label.setText(path.name)
        self.current_file_label.setToolTip(str(path))
        if self.isVisible():
            self.fit_image()
        else:
            self._fit_pending = True
        return True, ""

    def clear_image(self) -> None:
        self.set_hilum_editing(False)
        self.hilum_editing_cancelled.emit()
        self.set_hilum_landmark(None, None)
        """Return to the empty placeholder without retaining image-local state."""

        self._instance_preview_timer.stop()
        self._gamut_render_timer.stop()
        release_host_caches(self._analysis_result)
        self._clear_manual_seed_centre_drag_items()
        self._image_path = None
        self.current_file_label.setText("No image selected")
        self.current_file_label.setToolTip("")
        self._background_reference_mask = None
        self._foreground_reference_mask = None
        self._background_exclusion_mask = None
        self._foreground_exclusion_mask = None
        self._physical_edge_reference_mask = None
        self._non_edge_reference_mask = None
        self._instance_annotations = None
        self._manual_seed_centres = np.empty((0, 2), dtype=np.float64)
        self._manual_seed_centre_mode = "augment"
        self._manual_seed_centre_editing = False
        self._manual_seed_centre_selected = None
        self._manual_seed_centre_drag = None
        self._instance_bounds_cache.clear()
        self._instance_tool_points.clear()
        self._annotation_evidence_cache.clear()
        self._instance_trace_anchor = None
        self._instance_trace_geometry = None
        self._instance_preview_geometry = None
        self._instance_preview_region = None
        self._instance_shape_guided_region = None
        self._instance_preview_endpoint = None
        self._instance_preview_point = None
        self._pending_instance_preview_point = None
        self._reference_point_mode = None
        self._reference_paint_button = None
        self._last_reference_paint_point = None
        self._last_reference_hover_point = None
        self._show_placeholder()
        self._fit_pending = False

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

    def render_reference_annotations_without_analysis(self) -> None:
        """Show restored image-local references even before calibration reruns."""

        if self._analysis_result is not None:
            self._render_analysis()
            return
        self._clear_overlay_items()
        self._restore_source_image()
        if self._material_reference_annotations_visible:
            self._render_background_reference_points()
            self._render_foreground_reference_points()
            self._render_exclusion_masks()
        self._render_instance_annotations()

    @property
    def manual_seed_centre_editing(self) -> bool:
        return self._manual_seed_centre_editing

    def set_manual_seed_centres(
        self,
        centres_xy: np.ndarray | None,
        *,
        mode: str = "augment",
        render: bool = True,
    ) -> None:
        """Supply manual centres in full corrected-image coordinates."""

        if mode not in {"augment", "replace_automatic"}:
            raise ValueError(f"Unknown manual seed-centre mode {mode!r}.")
        if centres_xy is None:
            values = np.empty((0, 2), dtype=np.float64)
        else:
            values = np.asarray(centres_xy)
            if values.ndim != 2 or values.shape[1:] != (2,):
                raise ValueError("Manual seed centres must be an N×2 array.")
            if not np.all(np.isfinite(values)):
                raise ValueError("Manual seed centres must contain finite coordinates.")
            values = values.astype(np.float64, copy=True)
        self._manual_seed_centres = values
        self._manual_seed_centre_mode = mode
        self._manual_seed_centre_selected = None
        self._manual_seed_centre_drag = None
        self._clear_manual_seed_centre_drag_items()
        if render and self._analysis_result is not None:
            self._render_analysis()

    def manual_seed_centres(self) -> tuple[np.ndarray, str]:
        return self._manual_seed_centres.copy(), self._manual_seed_centre_mode

    def set_manual_seed_centre_editing(self, enabled: bool) -> None:
        """Enter the click/add, drag/move, and right-click/remove centre mode."""

        enabled = bool(enabled)
        if enabled == self._manual_seed_centre_editing:
            return
        self._manual_seed_centre_editing = enabled
        self._manual_seed_centre_selected = None
        self._manual_seed_centre_drag = None
        self._clear_manual_seed_centre_drag_items()
        if enabled:
            self._set_reference_point_mode(None)
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.viewport().setCursor(Qt.CursorShape.CrossCursor)
            self.setFocus(Qt.FocusReason.MouseFocusReason)
        elif self._reference_point_mode is None:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.viewport().unsetCursor()
        if self._analysis_result is not None:
            self._render_analysis()

    def editable_procedural_marker_centres(
        self,
        *,
        exclude_result_index: int | None = None,
    ) -> np.ndarray:
        """Return surviving non-annotation markers in corrected coordinates."""

        records = self._procedural_marker_records(include_current_manual=True)
        points: list[tuple[float, float]] = []
        for record in records:
            if record["source"] in {"annotation", "rejected"}:
                continue
            result_index = record.get("result_index")
            if exclude_result_index is not None and result_index == exclude_result_index:
                continue
            point = record["point"]
            candidate = (float(point.x()), float(point.y()))
            if any(np.hypot(candidate[0] - x, candidate[1] - y) < 0.25 for x, y in points):
                continue
            points.append(candidate)
        return np.asarray(points, dtype=np.float64).reshape(-1, 2)

    def _clear_overlay_items(self) -> None:
        self._clear_manual_seed_centre_drag_items()
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
            self._selected_procedural_label = 0
        self._analysis_result = result
        self._annotation_evidence_cache.clear()
        if render:
            self._render_analysis()

    def prepare_analysis_coordinates(self) -> None:
        """Adopt corrected dimensions before corrected-coordinate masks are bound.

        MainWindow installs a newly validated sidecar between storing an analysis
        result and rendering its selected overlay.  This lightweight transition
        makes that operation atomic even when the prior scene still displays the
        raw source dimensions.
        """

        result = self._analysis_result
        calibration = None if result is None else getattr(result, "calibration", None)
        corrected = (
            None if calibration is None else getattr(calibration, "corrected_bgr", None)
        )
        if corrected is not None:
            self._set_bgr_base_image(corrected)

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

    def set_automatic_background_reference_visible(self, visible: bool) -> None:
        """Show the retained automatic source while painting Background."""

        visible = bool(visible)
        if visible == self._automatic_background_reference_visible:
            return
        self._automatic_background_reference_visible = visible
        if self._reference_point_mode == "background":
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
        if selected_source not in ANNOTATION_EDGE_SOURCES:
            raise ValueError(f"Unknown annotation edge source {selected_source!r}.")
        if options.edge_source != selected_source:
            options = replace(options, edge_source=selected_source)
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
        if edge_source not in ANNOTATION_EDGE_SOURCES:
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
        previous_mode = self._reference_point_mode
        if mode is not None and self._manual_seed_centre_editing:
            self._manual_seed_centre_editing = False
            self._manual_seed_centre_selected = None
            self._manual_seed_centre_drag = None
            self._clear_manual_seed_centre_drag_items()
            self.manual_seed_centre_editing_cancelled.emit()
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
        if (previous_mode == "background") != (mode == "background"):
            self._render_analysis()

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
            else QColor("#ffad55")
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
        if self._overlay_mode == "ruler_evidence":
            self._render_ruler_evidence(result)
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
        if self._overlay_mode == "hue_only":
            self._render_rgba_overlay(
                result.layers.hue_only_rgba(),
                result.layers.offset_x,
                result.layers.offset_y,
            )
            self._render_context_annotations(result)
            return
        if self._overlay_mode.startswith("wavelet_detail_"):
            level = int(self._overlay_mode.rsplit("_", 1)[1]) - 1
            self._render_rgba_overlay(
                result.layers.wavelet_rgba(level),
                result.layers.offset_x,
                result.layers.offset_y,
            )
            self._render_context_annotations(result)
            return
        if self._overlay_mode == "wavelet_residual":
            self._render_rgba_overlay(
                result.layers.wavelet_rgba(None),
                result.layers.offset_x,
                result.layers.offset_y,
            )
            self._render_context_annotations(result)
            return
        if self._overlay_mode == "perimeter_background_reference":
            self._render_dish_edges(result, width=3, include_inner=False)
            self._render_background_sampling_band(result)
            self._render_background_starting_colour(result)
            self._render_context_annotations(result)
            return
        if self._overlay_mode in {
            "seed_scale_estimation",
            "seed_size_ovality_distribution",
            "seed_pose_shape_distributions",
            "seed_mean_shape_atlas",
            "seed_shape_uncertainty",
            "seed_boundary_curvature_distribution",
        }:
            self._render_seed_shape(result, self._overlay_mode)
            self._render_context_annotations(result)
            return
        if self._overlay_mode == "proposals":
            self._render_proposals(result)
            self._render_context_annotations(result)
            return

        if self._overlay_mode in {
            "foreground_mask",
            "distance_transform",
        }:
            if self._overlay_mode == "foreground_mask":
                raster = getattr(
                    result,
                    "foreground_colour_probability",
                    result.foreground_probability,
                )
            else:
                raster = getattr(result, self._overlay_mode)
            if raster is None:
                self._render_context_annotations(result)
                return
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

        material_rasters = {
            "material_seed_support": "seed_evidence_support",
            "material_background_support": "background_evidence_support",
            "material_other_support": "other_evidence_support",
            "material_nonseed_support": "nonseed_evidence_support",
            "material_seed_probability": "seed_material_probability",
            "material_nonseed_probability": "nonseed_material_probability",
            "material_ambiguity_probability": "material_ambiguity_probability",
            "material_unknown_probability": "material_unknown_probability",
            "material_background_subtype": "conditional_background_probability",
            "material_other_subtype": "conditional_other_probability",
            "material_subtype_ambiguity": "conditional_material_subtype_ambiguity",
            "material_subtype_unknown": "conditional_nonseed_unknown_probability",
        }
        if self._overlay_mode in material_rasters:
            raster = getattr(result.layers, material_rasters[self._overlay_mode])
            if raster is not None:
                self._render_scalar_raster(
                    raster,
                    *result.crop_offset,
                    valid_mask=result.layers.valid_mask,
                )
            self._render_context_annotations(result)
            return

        if self._overlay_mode.startswith("seed_coat_"):
            class_name = self._overlay_mode[
                len("seed_coat_") : -len("_probability")
            ]
            self._render_rgba_overlay(
                result.layers.reference_seed_trait_probability_rgba(
                    "coat", class_name
                ),
                *result.crop_offset,
            )
            self._render_context_annotations(result)
            return
        if self._overlay_mode.startswith("seed_condition_"):
            class_name = self._overlay_mode[
                len("seed_condition_") : -len("_probability")
            ]
            self._render_rgba_overlay(
                result.layers.reference_seed_trait_probability_rgba(
                    "condition", class_name
                ),
                *result.crop_offset,
            )
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
                    self._render_selected_procedural_instance(result, procedural)
                elif self._overlay_mode == "procedural_alternative_candidates":
                    self._render_rgba_overlay(
                        procedural.alternative_candidates_rgba,
                        *result.crop_offset,
                        source_shape=procedural.source_shape,
                    )
                elif self._overlay_mode == "procedural_reference_error":
                    self._render_rgba_overlay(
                        procedural.reference_error_rgba,
                        *result.crop_offset,
                        source_shape=procedural.source_shape,
                    )
                else:
                    raster = {
                        "procedural_seed_material": procedural.occupancy_likelihood,
                        "procedural_seed_mask": procedural.occupancy_mask,
                        "procedural_boundary_cost": procedural.boundary_cost,
                        "procedural_centres": procedural.centre_likelihood,
                        "procedural_confidence": procedural.confidence_raster(),
                        "procedural_concavity": procedural.concavity,
                    }[self._overlay_mode]
                    self._render_scalar_raster(
                        raster,
                        *result.crop_offset,
                        valid_mask=result.layers.valid_mask,
                        source_shape=procedural.source_shape,
                    )
                    if self._overlay_mode == "procedural_centres":
                        self._render_procedural_centres(result)
            if self._overlay_mode != "procedural_reference_error":
                # The reference-error raster already encodes the compared
                # annotation. Drawing categorical seed paint over it would
                # obscure the per-pixel cost shading it is meant to inspect.
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
        elif self._overlay_mode == "other_colour_probability":
            rgba = layers.other_colour_rgba()
        elif self._overlay_mode == "other_noise_probability":
            rgba = layers.other_noise_rgba()
        elif self._overlay_mode == "foreground_colour_excess":
            rgba = layers.foreground_colour_excess_rgba(
                getattr(
                    result,
                    "foreground_colour_probability",
                    result.foreground_probability,
                )
            )
        elif self._overlay_mode == "foreground_noise_excess":
            rgba = layers.foreground_noise_excess_rgba()
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
        elif self._overlay_mode == "reference_prototype_footprints":
            rgba = layers.reference_prototype_footprints_rgba()
        elif self._overlay_mode == "physical_edge_probability":
            rgba = layers.reference_edge_prototype_compatibility_rgba(True)
        elif self._overlay_mode == "non_edge_probability":
            rgba = layers.reference_edge_prototype_compatibility_rgba(False)
        elif self._overlay_mode == "reference_edge_comparison":
            rgba = layers.reference_edge_comparison_rgba()
        elif self._overlay_mode == "reference_edge_excess":
            rgba = layers.reference_edge_excess_rgba()
        elif self._overlay_mode == "physical_edge_interior_direction":
            rgba = layers.physical_edge_interior_direction_rgba()
        elif self._overlay_mode == "net_physical_edge_probability":
            rgba = layers.net_physical_edge_probability_rgba()
        elif self._overlay_mode == "reference_edge_probability":
            rgba = layers.reference_edge_supported_probability_rgba()
        elif self._overlay_mode == "conservative_net_physical_edge_evidence":
            rgba = layers.conservative_net_physical_edge_evidence_rgba()
        elif self._overlay_mode == "reference_edge_ridges":
            rgba = layers.reference_edge_ridges_rgba()
        elif self._overlay_mode == "locally_normalized_net_physical_edge":
            rgba = layers.locally_normalized_net_physical_edge_rgba()
        elif self._overlay_mode == "net_reference_edge_ridges":
            rgba = layers.net_reference_edge_ridges_rgba()
        elif self._overlay_mode == "normalized_net_reference_edge_ridges":
            rgba = layers.normalized_net_reference_edge_ridges_rgba()
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
        elif self._overlay_mode == "oval_centre_probability":
            rgba = layers._heat_rgba(layers.oval_centre_probability)
        elif self._overlay_mode == "edge_semantic_sides":
            rgba = layers._heat_rgba(layers.edge_semantic_sides)
        elif self._overlay_mode == "edge_rejections":
            rgba = layers.edge_rejection_rgba()
        elif self._overlay_mode == "edge_fit_geometry":
            self._render_edge_fit_geometry(result)
            self._render_context_annotations(result)
            return
        elif self._overlay_mode == "edge_oval_hypotheses":
            self._render_edge_oval_hypotheses(result)
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
            automatic_source_paint_view = (
                self._reference_point_mode == "background"
                and self._automatic_background_reference_visible
                and getattr(
                    layers, "background_reference_source_mask", None
                )
                is not None
            )
            if (
                self._automatic_background_reference_visible
                and not automatic_source_paint_view
            ):
                self._render_background_sampling_band(result, fill=False)
        self._render_context_annotations(result)

    def _render_selected_procedural_instance(self, result, procedural) -> None:
        """Outline the inspected working-resolution procedural label in yellow."""

        label = int(self._selected_procedural_label)
        if label <= 0 or label > procedural.count:
            return
        selected = np.asarray(procedural.labels) == label
        if not np.any(selected):
            return
        edge = cv2.morphologyEx(
            np.uint8(selected), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)
        ) > 0
        rgba = np.zeros((*edge.shape, 4), np.uint8)
        rgba[edge] = (255, 225, 36, 255)
        self._render_rgba_overlay(
            rgba,
            *result.crop_offset,
            source_shape=procedural.source_shape,
        )

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
        if self._manual_seed_centre_editing:
            # The editor renders the same real watershed markers with source
            # semantics and hit targets after the selected raster overlay.
            return
        procedural = getattr(result, "procedural_instances", None)
        if procedural is None:
            return
        for record in self._procedural_marker_records(include_current_manual=False):
            self._render_manual_seed_centre_marker(record, editing=False)

    @staticmethod
    def _normalised_procedural_marker_source(value: object) -> str:
        raw = getattr(value, "value", value)
        if isinstance(raw, str):
            normalised = raw.casefold().replace("-", "_")
            if normalised in {"automatic", "manual", "annotated"}:
                return "annotation" if normalised == "annotated" else normalised
        try:
            return {0: "automatic", 1: "manual", 2: "annotation"}[int(raw)]
        except (KeyError, TypeError, ValueError):
            return "automatic"

    def _procedural_marker_records(
        self, *, include_current_manual: bool
    ) -> list[dict[str, object]]:
        """Describe actual watershed markers in full corrected coordinates."""

        result = self._analysis_result
        procedural = (
            None if result is None else getattr(result, "procedural_instances", None)
        )
        records: list[dict[str, object]] = []
        used_manual: set[int] = set()
        if procedural is not None:
            manual_match_tolerance = max(
                2.0,
                0.75 / max(1e-6, float(getattr(procedural, "working_scale", 1.0)))
                + 0.5,
            )
            marker_centres = np.asarray(
                getattr(procedural, "marker_centres_xy", ()), dtype=np.float64
            ).reshape(-1, 2)
            marker_sources = np.asarray(
                getattr(procedural, "marker_sources", ()), dtype=object
            ).reshape(-1)
            if len(marker_sources) != len(marker_centres):
                marker_sources = np.zeros(len(marker_centres), dtype=np.uint8)
            offset_x, offset_y = result.crop_offset
            for result_index, (local_point, source_value) in enumerate(
                zip(marker_centres, marker_sources, strict=True)
            ):
                point = QPointF(
                    float(local_point[0] + offset_x),
                    float(local_point[1] + offset_y),
                )
                source = self._normalised_procedural_marker_source(source_value)
                manual_index = None
                if source == "manual" and len(self._manual_seed_centres):
                    distances = np.hypot(
                        self._manual_seed_centres[:, 0] - point.x(),
                        self._manual_seed_centres[:, 1] - point.y(),
                    )
                    if used_manual:
                        distances[list(used_manual)] = np.inf
                    candidate = int(np.argmin(distances))
                    if float(distances[candidate]) <= manual_match_tolerance:
                        manual_index = candidate
                        used_manual.add(candidate)
                records.append(
                    {
                        "point": point,
                        "source": source,
                        "manual_index": manual_index,
                        "result_index": result_index,
                        "reason": "",
                    }
                )

            rejected = np.asarray(
                getattr(procedural, "rejected_manual_centres_xy", ()),
                dtype=np.float64,
            ).reshape(-1, 2)
            reasons = tuple(
                str(value)
                for value in getattr(
                    procedural, "rejected_manual_centre_reasons", ()
                )
            )
            for rejected_index, local_point in enumerate(rejected):
                point = QPointF(
                    float(local_point[0] + offset_x),
                    float(local_point[1] + offset_y),
                )
                manual_index = None
                if len(self._manual_seed_centres):
                    distances = np.hypot(
                        self._manual_seed_centres[:, 0] - point.x(),
                        self._manual_seed_centres[:, 1] - point.y(),
                    )
                    if used_manual:
                        distances[list(used_manual)] = np.inf
                    candidate = int(np.argmin(distances))
                    if float(distances[candidate]) <= manual_match_tolerance:
                        manual_index = candidate
                        used_manual.add(candidate)
                records.append(
                    {
                        "point": point,
                        "source": "rejected",
                        "manual_index": manual_index,
                        "result_index": None,
                        "reason": (
                            reasons[rejected_index]
                            if rejected_index < len(reasons)
                            else "Manual centre was rejected by procedural inference."
                        ),
                    }
                )

        if include_current_manual:
            for manual_index, (x, y) in enumerate(self._manual_seed_centres):
                if manual_index in used_manual:
                    continue
                records.append(
                    {
                        "point": QPointF(float(x), float(y)),
                        "source": "manual",
                        "manual_index": manual_index,
                        "result_index": None,
                        "reason": "Pending procedural recomputation.",
                    }
                )
        return records

    def _render_manual_seed_centre_editor(self, result) -> None:
        del result
        for record in self._procedural_marker_records(include_current_manual=True):
            self._render_manual_seed_centre_marker(record, editing=True)

    def _render_manual_seed_centre_marker(
        self, record: dict[str, object], *, editing: bool
    ) -> None:
        point = record["point"]
        source = str(record["source"])
        selected = False
        if self._manual_seed_centre_selected is not None:
            selected = (
                self._manual_seed_centre_selected.get("source") == source
                and self._manual_seed_centre_selected.get("manual_index")
                == record.get("manual_index")
                and self._manual_seed_centre_selected.get("result_index")
                == record.get("result_index")
            )
        colours = {
            "automatic": QColor("#43ddff"),
            "manual": QColor("#ffe04f"),
            "annotation": QColor("#ff55de"),
            "rejected": QColor("#ff4545"),
        }
        colour = colours.get(source, colours["automatic"])
        pen = QPen(QColor("#ffffff") if selected else colour, 3.0 if selected else 2.0)
        pen.setCosmetic(True)
        brush = QBrush(
            colour
            if source in {"manual", "annotation"}
            else Qt.BrushStyle.NoBrush
        )
        radius = 7.0 if editing else 5.0
        items: list[QGraphicsItem] = []
        if source == "annotation":
            item = self._scene.addRect(
                -radius,
                -radius,
                radius * 2.0,
                radius * 2.0,
                pen,
                brush,
            )
            items.append(item)
        else:
            item = self._scene.addEllipse(
                -radius,
                -radius,
                radius * 2.0,
                radius * 2.0,
                pen,
                brush,
            )
            items.append(item)
        if source == "manual":
            dark_pen = QPen(QColor("#493f00"), 1.5)
            dark_pen.setCosmetic(True)
            items.extend(
                (
                    self._scene.addLine(-3.5, 0.0, 3.5, 0.0, dark_pen),
                    self._scene.addLine(0.0, -3.5, 0.0, 3.5, dark_pen),
                )
            )
        elif source == "rejected":
            items.extend(
                (
                    self._scene.addLine(-5.0, -5.0, 5.0, 5.0, pen),
                    self._scene.addLine(-5.0, 5.0, 5.0, -5.0, pen),
                )
            )
        tooltips = {
            "automatic": "Automatic procedural watershed marker. Dragging or deleting it switches to Replace automatic mode.",
            "manual": "Manual procedural watershed marker. Drag to adjust; right-click or Delete removes it.",
            "annotation": "Locked marker from an applied seed-instance annotation. Edit the annotation to move it.",
            "rejected": str(record.get("reason") or "Rejected manual marker."),
        }
        for marker_item in items:
            marker_item.setPos(point)
            marker_item.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True
            )
            marker_item.setOpacity(1.0 if editing else self._overlay_opacity)
            marker_item.setZValue(46 if editing else 15)
            marker_item.setToolTip(tooltips[source])
            marker_item.setData(0, "manual-seed-centre-marker")
            self._overlay_items.append(marker_item)

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

    def _render_edge_oval_hypotheses(self, result) -> None:
        """Draw only proposal-independent ovals inferred from curved edges."""

        geometry = result.layers.edge_fit_geometry
        if geometry is None:
            return
        values = geometry.materialize()
        offset_x, offset_y = result.crop_offset
        for centre, axes, angle, confidence in zip(
            values["oval_centres_xy"],
            values["oval_axes_xy"],
            values["oval_angle_radians"],
            values["oval_confidence"],
            strict=True,
        ):
            confidence = float(confidence)
            if confidence <= 0.0:
                continue
            major, minor = float(axes[0]), float(axes[1])
            pen = QPen(QColor("#32e6ff"), 2.0 + 2.0 * confidence)
            pen.setCosmetic(True)
            item = self._scene.addEllipse(
                -major,
                -minor,
                major * 2.0,
                minor * 2.0,
                pen,
            )
            item.setPos(float(centre[0] + offset_x), float(centre[1] + offset_y))
            item.setRotation(float(np.rad2deg(angle)))
            item.setOpacity(
                self._overlay_opacity * min(1.0, 0.30 + 0.70 * confidence)
            )
            item.setZValue(20)
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
        outer_axes = tuple(
            float(value)
            for value in getattr(dish, "outer_axes", (outer_radius, outer_radius))
        )
        inner_axes = tuple(
            float(value)
            for value in getattr(dish, "inner_axes", (inner_radius, inner_radius))
        )
        edges = [(outer_axes, QColor("#41d9ff"))]
        if include_inner:
            edges.append((inner_axes, QColor("#ff63d8")))
        rendered_axes: set[tuple[int, int]] = set()
        for axes, colour in edges:
            axis_x, axis_y = (float(value) for value in axes)
            key = (round(axis_x), round(axis_y))
            if key in rendered_axes:
                continue
            rendered_axes.add(key)
            pen = QPen(colour)
            pen.setWidth(width)
            pen.setCosmetic(True)
            if not getattr(dish, "rim_pair_detected", False):
                pen.setStyle(Qt.PenStyle.DashLine)
            item = self._scene.addEllipse(
                dish.center_x - axis_x,
                dish.center_y - axis_y,
                axis_x * 2,
                axis_y * 2,
                pen,
            )
            if abs(axis_x - axis_y) >= 0.5:
                item.setTransformOriginPoint(dish.center_x, dish.center_y)
                item.setRotation(float(getattr(dish, "ellipse_angle_degrees", 0.0)))
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

    def _render_seed_shape(self, result, mode: str) -> None:
        summary = getattr(result, "seed_measurement_summary", None)
        model = getattr(result, "seed_dimensions_shape_model", None)
        if mode == "seed_scale_estimation":
            self._render_seed_scale(result)
            self._render_reviewed_shape_geometry(result, summary, uncertainty=False)
        elif mode == "seed_shape_uncertainty":
            self._render_reviewed_shape_geometry(result, summary, uncertainty=True)
        elif mode == "seed_boundary_curvature_distribution":
            self._render_reference_curvature_distribution(summary)
        elif mode == "seed_size_ovality_distribution":
            self._render_size_ovality_distribution(result, summary, model)
        elif mode == "seed_pose_shape_distributions":
            self._render_pose_shape_distributions(result, model)
        elif mode == "seed_mean_shape_atlas":
            self._render_mean_shape_atlas(result, model)

    def _contrast_overlay_text(self, text, x, y, size=20., colour="#ffffff", z=30):
        label = self._scene.addText(text)
        font = QFont(label.font())
        font.setPixelSize(max(12, round(size)))
        font.setBold(True)
        label.setFont(font)
        label.setDefaultTextColor(QColor(colour))
        label.setPos(x, y)
        label.setZValue(z + .1)
        background = self._scene.addRect(label.sceneBoundingRect().adjusted(-4, -2, 4, 2),
            QPen(Qt.PenStyle.NoPen), QBrush(QColor("#17202a")))
        background.setZValue(z)
        # Text stays legible independent of image/overlay blend opacity.
        self._overlay_items.extend((background, label))
        return label

    def _shape_scale_to_pixels(self, result, measurement) -> float:
        return (
            float(result.calibration.pixels_per_mm or 1.0)
            if measurement.length_unit == "mm"
            else 1.0
        )

    def _render_reviewed_shape_geometry(
        self, result, summary, *, uncertainty: bool
    ) -> None:
        if summary is None:
            return
        if uncertainty:
            self._contrast_overlay_text(
                "Annotation measurement sensitivity — not automatic seed predictions\n"
                "Green: fitted body. Yellow dashes: ±2× boundary/scale sensitivity.\n"
                "Cyan chord: maximum span; reported ± values are 1σ-equivalent, not guaranteed confidence limits.",
                35, 45, max(18, self.image_size[0]*.006))
        for reviewed in summary.observations:
            measurement = reviewed.measurement
            if measurement is None:
                continue
            scale = self._shape_scale_to_pixels(result, measurement)
            ellipse = measurement.ellipse
            center_x, center_y = ellipse.center_xy
            length = ellipse.body_length * scale
            width = ellipse.body_width * scale
            colour = QColor("#48f5a2") if reviewed.eligible else QColor("#ff5364")
            if uncertainty and reviewed.eligible:
                uncertainty_values = measurement.measurement_uncertainty
                length_uncertainty = uncertainty_values[1] * scale
                width_uncertainty = uncertainty_values[2] * scale
                outer_pen = QPen(QColor("#ffcc4d"), 2.5)
                outer_pen.setCosmetic(True)
                outer_pen.setStyle(Qt.PenStyle.DashLine)
                for sign in (-1, 1):
                    outer_length = max(1., length + sign * 4 * length_uncertainty)
                    outer_width = max(1., width + sign * 4 * width_uncertainty)
                    outer = self._scene.addEllipse(
                        center_x-outer_length*.5, center_y-outer_width*.5,
                        outer_length, outer_width, outer_pen)
                    outer.setTransformOriginPoint(center_x, center_y)
                    outer.setRotation(ellipse.orientation_degrees)
                    outer.setOpacity(self._overlay_opacity)
                    outer.setZValue(13)
                    outer.setToolTip("Two-sided ±2× sensitivity envelope, not a segmentation prediction.")
                    self._overlay_items.append(outer)
            body_pen = QPen(colour, 3.0)
            body_pen.setCosmetic(True)
            body = self._scene.addEllipse(
                center_x - length * 0.5,
                center_y - width * 0.5,
                length,
                width,
                body_pen,
            )
            body.setTransformOriginPoint(center_x, center_y)
            body.setRotation(ellipse.orientation_degrees)
            body.setOpacity(self._overlay_opacity)
            body.setVisible(reviewed.eligible)
            body.setZValue(14)
            body.setToolTip(
                f"Seed {reviewed.seed_id}: robust body {ellipse.body_length:.2f} × "
                f"{ellipse.body_width:.2f} {measurement.length_unit}; ovality "
                f"{measurement.ovality:.3f}; pose {reviewed.annotation.pose}; "
                f"support {ellipse.support_fraction:.0%}."
            )
            self._overlay_items.append(body)
            first, second = measurement.maximum_span_endpoints
            span_pen = QPen(QColor("#42d9ff"), 3.0)
            span_pen.setCosmetic(True)
            span = self._scene.addLine(
                first[0], first[1], second[0], second[1], span_pen
            )
            span.setOpacity(self._overlay_opacity)
            span.setZValue(15)
            self._overlay_items.append(span)
            for endpoint in (first, second):
                radius = max(2., length * .025)
                marker = self._scene.addEllipse(endpoint[0]-radius, endpoint[1]-radius,
                    radius*2, radius*2, span_pen)
                marker.setZValue(15)
                self._overlay_items.append(marker)
            text = (f"Seed {reviewed.seed_id}: span {measurement.maximum_span:.2f} "
                    f"±{measurement.measurement_uncertainty[0]:.2f} {measurement.length_unit}")
            if reviewed.eligible:
                text += (f"\novality {measurement.ovality:.3f} ±{measurement.measurement_uncertainty[3]:.3f}"
                         f"; concavity {measurement.concavity_fraction:.1%}")
            else:
                text += "\nSpan only; shape not eligible"
            self._contrast_overlay_text(text, center_x-length*.5, center_y-width*.5-length*.28,
                                        max(12., length*.08), "#42d9ff", 18)
            if reviewed.annotation.hilum_point is not None:
                hx, hy = reviewed.annotation.hilum_point
                hilum = self._scene.addEllipse(
                    hx - 5, hy - 5, 10, 10,
                    QPen(QColor("#ff5de4"), 3.0),
                    QBrush(QColor("#ff5de4")),
                )
                hilum.setOpacity(self._overlay_opacity)
                hilum.setZValue(16)
                hilum.setToolTip("Reviewer-supplied hilum landmark.")
                self._overlay_items.append(hilum)

    def _shape_chart_panel(self, title: str) -> tuple[QRectF, float]:
        image_size = self.image_size
        if image_size is None:
            return QRectF(), 0.0
        image_width, image_height = image_size
        margin = max(24.0, min(image_width, image_height) * 0.025)
        panel = QRectF(
            margin,
            margin,
            min(image_width - 2 * margin, max(520.0, image_width * 0.48)),
            min(image_height - 2 * margin, max(330.0, image_height * 0.42)),
        )
        background = QColor("#111923")
        background.setAlpha(232)
        item = self._scene.addRect(
            panel, QPen(QColor("#526374"), 2.0), QBrush(background)
        )
        item.setOpacity(self._overlay_opacity)
        item.setZValue(20)
        self._overlay_items.append(item)
        label = self._scene.addText(title)
        font = QFont(label.font())
        font.setBold(True)
        font.setPixelSize(max(18, round(panel.height() * 0.07)))
        label.setFont(font)
        label.setDefaultTextColor(QColor("#f3f6f8"))
        label.setPos(panel.left() + 16, panel.top() + 10)
        label.setOpacity(self._overlay_opacity)
        label.setZValue(21)
        self._overlay_items.append(label)
        return panel.adjusted(panel.width()*.16, panel.height()*.22,
                              -panel.width()*.06, -panel.height()*.26), 21.0

    def _render_size_ovality_distribution(self, result, summary, model) -> None:
        plot, z = self._shape_chart_panel(
            "Reviewed maximum span versus ovality"
        )
        if summary is None or plot.isEmpty():
            return
        points = [
            (item.measurement.maximum_span, item.measurement.ovality, item)
            for item in summary.observations
            if item.eligible and item.measurement is not None
        ]
        if not points:
            note = self._scene.addText("No explicitly reviewed complete shapes.")
            note.setDefaultTextColor(QColor("#ffcc4d"))
            note.setPos(plot.left(), plot.top())
            note.setZValue(z + 1)
            self._overlay_items.append(note)
            return
        x_values = np.asarray([item[0] for item in points], float)
        y_values = np.asarray([item[1] for item in points], float)
        x_errors = np.asarray([p[2].measurement.measurement_uncertainty[0] for p in points])
        y_errors = np.asarray([p[2].measurement.measurement_uncertainty[3] for p in points])
        x_min, x_max = float((x_values-x_errors).min()), float((x_values+x_errors).max())
        y_min, y_max = float((y_values-y_errors).min()), float((y_values+y_errors).max())
        x_pad = max((x_max - x_min) * 0.12, max(abs(x_max), 1.0) * 0.03)
        y_pad = max((y_max - y_min) * 0.12, 0.03)
        font_size = max(13., plot.height()*.065)
        axis_pen = QPen(QColor("#a7b7c7"), 1.)
        axis_pen.setCosmetic(True)
        for fraction in np.linspace(0., 1., 5):
            x = plot.left() + fraction*plot.width()
            y = plot.bottom() - fraction*plot.height()
            for coords in ((x, plot.top(), x, plot.bottom()),
                           (plot.left(), y, plot.right(), y)):
                item = self._scene.addLine(*coords, axis_pen)
                item.setZValue(z)
                self._overlay_items.append(item)
            self._contrast_overlay_text(f"{x_min-x_pad+fraction*(x_max-x_min+2*x_pad):.2f}",
                                        x-font_size, plot.bottom()+font_size*.4, font_size, z=z+1)
            self._contrast_overlay_text(f"{y_min-y_pad+fraction*(y_max-y_min+2*y_pad):.2f}",
                                        plot.left()-font_size*4, y-font_size*.5, font_size, z=z+1)
        self._contrast_overlay_text(f"Maximum span ({summary.length_unit})",
            plot.center().x()-font_size*5, plot.bottom()+font_size*2, font_size, z=z+1)
        self._contrast_overlay_text("Ovality = body length / width",
            plot.left(), plot.top()-font_size*2, font_size, z=z+1)
        self._contrast_overlay_text(
            f"n={len(points)} complete shapes · bars: ±1σ sensitivity\n"
            "Green: flat · Cyan: oblique · Orange: side · Purple: uncertain",
            plot.left(), plot.bottom()+font_size*4, font_size, z=z+1)
        for x, y, reviewed in points:
            px = plot.left() + (x - x_min + x_pad) / max(x_max - x_min + 2 * x_pad, 1e-9) * plot.width()
            py = plot.bottom() - (y - y_min + y_pad) / max(y_max - y_min + 2 * y_pad, 1e-9) * plot.height()
            colour = {
                "flat": "#48f5a2", "oblique": "#42d9ff",
                "side": "#ffb84a", "uncertain": "#c58cff",
            }.get(reviewed.annotation.pose, "#a8b2bd")
            dx = reviewed.measurement.measurement_uncertainty[0] / (x_max-x_min+2*x_pad)*plot.width()
            dy = reviewed.measurement.measurement_uncertainty[3] / (y_max-y_min+2*y_pad)*plot.height()
            for coords in ((px-dx, py, px+dx, py), (px, py-dy, px, py+dy)):
                bar = self._scene.addLine(*coords, QPen(QColor(colour), 2))
                bar.setZValue(z+1)
                self._overlay_items.append(bar)
            point = self._scene.addEllipse(
                px - 5, py - 5, 10, 10,
                QPen(QColor(colour), 2), QBrush(QColor(colour)),
            )
            point.setZValue(z + 1)
            point.setOpacity(self._overlay_opacity)
            point.setToolTip(
                f"Seed {reviewed.seed_id}: span {x:.3f}, ovality {y:.3f}, "
                f"pose {reviewed.annotation.pose}."
            )
            self._overlay_items.append(point)

    def _render_reference_curvature_distribution(self, summary):
        plot, z = self._shape_chart_panel("Reference perimeter curvature")
        if plot.isEmpty():
            return
        values = np.asarray(getattr(summary, "boundary_curvature_times_diameter", ()), float)
        size = max(13., plot.height()*.06)
        if not len(values):
            self._contrast_overlay_text("No complete reviewed outlines", plot.left(), plot.top(), size, z=z+1)
            return
        low, high = float(values.min()), float(values.max())
        high = max(high, low + .01)
        counts, bins = np.histogram(values, bins=32, range=(low, high))
        for i, count in enumerate(counts):
            height = plot.height()*float(count)/max(float(counts.max()), 1.)
            bar = self._scene.addRect(plot.left()+i*plot.width()/32, plot.bottom()-height,
                plot.width()/32-1, height, QPen(Qt.PenStyle.NoPen),
                QBrush(QColor("#ff6c73" if bins[i+1] <= 0 else "#48f5a2")))
            bar.setToolTip(f"kD [{bins[i]:.2f}, {bins[i+1]:.2f}): {count} samples")
            bar.setZValue(z+1)
            self._overlay_items.append(bar)
        for fraction in (0., .25, .5, .75, 1.):
            self._contrast_overlay_text(f"{low+fraction*(high-low):.2f}",
                plot.left()+fraction*plot.width()-size, plot.bottom()+size*.5, size, z=z+2)
        turns = np.quantile(summary.boundary_turn_degrees, [.1, .5, .9])
        self._contrast_overlay_text(
            f"Equal samples / seed; {summary.eligible_count} complete references; y = sample count\n"
            f"Mean internal concavity: {summary.mean_internal_concavity:.1%}",
            plot.left(), plot.top()-size*3, size, z=z+2)
        self._contrast_overlay_text(
            "Curvature × maximum span (kD); negative = concave, positive = convex\n"
            f"Turning angles (10 / 50 / 90%): {turns[0]:.1f}° / {turns[1]:.1f}° / {turns[2]:.1f}°",
            plot.left(), plot.bottom()+size*2.5, size, z=z+2)

    def _render_pose_shape_distributions(self, result, model) -> None:
        plot, z = self._shape_chart_panel("Pose-conditioned predictive bodies")
        if model is None or not model.families or plot.isEmpty():
            return
        families = model.families
        cell_width = plot.width() / max(len(families), 1)
        pixels_per_mm = float(result.calibration.pixels_per_mm or 1.0)
        for index, family in enumerate(families):
            component = family.component
            if component.physical_dimensions_available:
                length = float(component.mean[1]) * pixels_per_mm
                width = float(component.mean[2]) * pixels_per_mm
                dimension_label = "calibrated dimensions"
            else:
                length = 1.0
                width = 1.0 / max(float(component.mean[3]), 1.0)
                dimension_label = "dimensionless shape only"
            max_size = min(cell_width * 0.72, plot.height() * 0.58)
            scale = max_size / max(length, width, 1e-9)
            center_x = plot.left() + (index + 0.5) * cell_width
            center_y = plot.center().y()
            body = self._scene.addEllipse(
                center_x - length * scale * 0.5,
                center_y - width * scale * 0.5,
                length * scale,
                width * scale,
                QPen(QColor("#42d9ff"), 3.0),
                QBrush(QColor(66, 217, 255, 45)),
            )
            body.setZValue(z + 1)
            body.setOpacity(self._overlay_opacity)
            body.setToolTip(
                f"{family.pose}: {component.effective_physical_seed_count:.1f} "
                f"effective physical seed(s), {component.source_count} source(s); "
                f"{dimension_label}."
            )
            self._overlay_items.append(body)
            label = self._scene.addText(family.pose)
            label.setDefaultTextColor(QColor("#f3f6f8"))
            label.setPos(center_x - label.boundingRect().width() * 0.5, plot.bottom() + 8)
            label.setZValue(z + 1)
            self._overlay_items.append(label)

    def _render_mean_shape_atlas(self, result, model) -> None:
        plot, z = self._shape_chart_panel("Mean shape and retained contour modes")
        if model is None or not model.families or plot.isEmpty():
            return
        family = model.family("flat") or model.families[0]
        component = family.component
        signatures = [np.asarray(component.contour_mean, float)]
        labels = [f"{family.pose} mean"]
        for index, (mode, variance) in enumerate(
            zip(component.contour_modes[:2], component.contour_variances[:2])
        ):
            amplitude = np.sqrt(max(float(variance), 0.0))
            signatures.extend(
                (
                    signatures[0] + np.asarray(mode) * amplitude,
                    signatures[0] - np.asarray(mode) * amplitude,
                )
            )
            labels.extend((f"mode {index + 1} +", f"mode {index + 1} −"))
        if not signatures[0].size:
            return
        cell_width = plot.width() / len(signatures)
        for index, (signature, label_text) in enumerate(zip(signatures, labels)):
            angles = np.linspace(0.0, 2.0 * np.pi, len(signature), endpoint=False)
            radius = np.maximum(0.15, 1.0 + signature)
            x_radius = cell_width * 0.34
            y_radius = min(plot.height() * 0.34, x_radius / max(float(component.mean[3]), 1.0))
            center = QPointF(plot.left() + (index + 0.5) * cell_width, plot.center().y())
            points = [
                QPointF(
                    center.x() + np.cos(angle) * rad * x_radius,
                    center.y() + np.sin(angle) * rad * y_radius,
                )
                for angle, rad in zip(angles, radius)
            ]
            path = QPainterPath(points[0])
            for point in points[1:]:
                path.lineTo(point)
            path.closeSubpath()
            item = self._scene.addPath(
                path,
                QPen(QColor("#48f5a2"), 3.0),
                QBrush(QColor(72, 245, 162, 38)),
            )
            item.setZValue(z + 1)
            item.setOpacity(self._overlay_opacity)
            self._overlay_items.append(item)
            label = self._scene.addText(label_text)
            label.setDefaultTextColor(QColor("#f3f6f8"))
            label.setPos(center.x() - label.boundingRect().width() * 0.5, plot.bottom() + 8)
            label.setZValue(z + 1)
            self._overlay_items.append(label)

    def _render_seed_scale(self, result) -> None:
        x0, y0, x1, y1 = result.reference_roi
        roi_pen = QPen(QColor("#41d9ff"), 5)
        roi_pen.setCosmetic(True)
        rectangle = self._scene.addRect(x0, y0, x1 - x0, y1 - y0, roi_pen)
        rectangle.setOpacity(self._overlay_opacity)
        rectangle.setZValue(10)
        self._overlay_items.append(rectangle)
        diameter = float(result.estimated_seed_diameter_px)
        bounds = tuple(getattr(result, "reference_seed_bounds", ()))
        fitted_diameters = tuple(
            float(value)
            for value in getattr(result, "reference_seed_diameters_px", ())
        )
        if bounds:
            seed_rects = tuple(
                QRectF(
                    float(left),
                    float(top),
                    float(right) - float(left),
                    float(bottom) - float(top),
                )
                for left, top, right, bottom in bounds
            )
        else:
            centre = QPointF(x0 + diameter * 0.7, y0 + diameter * 0.7)
            seed_rects = (
                QRectF(
                    centre.x() - diameter * 0.5,
                    centre.y() - diameter * 0.5,
                    diameter,
                    diameter,
                ),
            )
        scale_pen = QPen(QColor("#ffd84a"), 3)
        scale_pen.setCosmetic(True)
        for index, seed_rect in enumerate(seed_rects):
            circle = self._scene.addEllipse(seed_rect, scale_pen)
            circle.setOpacity(self._overlay_opacity)
            circle.setZValue(11)
            self._overlay_items.append(circle)
            fitted_diameter = (
                fitted_diameters[index]
                if index < len(fitted_diameters)
                else diameter
            )
            diameter_text = (
                f"Initial fit: {fitted_diameter:.1f}".rstrip("0").rstrip(".") + " px"
            )
            label = self._scene.addText(diameter_text)
            label_font = QFont(label.font())
            label_font.setPixelSize(max(12, min(36, round(diameter * 0.16))))
            label_font.setBold(True)
            label.setFont(label_font)
            label.setDefaultTextColor(QColor("#ffd84a"))
            label_bounds = label.boundingRect()
            label.setPos(
                seed_rect.center().x() - label_bounds.width() * 0.5,
                seed_rect.top()
                - label_bounds.height()
                - max(2.0, diameter * 0.025),
            )
            label_background_colour = QColor("#17202a")
            label_background_colour.setAlpha(224)
            label_scene_bounds = label.sceneBoundingRect().adjusted(
                -6.0, -3.0, 6.0, 3.0
            )
            label_background = self._scene.addRect(
                label_scene_bounds,
                QPen(Qt.PenStyle.NoPen),
                QBrush(label_background_colour),
            )
            label_background.setOpacity(self._overlay_opacity)
            label_background.setZValue(10.9)
            label_background.setToolTip(
                "Contrast background for the isolated reference diameter label."
            )
            self._overlay_items.append(label_background)
            label.setOpacity(self._overlay_opacity)
            label.setZValue(11)
            label.setToolTip(
                f"Locally fitted isolated-seed width: {fitted_diameter:.1f} pixels."
            )
            self._overlay_items.append(label)

        measurements = tuple(
            getattr(result, "annotated_seed_diameters", ())
        )
        for measurement in measurements:
            colour = (
                QColor("#52f58a")
                if measurement.selected and measurement.complete
                else QColor("#ff4d5f")
                if not measurement.complete
                else QColor("#a8b2bd")
            )
            pen = QPen(colour, 4.0 if measurement.selected else 2.5)
            pen.setCosmetic(True)
            if not measurement.complete:
                pen.setStyle(Qt.PenStyle.DashLine)
            endpoint_a = measurement.endpoint_a
            endpoint_b = measurement.endpoint_b
            line = self._scene.addLine(
                float(endpoint_a[0]),
                float(endpoint_a[1]),
                float(endpoint_b[0]),
                float(endpoint_b[1]),
                pen,
            )
            line.setOpacity(self._overlay_opacity)
            line.setZValue(12)
            line.setToolTip(
                f"Annotated seed {measurement.identifier}: maximum width "
                f"{measurement.diameter_px:.1f}px; "
                + (
                    "selected for final estimate"
                    if measurement.selected and measurement.complete
                    else "excluded: full length is not confirmed or mask is disconnected"
                    if not measurement.complete
                    else "not used in size model"
                )
            )
            self._overlay_items.append(line)
            midpoint_x = (float(endpoint_a[0]) + float(endpoint_b[0])) * 0.5
            midpoint_y = (float(endpoint_a[1]) + float(endpoint_b[1])) * 0.5
            tag = self._scene.addText(
                f"{measurement.identifier}: {measurement.diameter_px:.0f}px"
            )
            tag_font = QFont(tag.font())
            tag_font.setPixelSize(max(11, min(24, round(diameter * 0.10))))
            tag_font.setBold(bool(measurement.selected))
            tag.setFont(tag_font)
            tag.setDefaultTextColor(colour)
            tag.setPos(midpoint_x + 4.0, midpoint_y + 4.0)
            tag.setOpacity(self._overlay_opacity)
            tag.setZValue(12)
            self._overlay_items.append(tag)

        self._render_seed_diameter_histogram(result, measurements)

    def _render_seed_diameter_histogram(self, result, measurements) -> None:
        """Draw the annotated-width distribution and authoritative estimate."""

        diameter = float(result.estimated_seed_diameter_px)
        image_size = self.image_size
        if image_size is None:
            return
        image_width, image_height = image_size
        panel_width = max(360.0, min(760.0, diameter * 4.8))
        panel_height = max(190.0, min(390.0, diameter * 2.5))
        margin = max(24.0, min(image_width, image_height) * 0.025)
        panel_x = margin
        panel_y = margin
        background = QColor("#111923")
        background.setAlpha(220)
        panel = self._scene.addRect(
            panel_x,
            panel_y,
            panel_width,
            panel_height,
            QPen(QColor("#d7e1ec"), 2.0),
            QBrush(background),
        )
        panel.setZValue(16)
        panel.setOpacity(self._overlay_opacity)
        self._overlay_items.append(panel)

        summary = getattr(result, "seed_measurement_summary", None)
        statistics = None if summary is None else summary.size_statistics
        title_text = f"Final mean maximum span: {diameter:.1f}px\n"
        if statistics is not None:
            mean, sd, mean_error = statistics
            title_text += (f"{mean:.2f} ±{mean_error:.2f} {summary.length_unit} (mean uncertainty)\n"
                           f"Seed SD: {sd:.2f}; n={len(summary.size_observations)}; full distribution")
        else:
            title_text += f"Source: {getattr(result, 'seed_diameter_source', 'unknown')}"
        title = self._scene.addText(title_text)
        title_font = QFont(title.font())
        title_font.setPixelSize(max(14, min(30, round(panel_height * 0.095))))
        title_font.setBold(True)
        title.setFont(title_font)
        title.setDefaultTextColor(QColor("#ffffff"))
        title.setPos(panel_x + 14.0, panel_y + 8.0)
        title.setZValue(17)
        title.setOpacity(self._overlay_opacity)
        self._overlay_items.append(title)

        widths = np.asarray(
            [
                float(item.diameter_px)
                for item in measurements
                if item.complete and item.diameter_px > 0.0
            ],
            dtype=np.float64,
        )
        if not widths.size:
            empty = self._scene.addText("No complete annotated instances")
            empty.setDefaultTextColor(QColor("#a8b2bd"))
            empty.setPos(panel_x + 14.0, panel_y + panel_height * 0.58)
            empty.setZValue(17)
            empty.setOpacity(self._overlay_opacity)
            self._overlay_items.append(empty)
            return

        plot_x = panel_x + 24.0
        plot_y = panel_y + panel_height * 0.55
        plot_width = panel_width - 48.0
        plot_height = panel_height * 0.31
        bin_count = max(3, min(12, round(np.sqrt(widths.size) * 1.7)))
        low = min(float(widths.min()), diameter) * 0.94
        high = max(float(widths.max()), diameter) * 1.06
        if high <= low:
            high = low + 1.0
        counts, edges = np.histogram(widths, bins=bin_count, range=(low, high))
        maximum_count = max(1, int(counts.max(initial=0)))
        bin_width = plot_width / bin_count
        for index, count in enumerate(counts.tolist()):
            bar_height = plot_height * float(count) / maximum_count
            bar = self._scene.addRect(
                plot_x + index * bin_width + 1.0,
                plot_y + plot_height - bar_height,
                max(1.0, bin_width - 2.0),
                bar_height,
                QPen(Qt.PenStyle.NoPen),
                QBrush(QColor("#7895ad")),
            )
            bar.setZValue(17)
            bar.setOpacity(self._overlay_opacity)
            self._overlay_items.append(bar)
        estimate_x = plot_x + (diameter - low) / (high - low) * plot_width
        estimate_line = self._scene.addLine(
            estimate_x,
            plot_y - 3.0,
            estimate_x,
            plot_y + plot_height + 3.0,
            QPen(QColor("#52f58a"), 4.0),
        )
        estimate_line.setZValue(18)
        estimate_line.setOpacity(self._overlay_opacity)
        self._overlay_items.append(estimate_line)
        axis = self._scene.addText(f"{low:.0f}px                 {high:.0f}px")
        axis.setDefaultTextColor(QColor("#d7e1ec"))
        axis.setPos(plot_x, plot_y + plot_height + 3.0)
        axis.setZValue(17)
        axis.setOpacity(self._overlay_opacity)
        self._overlay_items.append(axis)

    def _render_context_annotations(
        self, result, *, include_scale: bool = False
    ) -> None:
        if include_scale:
            self._render_scale_bar(result)
        if self._material_reference_annotations_visible:
            if (
                self._reference_point_mode == "background"
                and self._automatic_background_reference_visible
                and self._render_automatic_background_reference_area(result)
            ):
                if self._overlay_mode != "perimeter_background_reference":
                    # The outline communicates the configured annulus; the
                    # translucent pixels below show the colour-filtered subset
                    # that was actually retained.
                    self._render_background_sampling_band(result, fill=False)
            self._render_background_reference_points()
            self._render_foreground_reference_points()
            self._render_exclusion_masks()
        self._render_instance_annotations()
        if self._manual_seed_centre_editing:
            self._render_manual_seed_centre_editor(result)

    def _render_automatic_background_reference_area(self, result) -> bool:
        """Render a coverage-preserving view of the retained source mask."""

        band = getattr(result, "perimeter_background_band", None)
        ring_raster = (
            None if band is None else getattr(band, "accepted_sample_mask", None)
        )
        if ring_raster is None:
            return False
        def render_mask(raster, offset: tuple[int, int], tooltip: str) -> bool:
            if raster is None:
                return False
            values = np.asarray(raster)
            if values.ndim != 2:
                values = np.squeeze(values)
            if values.ndim != 2 or not bool(np.any(values > 0)):
                return False
            mask = values > 0
            source_height, source_width = mask.shape
            maximum_display_dimension = 2048
            if max(source_height, source_width) > maximum_display_dimension:
                factor = maximum_display_dimension / max(source_height, source_width)
                display_width = max(1, round(source_width * factor))
                display_height = max(1, round(source_height * factor))
                display_mask = cv2.resize(
                    mask.astype(np.float32),
                    (display_width, display_height),
                    interpolation=cv2.INTER_AREA,
                ) > 0.0
            else:
                display_mask = mask
            rgba = np.zeros((*display_mask.shape, 4), dtype=np.uint8)
            rgba[:, :, :3] = (65, 217, 255)
            rgba[:, :, 3] = np.uint8(display_mask) * 92
            height, width = display_mask.shape
            image = QImage(
                rgba.data,
                width,
                height,
                int(rgba.strides[0]),
                QImage.Format.Format_RGBA8888,
            ).copy()
            item = self._scene.addPixmap(QPixmap.fromImage(image))
            item.setPos(*offset)
            item.setTransformationMode(Qt.TransformationMode.FastTransformation)
            item.setTransform(
                QTransform.fromScale(source_width / width, source_height / height)
            )
            item.setOpacity(self._overlay_opacity)
            item.setZValue(29)
            item.setToolTip(tooltip)
            item.setData(0, "automatic-background-reference-area")
            self._overlay_items.append(item)
            return True

        rendered_ring = render_mask(
            ring_raster,
            (0, 0),
            "Colour-filtered pixels retained from the perimeter sampling ring; "
            "foreign colours such as calibration-card patches are excluded.",
        )
        if rendered_ring:
            for item in self._overlay_items[-1:]:
                item.setToolTip(
                    item.toolTip()
                    + " Green areas are manually painted Background references."
                )
        return rendered_ring

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
            (255, 173, 85),
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
            "other": QColor(255, 173, 85, 205),
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
            round(float(profile.material_context_radius_px), 4),
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
            top = 174
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
                QRectF(40, 47, canvas_width - 80, 112),
                Qt.AlignmentFlag.AlignHCenter
                | Qt.AlignmentFlag.AlignTop
                | Qt.TextFlag.TextWordWrap,
                "Thumbnails are visual context only; the whole square is never a "
                "matching template. Material references contribute candidate centre "
                "pixels: each medoid stores one per-pixel Lab/noise/edge/ridge vector "
                f"plus local residual/density context (radius {profile.material_context_radius_px:.1f} source px). "
                "Each edge medoid is an amalgam of three separate tangent-aligned "
                f"lines (side offset {profile.edge_strip_normal_offset_px:.1f} px; "
                f"half-length {profile.edge_strip_tangent_half_length_px:.1f} px). "
                "Exactly five weighted points are pooled on each yellow/cyan line; "
                "pixels between the lines are not sampled. At every query pixel, all "
                "retained medoids are compared and the best supported class similarity "
                "is used; edge inference tests both side assignments.",
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
            "matched as image patches. Material medoids are individual centre feature "
            "vectors with local residual/density context. On tangent-aligned edge "
            "thumbnails, each yellow/cyan line pools five weighted points; the area "
            "between lines is not sampled. Matching evaluates both possible side "
            "assignments."
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
        self._render_imperial_measurement(calibration, ruler)
        point_a = QPointF(float(endpoints[0, 0]), float(endpoints[0, 1]))
        point_b = QPointF(float(endpoints[1, 0]), float(endpoints[1, 1]))
        line_pen = QPen(QColor("#ff58cc"), 11)
        line = self._scene.addLine(QLineF(point_a, point_b), line_pen)
        line.setZValue(13)
        line.setOpacity(self._overlay_opacity)
        self._overlay_items.append(line)
        tick_points = calibration.ruler_tick_points_corrected()
        if tick_points.size:
            angle = np.deg2rad(float(ruler.angle_degrees) + 90.0)
            half_length = max(4.0, ruler.width_px * 0.035)
            offset_x = float(np.cos(angle) * half_length)
            offset_y = float(np.sin(angle) * half_length)
            tick_pen = QPen(QColor("#42dff5"), max(2.0, ruler.width_px * 0.008))
            for x, y in tick_points:
                tick = self._scene.addLine(
                    float(x) - offset_x,
                    float(y) - offset_y,
                    float(x) + offset_x,
                    float(y) + offset_y,
                    tick_pen,
                )
                tick.setZValue(14)
                tick.setOpacity(self._overlay_opacity * 0.80)
                self._overlay_items.append(tick)
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

    def _render_imperial_measurement(self, calibration, ruler) -> None:
        imperial = calibration.imperial_measurement_endpoints_corrected()
        if len(imperial) == 2:
            pen = QPen(QColor("#ff961f"), 3)
            pen.setCosmetic(True)
            if not ruler.imperial_scale_reliable:
                pen.setStyle(Qt.PenStyle.DashLine)
            measurement_line = self._scene.addLine(
                QLineF(QPointF(*imperial[0]), QPointF(*imperial[1])), pen)
            measurement_line.setZValue(14)
            measurement_line.setOpacity(self._overlay_opacity)
            self._overlay_items.append(measurement_line)
            for x, y in imperial:
                radius = max(7., ruler.width_px * .04)
                item = self._scene.addEllipse(float(x)-radius, float(y)-radius,
                                             2*radius, 2*radius, pen)
                item.setZValue(15)
                item.setOpacity(self._overlay_opacity)
                self._overlay_items.append(item)
            scale = float(ruler.imperial_pixels_per_mm)
            span = float(np.linalg.norm(imperial[1]-imperial[0])) / (scale*25.4) if scale > 0 else None
            label = "Imperial observed span" + (f": {span:.2f} in" if span is not None else ": scale unresolved")
            if not ruler.imperial_scale_reliable:
                label += " (unconfirmed)"
            self._contrast_overlay_text(label, float(imperial[0, 0]), float(imperial[0, 1])+12,
                                        max(16, ruler.width_px*.09), "#ffb653", 16)

    def _render_ruler_evidence(self, result) -> None:
        """Draw independently classified ruler evidence in fixed semantic colours."""
        calibration = getattr(result, "calibration", None)
        ruler = None if calibration is None else calibration.ruler
        if ruler is None:
            return

        def polygon(points, colour: str, width: float, z: float) -> None:
            if len(points) < 3:
                return
            item = self._scene.addPolygon(
                QPolygonF([QPointF(float(x), float(y)) for x, y in points]),
                QPen(QColor(colour), width),
            )
            item.setZValue(z)
            item.setOpacity(self._overlay_opacity)
            self._overlay_items.append(item)

        polygon(ruler.outline_points, "#28d65f", 5.0, 13.0)
        angle = np.deg2rad(float(ruler.angle_degrees) + 90.0)
        half_length = max(4.0, float(ruler.width_px) * 0.055)
        offset_x = float(np.cos(angle) * half_length)
        offset_y = float(np.sin(angle) * half_length)

        def ticks(
            points,
            segments,
            classes,
            unit_dividers,
            family: str,
            colour: str,
            z: float,
        ) -> None:
            pen = QPen(QColor(colour), max(2.0, float(ruler.width_px) * 0.010))
            unit_pen = QPen(
                QColor(colour), max(3.0, float(ruler.width_px) * 0.016)
            )
            unit_divider_set = {int(value) for value in unit_dividers}
            if segments:
                lines = segments
            else:
                lines = tuple(
                    (
                        (float(x) - offset_x, float(y) - offset_y),
                        (float(x) + offset_x, float(y) + offset_y),
                    )
                    for x, y in points
                )
            class_names = (
                ("1 mm", "5 mm", "1 cm")
                if family == "metric"
                else ("1/16 in", "1/8 in", "1/4 in", "1/2 in", "1 in")
            )
            for index, (outer, inner) in enumerate(lines):
                draw_outer = (float(outer[0]), float(outer[1]))
                if family == "imperial":
                    # The measured root is the last dark-response centre, not
                    # the outside end of the printed bar. Extend only the
                    # display stroke a few pixels outward so the bottom row is
                    # visibly complete without changing tick pitch, length
                    # classes, ruler geometry, or absolute-scale analysis.
                    dx = float(outer[0]) - float(inner[0])
                    dy = float(outer[1]) - float(inner[1])
                    length = max(float(np.hypot(dx, dy)), 1e-6)
                    extension = max(
                        2.0, min(6.0, float(ruler.width_px) * 0.018)
                    )
                    draw_outer = (
                        float(outer[0]) + dx * extension / length,
                        float(outer[1]) + dy * extension / length,
                    )
                item = self._scene.addLine(
                    draw_outer[0],
                    draw_outer[1],
                    float(inner[0]),
                    float(inner[1]),
                    unit_pen if index in unit_divider_set else pen,
                )
                item.setZValue(z)
                item.setOpacity(self._overlay_opacity)
                if index < len(classes):
                    level = int(classes[index])
                    name = class_names[min(level, len(class_names) - 1)]
                    item.setToolTip(
                        f"{family.title()} tick {index}: inferred {name} length class"
                        + ("; unit divider" if index in unit_divider_set else "")
                    )
                self._overlay_items.append(item)

        ticks(
            ruler.metric_tick_points,
            ruler.metric_tick_segments,
            ruler.metric_tick_classes,
            ruler.metric_unit_divider_indices,
            "metric",
            "#ff3038",
            14.0,
        )
        ticks(
            ruler.imperial_tick_points,
            ruler.imperial_tick_segments,
            ruler.imperial_tick_classes,
            ruler.imperial_unit_divider_indices,
            "imperial",
            "#ff961f",
            14.0,
        )
        for box in ruler.metric_number_boxes:
            polygon(box, "#82131b", 3.0, 15.0)
        for box in ruler.imperial_number_boxes:
            polygon(box, "#9a4a00", 3.0, 15.0)

        def semantic_labels(labels, colour: str) -> None:
            for label_info in labels:
                if not label_info.observed and label_info.kind != "unit":
                    # Retain inferred values in metadata/status while keeping
                    # the image overlay focused on positions corroborated by
                    # actual printed glyph evidence.
                    continue
                label = self._scene.addText(
                    label_info.text
                    + (f" {label_info.unit}" if label_info.unit else "")
                )
                font = QFont(label.font())
                font.setPixelSize(max(12, min(28, round(float(ruler.width_px) * 0.10))))
                font.setBold(bool(label_info.observed))
                label.setFont(font)
                label.setDefaultTextColor(QColor(colour))
                position = QPointF(*label_info.position)
                label.setPos(position + QPointF(4.0, 3.0))
                label.setZValue(16)
                label.setOpacity(
                    self._overlay_opacity * (1.0 if label_info.observed else 0.68)
                )
                label.setToolTip(
                    ("Printed glyphs corroborate" if label_info.observed else "Tick lattice infers")
                    + f" this {label_info.text} {label_info.unit} divider."
                )
                self._overlay_items.append(label)

        semantic_labels(ruler.metric_labels, "#82131b")
        semantic_labels(ruler.imperial_labels, "#9a4a00")

        metric_scale = float(getattr(ruler, "metric_pixels_per_mm", 0.0))
        imperial_scale = float(getattr(ruler, "imperial_pixels_per_mm", 0.0))
        disagreement = getattr(ruler, "scale_disagreement_percent", None)
        lines = [
            f"Metric: {metric_scale:.3f} px/mm"
            if metric_scale > 0.0
            else "Metric: unresolved",
            f"Imperial: {imperial_scale:.3f} px/mm"
            if imperial_scale > 0.0
            else "Imperial: unresolved",
            (
                f"Cross-scale error: {float(disagreement):.2f}%"
                if disagreement is not None
                else "Cross-scale error: unavailable"
            ),
            (
                "Length hierarchy: metric "
                f"{float(getattr(ruler, 'metric_hierarchy_consistency', 0.0)):.0%}; "
                "imperial "
                f"{float(getattr(ruler, 'imperial_hierarchy_consistency', 0.0)):.0%}"
            ),
        ]
        summary = self._scene.addText("\n".join(lines))
        summary_font = QFont(summary.font())
        summary_font.setPixelSize(max(13, min(30, round(float(ruler.width_px) * 0.095))))
        summary_font.setBold(True)
        summary.setFont(summary_font)
        summary.setDefaultTextColor(
            QColor("#ff5964")
            if (
                disagreement is not None and float(disagreement) > 3.0
            )
            or (
                bool(getattr(ruler, "metric_tick_segments", ()))
                and not bool(getattr(ruler, "metric_hierarchy_reliable", False))
            )
            or (
                bool(getattr(ruler, "imperial_tick_segments", ()))
                and not bool(getattr(ruler, "imperial_hierarchy_reliable", False))
            )
            else QColor("#ecf6ff")
        )
        outline = np.asarray(ruler.outline_points, dtype=np.float64)
        if outline.size:
            summary.setPos(float(np.min(outline[:, 0])), float(np.min(outline[:, 1])) - summary.boundingRect().height() - 8.0)
        summary_plate = self._scene.addRect(
            summary.sceneBoundingRect().adjusted(-7.0, -5.0, 7.0, 5.0),
            QPen(Qt.PenStyle.NoPen),
            QBrush(QColor(8, 12, 16, 225)),
        )
        summary_plate.setZValue(16.5)
        summary_plate.setOpacity(max(0.55, self._overlay_opacity))
        summary_plate.setToolTip(
            "Contrast plate for the ruler evidence scale and hierarchy summary."
        )
        self._overlay_items.append(summary_plate)
        summary.setZValue(17)
        summary.setOpacity(self._overlay_opacity)
        summary.setToolTip(
            "Metric and imperial pixels/mm are calculated independently from their "
            "own tick bars. Measured tick lengths must also put the longest class "
            "on the major increments before either family is accepted."
        )
        self._overlay_items.append(summary)

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
        self._layout_calculating_banner()
        self._layout_instance_continuity_warning()
        self._layout_context_panel()
        if self._context_panel is not None:
            self._context_panel.raise_()
        self.instance_continuity_warning_banner.raise_()
        self.calculating_banner.raise_()
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
        analysis_layers = getattr(self._analysis_result, "layers", None)
        # net_physical is the saved assisted-tool ID for the authoritative
        # Reference-edge probability. Conservative evidence has its own ID.
        probability_attribute = {
            "net_physical": "reference_edge_probability",
            "conservative_net_physical": "conservative_net_physical_edge_evidence",
        }.get(source)
        has_cached_edge_probability = bool(
            probability_attribute is not None
            and getattr(analysis_layers, probability_attribute, None) is not None
        )
        net_scale = (
            float(
                getattr(
                    analysis_layers,
                    "net_physical_edge_internal_scale",
                    0.50,
                )
            )
            if source == "conservative_net_physical" and not has_cached_edge_probability
            else None
        )
        cache_key = (
            f"edge_source:{source}:{net_scale:.6f}"
            if net_scale is not None
            else f"edge_source:{source}:cached"
            if has_cached_edge_probability
            else f"edge_source:{source}"
        )
        cached = self._annotation_evidence_cache.get(cache_key)
        if cached is not None:
            return cached

        attributes = {
            "magnitude": "edge_likelihood",
            "ridges": "edge_ridges",
            "reference_ridges": "reference_edge_ridges",
            "conservative_reference_ridges": "net_reference_edge_ridges",
            "normalized_reference_ridges": (
                "normalized_net_reference_edge_ridges"
            ),
            "traces": "edge_trace_labels",
            "normalized_net_physical": (
                "locally_normalized_net_physical_edge"
            ),
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
        elif probability_attribute is not None:
            if has_cached_edge_probability:
                result = display_u8(probability_attribute)
            else:
                # Compatibility for compact test/draft results made before the
                # Reference-edges node acquired an authoritative edge-supported
                # output. Even this fallback applies true-edge support; only
                # the explicitly conservative source subtracts Non-physical.
                physical = display_u8("physical_edge_probability").astype(
                    np.float32
                )
                compatibility = physical
                if source == "conservative_net_physical":
                    try:
                        non_physical = display_u8("non_edge_probability").astype(
                            np.float32
                        )
                    except RuntimeError:
                        non_physical = np.zeros_like(physical)
                    compatibility = np.clip(
                        physical - float(net_scale) * non_physical, 0.0, 255.0
                    )
                true_edge_support = (
                    display_u8("edge_ridges").astype(np.float32) / 255.0
                )
                result = np.rint(
                    compatibility * true_edge_support
                ).astype(np.uint8)
        elif source == "adaptive":
            precise = []
            for attribute, binary in (
                ("edge_ridges", False),
                ("normalized_net_reference_edge_ridges", False),
                ("edge_trace_labels", True),
            ):
                try:
                    precise.append(display_u8(attribute, binary=binary))
                except RuntimeError:
                    pass
            broad = []
            learned_attribute = "locally_normalized_net_physical_edge"
            try:
                broad.append(display_u8(learned_attribute))
            except RuntimeError:
                try:
                    broad.append(display_u8("reference_edge_probability"))
                except RuntimeError:
                    pass
            for attribute in ("edge_likelihood",):
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

    def _manual_seed_centre_hit(
        self, scene_point: QPointF
    ) -> dict[str, object] | None:
        """Return the closest constant-screen-size marker hit."""

        pointer = self.mapFromScene(scene_point)
        candidates: list[tuple[float, int, dict[str, object]]] = []
        priority = {"manual": 0, "rejected": 0, "automatic": 1, "annotation": 2}
        for record in self._procedural_marker_records(include_current_manual=True):
            marker = self.mapFromScene(record["point"])
            distance = float(np.hypot(pointer.x() - marker.x(), pointer.y() - marker.y()))
            if distance <= 12.0:
                candidates.append(
                    (distance, priority.get(str(record["source"]), 3), record)
                )
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1]))
        return dict(candidates[0][2])

    def _clamped_manual_seed_centre_point(self, scene_point: QPointF) -> QPointF:
        if self._image_item is None:
            return QPointF(scene_point)
        bounds = self._image_item.boundingRect()
        return QPointF(
            min(max(float(scene_point.x()), bounds.left()), bounds.right() - 1e-6),
            min(max(float(scene_point.y()), bounds.top()), bounds.bottom() - 1e-6),
        )

    def _start_manual_seed_centre_drag(
        self,
        scene_point: QPointF,
        record: dict[str, object] | None,
    ) -> None:
        point = self._clamped_manual_seed_centre_point(scene_point)
        if record is None:
            base = self._manual_seed_centres.copy()
            index = len(base)
            base = np.vstack((base, (point.x(), point.y())))
            mode = self._manual_seed_centre_mode
            source = "new"
            original = QPointF(point)
            result_index = None
        else:
            source = str(record["source"])
            if source == "annotation":
                self._manual_seed_centre_selected = dict(record)
                self.instance_tool_status.emit(
                    "That magenta centre is locked to an applied seed annotation; "
                    "move the annotation itself to adjust it."
                )
                self._render_analysis()
                return
            if source == "rejected":
                self.instance_tool_status.emit(
                    "Rejected manual centre: "
                    + str(record.get("reason") or "not accepted by procedural inference")
                    + ". Drag it to try another location, or remove it."
                )
            original = QPointF(record["point"])
            result_index = record.get("result_index")
            manual_index = record.get("manual_index")
            if source == "automatic":
                # A distant move would not suppress the old automatic maximum
                # in Augment mode. Convert the currently surviving editable
                # marker set into an explicit replacement before moving it.
                base = self.editable_procedural_marker_centres()
                if not len(base):
                    return
                distances = np.hypot(
                    base[:, 0] - original.x(), base[:, 1] - original.y()
                )
                index = int(np.argmin(distances))
                mode = "replace_automatic"
            elif manual_index is not None:
                base = self._manual_seed_centres.copy()
                index = int(manual_index)
                if not 0 <= index < len(base):
                    return
                mode = self._manual_seed_centre_mode
            else:
                return
        self._manual_seed_centre_selected = None if record is None else dict(record)
        self._manual_seed_centre_drag = {
            "base": base,
            "index": index,
            "mode": mode,
            "source": source,
            "result_index": result_index,
            "press_view": self.mapFromScene(scene_point),
            "point": point,
            "original": original,
        }
        self._show_manual_seed_centre_drag_point(point)

    def _show_manual_seed_centre_drag_point(self, point: QPointF) -> None:
        self._clear_manual_seed_centre_drag_items()
        pen = QPen(QColor("#ffffff"), 2.5)
        pen.setCosmetic(True)
        brush = QBrush(QColor("#ffe04f"))
        circle = self._scene.addEllipse(-8.0, -8.0, 16.0, 16.0, pen, brush)
        cross_pen = QPen(QColor("#493f00"), 1.5)
        cross_pen.setCosmetic(True)
        horizontal = self._scene.addLine(-4.0, 0.0, 4.0, 0.0, cross_pen)
        vertical = self._scene.addLine(0.0, -4.0, 0.0, 4.0, cross_pen)
        for item in (circle, horizontal, vertical):
            item.setPos(point)
            item.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True
            )
            item.setZValue(60)
            self._manual_seed_centre_drag_items.append(item)

    def _clear_manual_seed_centre_drag_items(self) -> None:
        for item in self._manual_seed_centre_drag_items:
            if item.scene() is self._scene:
                self._scene.removeItem(item)
        self._manual_seed_centre_drag_items.clear()

    def _finish_manual_seed_centre_drag(self, scene_point: QPointF) -> None:
        drag = self._manual_seed_centre_drag
        if drag is None:
            return
        point = self._clamped_manual_seed_centre_point(scene_point)
        press_view = drag["press_view"]
        release_view = self.mapFromScene(scene_point)
        moved = float(
            np.hypot(
                release_view.x() - press_view.x(),
                release_view.y() - press_view.y(),
            )
        )
        source = str(drag["source"])
        self._manual_seed_centre_drag = None
        self._clear_manual_seed_centre_drag_items()
        if source != "new" and moved < 2.0:
            self._render_analysis()
            return
        values = np.asarray(drag["base"], dtype=np.float64).copy()
        index = int(drag["index"])
        values[index] = (point.x(), point.y())
        mode = str(drag["mode"])
        if source == "automatic":
            label = "moved automatic centre and switched to Replace automatic"
        elif source == "new":
            label = "added manual seed centre"
        else:
            label = "moved manual seed centre"
        self._commit_manual_seed_centre_edit(values, mode, label)

    def _remove_manual_seed_centre_record(
        self, record: dict[str, object]
    ) -> None:
        source = str(record["source"])
        if source == "annotation":
            self.instance_tool_status.emit(
                "That magenta centre is locked to an applied seed annotation; "
                "edit the annotation to remove it."
            )
            return
        manual_index = record.get("manual_index")
        if source == "automatic":
            values = self.editable_procedural_marker_centres(
                exclude_result_index=int(record["result_index"])
            )
            self._commit_manual_seed_centre_edit(
                values,
                "replace_automatic",
                "removed automatic centre and switched to Replace automatic",
            )
            return
        if manual_index is None:
            return
        index = int(manual_index)
        if not 0 <= index < len(self._manual_seed_centres):
            return
        values = np.delete(self._manual_seed_centres, index, axis=0)
        self._commit_manual_seed_centre_edit(
            values,
            self._manual_seed_centre_mode,
            "removed manual seed centre",
        )

    def _commit_manual_seed_centre_edit(
        self, values: np.ndarray, mode: str, label: str
    ) -> None:
        self._manual_seed_centres = np.asarray(values, dtype=np.float64).reshape(-1, 2)
        self._manual_seed_centre_mode = str(mode)
        self._manual_seed_centre_selected = None
        self.manual_seed_centres_edited.emit(
            self._manual_seed_centres.copy(), self._manual_seed_centre_mode, label
        )
        self.instance_tool_status.emit(label.capitalize() + "; recomputing procedural instances.")
        self._render_analysis()

    def cancel_manual_seed_centre_drag(self) -> bool:
        if self._manual_seed_centre_drag is None:
            return False
        self._manual_seed_centre_drag = None
        self._clear_manual_seed_centre_drag_items()
        self.instance_tool_status.emit("Cancelled manual seed-centre adjustment.")
        self._render_analysis()
        return True

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._hilum_editing and self._image_item is not None:
            point = self.mapToScene(event.position().toPoint())
            if event.button() == Qt.MouseButton.LeftButton and self._image_item.boundingRect().contains(point):
                anchor = point
                if self._hilum_point is not None and self._hilum_direction is not None:
                    start = QPointF(*self._hilum_point)
                    tip = start + QPointF(*self._hilum_direction) * self._hilum_arrow_length()
                    if QLineF(tip, point).length() * self.transform().m11() < 14:
                        anchor = start
                self._hilum_drag = anchor
                self.set_hilum_landmark((anchor.x(), anchor.y()), None)
            event.accept()
            return
        if self._manual_seed_centre_editing and self._image_item is not None:
            scene_point = self.mapToScene(event.position().toPoint())
            if not self._image_item.boundingRect().contains(scene_point):
                event.ignore()
                return
            record = self._manual_seed_centre_hit(scene_point)
            if event.button() == Qt.MouseButton.RightButton:
                if record is not None:
                    self._manual_seed_centre_selected = dict(record)
                    self._remove_manual_seed_centre_record(record)
                event.accept()
                return
            if event.button() == Qt.MouseButton.LeftButton:
                self._start_manual_seed_centre_drag(scene_point, record)
                event.accept()
                return
        if (
            self._reference_point_mode is None
            and not self._manual_seed_centre_editing
            and self._overlay_mode == "procedural_instances"
            and event.button() == Qt.MouseButton.LeftButton
            and self._analysis_result is not None
            and self._image_item is not None
        ):
            scene_point = self.mapToScene(event.position().toPoint())
            procedural = getattr(
                self._analysis_result, "procedural_instances", None
            )
            if procedural is not None:
                offset_x, offset_y = self._analysis_result.crop_offset
                scale = float(procedural.working_scale)
                x = int(np.floor((scene_point.x() - offset_x) * scale))
                y = int(np.floor((scene_point.y() - offset_y) * scale))
                if 0 <= y < procedural.labels.shape[0] and 0 <= x < procedural.labels.shape[1]:
                    label = int(procedural.labels[y, x])
                    if label > 0:
                        self._selected_procedural_label = label
                        self.procedural_instance_selected.emit(label)
                        self._render_analysis()
                        event.accept()
                        return
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
        if self._hilum_editing:
            if self._hilum_drag is not None:
                point = self.mapToScene(event.position().toPoint())
                delta = point - self._hilum_drag
                length = np.hypot(delta.x(), delta.y())
                direction = (delta.x()/length, delta.y()/length) if length > 2 else None
                self.set_hilum_landmark((self._hilum_drag.x(), self._hilum_drag.y()), direction)
            event.accept()
            return
        if self._manual_seed_centre_editing and self._manual_seed_centre_drag is not None:
            scene_point = self._clamped_manual_seed_centre_point(
                self.mapToScene(event.position().toPoint())
            )
            self._manual_seed_centre_drag["point"] = scene_point
            self._show_manual_seed_centre_drag_point(scene_point)
            event.accept()
            return
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
        if self._hilum_editing:
            if self._hilum_drag is not None and event.button() == Qt.MouseButton.LeftButton:
                self.hilum_landmark_edited.emit(self._hilum_point, self._hilum_direction)
                self._hilum_drag = None
            event.accept()
            return
        if (
            self._manual_seed_centre_editing
            and self._manual_seed_centre_drag is not None
            and event.button() == Qt.MouseButton.LeftButton
        ):
            self._finish_manual_seed_centre_drag(
                self.mapToScene(event.position().toPoint())
            )
            event.accept()
            return
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

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self._hilum_editing and event.key() == Qt.Key.Key_Escape:
            self.set_hilum_editing(False)
            self.hilum_editing_cancelled.emit()
            event.accept()
            return
        if self._manual_seed_centre_editing:
            if event.key() == Qt.Key.Key_Escape:
                if not self.cancel_manual_seed_centre_drag():
                    self.manual_seed_centre_editing_cancelled.emit()
                event.accept()
                return
            if event.key() in {Qt.Key.Key_Delete, Qt.Key.Key_Backspace}:
                if self._manual_seed_centre_selected is not None:
                    self._remove_manual_seed_centre_record(
                        dict(self._manual_seed_centre_selected)
                    )
                event.accept()
                return
        super().keyPressEvent(event)

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
