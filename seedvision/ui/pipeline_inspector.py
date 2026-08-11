"""Property inspector for the currently selected pipeline node."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QSignalBlocker, QRectF, QSize, Slot, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPalette, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from seedvision.pipeline import PipelineNode


class BackgroundColourGamut(QWidget):
    """Compact hue/tint/shade HSV projection with fitted probabilities."""

    CONTOURS = (
        (0.25, QColor("#11181d")),
        (0.50, QColor("#41d9ff")),
        (0.75, QColor("#ffe05a")),
        (0.90, QColor("#ff63d8")),
    )

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._image: QImage | None = None
        self._probability: np.ndarray | None = None
        self._centres = np.empty((0, 3), np.float32)
        self._centre_points = np.empty((0, 2), np.float32)
        self._weights = np.empty((0,), np.float32)
        self._class_name = "background"
        self.setMinimumHeight(205)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._update_tooltip()

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt override
        return QSize(300, 215)

    def set_profile(
        self,
        profile,
        parameters: dict[str, object],
        *,
        class_name: str = "background",
    ) -> None:
        self._class_name = class_name
        self._update_tooltip()
        if profile is None:
            self._image = None
            self._probability = None
            self._centres = np.empty((0, 3), np.float32)
            self._centre_points = np.empty((0, 2), np.float32)
            self._weights = np.empty((0,), np.float32)
            self.update()
            return

        centres = np.asarray(
            profile.component_centres_lab or (profile.centre_lab,),
            dtype=np.float32,
        ).reshape(-1, 3)
        scales = np.asarray(
            profile.component_scales_lab or (profile.scale_lab,),
            dtype=np.float32,
        ).reshape(-1, 3)
        weights = np.asarray(
            profile.component_weights or (1.0,), dtype=np.float32
        ).reshape(-1)
        excluded_centres = np.asarray(
            profile.excluded_component_centres_lab,
            dtype=np.float32,
        ).reshape(-1, 3)
        excluded_scales = np.asarray(
            profile.excluded_component_scales_lab,
            dtype=np.float32,
        ).reshape(-1, 3)
        excluded_weights = np.asarray(
            profile.excluded_component_weights,
            dtype=np.float32,
        ).reshape(-1)
        if len(scales) != len(centres):
            scales = np.repeat(
                np.asarray(profile.scale_lab, np.float32)[None], len(centres), axis=0
            )
        if len(weights) != len(centres):
            weights = np.full(len(centres), 1.0 / max(1, len(centres)), np.float32)
        scales = np.maximum(scales, 0.25)
        scale_multiplier = max(
            0.05,
            float(
                parameters.get(
                    f"{class_name}_distribution_scale_multiplier", 1.0
                )
            ),
        )
        frequency_power = max(
            0.0,
            float(parameters.get(f"{class_name}_frequency_weight_power", 0.0)),
        )
        effective_scales = scales * scale_multiplier
        if len(excluded_scales) != len(excluded_centres):
            excluded_scales = np.repeat(
                np.asarray(profile.scale_lab, np.float32)[None],
                len(excluded_centres),
                axis=0,
            )
        if len(excluded_weights) != len(excluded_centres):
            excluded_weights = np.full(
                len(excluded_centres),
                1.0 / max(1, len(excluded_centres)),
                np.float32,
            )
        excluded_scales = np.maximum(excluded_scales, 0.25) * scale_multiplier
        excluded_adjusted_weights = np.maximum(excluded_weights, 1e-6)
        if len(excluded_adjusted_weights):
            excluded_adjusted_weights /= max(
                float(excluded_adjusted_weights.max()), 1e-6
            )

        width, height = 320, 180
        hsv = self._hsv_projection(width, height)

        import cv2

        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        adjusted_weights = np.maximum(weights, 1e-6) ** frequency_power
        adjusted_weights /= max(float(adjusted_weights.max()), 1e-6)
        # The attached-style hue/tint/shade square has one hidden colour
        # dimension. Project the fitted distribution by taking maximum
        # membership across saturation at each hue and tone, so neutral and
        # partially saturated learned colours are not omitted from the map.
        hue = hsv[:, :, 0]
        lightness = np.uint8(
            np.rint(np.linspace(255.0, 0.0, height, dtype=np.float32))
        )[:, None]
        probability = np.zeros((height, width), np.float32)
        for saturation in np.linspace(0.0, 255.0, 11, dtype=np.float32):
            hls = np.empty_like(hsv)
            hls[:, :, 0] = hue
            hls[:, :, 1] = lightness
            hls[:, :, 2] = np.uint8(np.rint(saturation))
            projected_rgb = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)
            projected_lab = cv2.cvtColor(
                projected_rgb, cv2.COLOR_RGB2LAB
            ).astype(np.float32)
            plane_probability = np.zeros((height, width), np.float32)
            for centre, scale, weight in zip(
                centres, effective_scales, adjusted_weights, strict=True
            ):
                delta = (projected_lab - centre) / scale
                distance = np.sum(
                    (delta * delta)
                    * np.asarray((1.0, 1.25, 1.25), np.float32),
                    axis=2,
                )
                plane_probability += np.exp(-0.5 * distance) * weight
            excluded_probability = np.zeros((height, width), np.float32)
            for centre, scale, weight in zip(
                excluded_centres,
                excluded_scales,
                excluded_adjusted_weights,
                strict=True,
            ):
                delta = (projected_lab - centre) / scale
                distance = np.sum(
                    (delta * delta)
                    * np.asarray((1.0, 1.25, 1.25), np.float32),
                    axis=2,
                )
                excluded_probability += np.exp(-0.5 * distance) * weight
            plane_probability *= 1.0 - float(profile.exclusion_strength) * np.clip(
                excluded_probability, 0.0, 1.0
            )
            probability = np.maximum(
                probability, np.clip(plane_probability, 0.0, 1.0)
            )

        # Preserve the actual projected colour everywhere. Membership is
        # communicated exclusively by the high-contrast contour lines; dimming
        # unselected regions made the represented colours impossible to judge.
        display = rgb.copy()
        for level, colour in self.CONTOURS:
            edge = self._contour_edge(probability, level)
            display[edge] = (colour.red(), colour.green(), colour.blue())
        display = np.ascontiguousarray(display)
        self._image = QImage(
            display.data,
            width,
            height,
            display.strides[0],
            QImage.Format.Format_RGB888,
        ).copy()
        self._probability = probability
        self._centres = centres
        self._centre_points = self._project_centres(
            centres, width, height
        )
        self._weights = weights
        self.update()

    def _update_tooltip(self) -> None:
        self.setToolTip(
            "An HSV hue/tint/shade projection. Pixel colour is the represented "
            f"colour; contour lines show fitted {self._class_name} membership "
            "probability projected across the unshown saturation dimension in "
            "the model's Lab space. "
            "Circle labels show learned colour-mode occurrence in "
            "the references or automatic high-confidence samples."
        )

    @staticmethod
    def _hsv_projection(width: int, height: int) -> np.ndarray:
        """Return hue across X and white-to-pure-to-black colour down Y."""

        hue = np.linspace(0.0, 179.0, width, dtype=np.float32)
        row = np.linspace(0.0, 1.0, height, dtype=np.float32)
        saturation = np.where(row <= 0.5, row * 2.0, 1.0) * 255.0
        value = np.where(row <= 0.5, 1.0, (1.0 - row) * 2.0) * 255.0
        hsv = np.empty((height, width, 3), np.uint8)
        hsv[:, :, 0] = np.uint8(np.rint(hue))[None, :]
        hsv[:, :, 1] = np.uint8(np.rint(saturation))[:, None]
        hsv[:, :, 2] = np.uint8(np.rint(value))[:, None]
        return hsv

    @staticmethod
    def _project_centres(
        centres: np.ndarray, width: int, height: int
    ) -> np.ndarray:
        """Place learned Lab modes by their hue and HLS-equivalent tone."""

        import cv2

        lab = np.clip(np.rint(centres), 0, 255).astype(np.uint8)[:, None, :]
        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        hls = cv2.cvtColor(rgb, cv2.COLOR_RGB2HLS)[:, 0].astype(np.float32)
        return np.column_stack(
            (
                hls[:, 0] / 179.0 * max(1, width - 1),
                (1.0 - hls[:, 1] / 255.0) * max(1, height - 1),
            )
        ).astype(np.float32)

    @staticmethod
    def _contour_edge(probability: np.ndarray, level: float) -> np.ndarray:
        mask = probability >= level
        edge = np.zeros_like(mask)
        difference = mask[1:, :] != mask[:-1, :]
        edge[1:, :] |= difference
        edge[:-1, :] |= difference
        difference = mask[:, 1:] != mask[:, :-1]
        edge[:, 1:] |= difference
        edge[:, :-1] |= difference
        return edge

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        text_colour = self.palette().color(QPalette.ColorRole.Text)
        plot = QRectF(28.0, 6.0, max(40.0, self.width() - 36.0), 166.0)
        if self._image is None:
            painter.setPen(text_colour)
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                f"Run the {self._class_name}-colour node\n"
                "to display its probability gamut.",
            )
            return
        painter.drawImage(plot, self._image)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(text_colour, 1.0))
        painter.drawRect(plot)
        painter.drawText(
            QRectF(0.0, plot.top(), 24.0, plot.height()),
            Qt.AlignmentFlag.AlignCenter,
            "tone",
        )
        painter.drawText(
            QRectF(plot.left(), plot.bottom(), plot.width(), 18.0),
            Qt.AlignmentFlag.AlignCenter,
            "HSV hue",
        )

        image_width = max(1, self._image.width() - 1)
        image_height = max(1, self._image.height() - 1)
        for point, weight in zip(
            self._centre_points, self._weights, strict=False
        ):
            x = plot.left() + float(point[0]) / image_width * plot.width()
            y = plot.top() + float(point[1]) / image_height * plot.height()
            painter.setPen(QPen(QColor("#101418"), 3.0))
            painter.setBrush(QColor("#ffffff"))
            painter.drawEllipse(QRectF(x - 4.0, y - 4.0, 8.0, 8.0))
            painter.setPen(QColor("#ffffff"))
            painter.drawText(QRectF(x + 6.0, y - 9.0, 48.0, 18.0), f"{weight:.0%}")

        legend_y = self.height() - 14.0
        painter.setPen(text_colour)
        painter.drawText(
            QRectF(plot.left(), legend_y - 10.0, 82.0, 18.0),
            f"P({self._class_name})",
        )
        x = plot.left() + 82.0
        for level, colour in self.CONTOURS:
            painter.setPen(QPen(colour, 3.0))
            painter.drawLine(round(x), round(legend_y), round(x + 13.0), round(legend_y))
            painter.setPen(text_colour)
            painter.drawText(QRectF(x + 16.0, legend_y - 10.0, 36.0, 18.0), f"{level:.0%}")
            x += 42.0


class PipelineInspector(QWidget):
    """Edit node enablement and typed parameter values."""

    parameter_changed = Signal(str, str, object)
    enabled_changed = Signal(str, bool)
    parameters_reset = Signal(str)
    overlay_selected = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._node: PipelineNode | None = None
        self._analysis_result = None
        self._parameter_widgets: list[QWidget] = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.title_label = QLabel("Select a pipeline node", self)
        self.title_label.setStyleSheet("font-weight: 650; font-size: 14px;")
        self.overlay_combo = QComboBox(self)
        self.overlay_combo.setObjectName("nodeOverlaySelector")
        self.overlay_combo.setToolTip(
            "Choose among the image overlays produced by this pipeline node."
        )
        self.overlay_combo.setVisible(False)
        self.overlay_combo.currentIndexChanged.connect(self._overlay_changed)
        self.description_label = QLabel(
            "Node controls and parameters will appear here.", self
        )
        self.description_label.setWordWrap(True)
        self._secondary_colour = _adaptive_text_colour(self, emphasized=False)
        detail_colour = _adaptive_text_colour(self, emphasized=True)
        self.description_label.setStyleSheet(f"color: {self._secondary_colour};")
        self.method_heading = QToolButton(self)
        self.method_heading.setText("How it works")
        self.method_heading.setCheckable(True)
        self.method_heading.setChecked(False)
        self.method_heading.setArrowType(Qt.ArrowType.RightArrow)
        self.method_heading.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.method_heading.setStyleSheet(
            "QToolButton { font-weight: 600; margin-top: 6px; border: 0; "
            "padding: 3px 1px; text-align: left; }"
            "QToolButton:hover { text-decoration: underline; }"
        )
        self.method_heading.setToolTip("Expand the node's method explanation.")
        self.method_heading.toggled.connect(self._method_expanded_changed)
        self.method_heading.setVisible(False)
        self.details_label = QLabel("", self)
        self.details_label.setWordWrap(True)
        self.details_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.details_label.setStyleSheet(f"color: {detail_colour};")
        self.details_label.setVisible(False)
        self.enabled_checkbox = QCheckBox("Enabled", self)
        self.enabled_checkbox.setVisible(False)
        self.enabled_checkbox.toggled.connect(self._enabled_toggled)
        self.status_label = QLabel("", self)
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(f"color: {self._secondary_colour};")
        self.gamut_heading = QLabel("Accepted background colours", self)
        self.gamut_heading.setStyleSheet("font-weight: 600; margin-top: 6px;")
        self.gamut_heading.setVisible(False)
        self.gamut_widget = BackgroundColourGamut(self)
        self.gamut_widget.setVisible(False)
        self.gamut_caption = QLabel("", self)
        self.gamut_caption.setWordWrap(True)
        self.gamut_caption.setStyleSheet(f"color: {self._secondary_colour};")
        self.gamut_caption.setVisible(False)
        self.parameter_container = QWidget(self)
        self.parameter_form = QFormLayout(self.parameter_container)
        self.parameter_form.setContentsMargins(0, 0, 0, 0)
        self.parameters_header = QWidget(self)
        parameters_header_layout = QHBoxLayout(self.parameters_header)
        parameters_header_layout.setContentsMargins(0, 0, 0, 0)
        self.parameters_heading = QLabel("Settings", self)
        self.parameters_heading.setStyleSheet("font-weight: 600; margin-top: 6px;")
        self.reset_parameters_button = QPushButton("Reset", self.parameters_header)
        self.reset_parameters_button.setToolTip(
            "Restore every setting on this node to its authored default."
        )
        self.reset_parameters_button.setMaximumWidth(72)
        self.reset_parameters_button.clicked.connect(self._reset_parameters)
        parameters_header_layout.addWidget(self.parameters_heading)
        parameters_header_layout.addStretch(1)
        parameters_header_layout.addWidget(self.reset_parameters_button)
        self.parameters_header.setVisible(False)
        layout.addWidget(self.title_label)
        layout.addWidget(self.overlay_combo)
        layout.addWidget(self.description_label)
        layout.addWidget(self.method_heading)
        layout.addWidget(self.details_label)
        layout.addWidget(self.enabled_checkbox)
        layout.addWidget(self.status_label)
        layout.addWidget(self.gamut_heading)
        layout.addWidget(self.gamut_widget)
        layout.addWidget(self.gamut_caption)
        layout.addWidget(self.parameters_header)
        layout.addWidget(self.parameter_container)

    def set_node(self, node: PipelineNode) -> None:
        self._node = node
        self.title_label.setText(f"Node: {node.title}")
        self.description_label.setText(node.description)
        self.details_label.setText(node.details)
        self.method_heading.blockSignals(True)
        self.method_heading.setChecked(False)
        self.method_heading.blockSignals(False)
        self.method_heading.setArrowType(Qt.ArrowType.RightArrow)
        self.method_heading.setToolTip("Expand the node's method explanation.")
        self.details_label.setVisible(False)
        self.method_heading.setVisible(bool(node.details))
        self.status_label.setText(
            f"Status: {node.status.value}. {node.status_detail}"
        )
        self.enabled_checkbox.blockSignals(True)
        self.enabled_checkbox.setChecked(node.enabled)
        self.enabled_checkbox.setEnabled(node.implemented and node.bypassable)
        self.enabled_checkbox.setVisible(True)
        self.enabled_checkbox.blockSignals(False)
        self._clear_parameters()
        for spec in node.parameter_specs:
            value = node.parameters[spec.key]
            if spec.kind == "float":
                editor = QDoubleSpinBox(self.parameter_container)
                editor.setDecimals(3)
                editor.setRange(float(spec.minimum), float(spec.maximum))
                editor.setSingleStep(float(spec.step or 0.01))
                editor.setValue(float(value))
                editor.editingFinished.connect(
                    lambda key=spec.key, widget=editor: self._parameter_edited(
                        key, widget.value()
                    )
                )
            elif spec.kind == "int":
                editor = QSpinBox(self.parameter_container)
                editor.setRange(int(spec.minimum), int(spec.maximum))
                editor.setSingleStep(int(spec.step or 1))
                editor.setValue(int(value))
                editor.editingFinished.connect(
                    lambda key=spec.key, widget=editor: self._parameter_edited(
                        key, widget.value()
                    )
                )
            elif spec.kind == "choice":
                editor = QComboBox(self.parameter_container)
                editor.addItems(spec.choices)
                editor.setCurrentText(str(value))
                editor.currentTextChanged.connect(
                    lambda selected, key=spec.key: self._parameter_edited(key, selected)
                )
            elif spec.kind == "text":
                editor = QLineEdit(self.parameter_container)
                editor.setText(str(value))
                editor.editingFinished.connect(
                    lambda key=spec.key, widget=editor: self._parameter_edited(
                        key, widget.text()
                    )
                )
            else:
                editor = QCheckBox(self.parameter_container)
                editor.setChecked(bool(value))
                editor.toggled.connect(
                    lambda checked, key=spec.key: self._parameter_edited(key, checked)
                )
            editor.setEnabled(node.enabled and node.implemented)
            editor.setToolTip(spec.description)
            field_label = QLabel(spec.label, self.parameter_container)
            field_label.setToolTip(spec.description)
            self.parameter_form.addRow(field_label, editor)
            self._parameter_widgets.append(editor)
        self.parameter_container.setVisible(bool(node.parameter_specs))
        self.parameters_header.setVisible(bool(node.parameter_specs))
        self._update_background_gamut()

    def set_overlay_options(
        self,
        options: tuple[tuple[str, str], ...],
        current_mode: str,
    ) -> None:
        """Show only the overlays owned by the selected pipeline node."""

        with QSignalBlocker(self.overlay_combo):
            self.overlay_combo.clear()
            for label, mode in options:
                self.overlay_combo.addItem(label, mode)
            if options:
                index = self.overlay_combo.findData(current_mode)
                self.overlay_combo.setCurrentIndex(max(0, index))
                self.overlay_combo.setEnabled(True)
                self.overlay_combo.setToolTip(
                    "Choose among the image overlays produced by this pipeline node."
                )
            else:
                self.overlay_combo.addItem("No image overlays", "")
                self.overlay_combo.setCurrentIndex(0)
                self.overlay_combo.setEnabled(False)
                self.overlay_combo.setToolTip(
                    "This pipeline node does not produce a viewable image overlay."
                )
        self.overlay_combo.setVisible(self._node is not None)

    @Slot(int)
    def _overlay_changed(self, index: int) -> None:
        del index
        mode = str(self.overlay_combo.currentData() or "")
        if mode:
            self.overlay_selected.emit(mode)

    def _method_expanded_changed(self, expanded: bool) -> None:
        has_details = self._node is not None and bool(self._node.details)
        self.details_label.setVisible(bool(expanded and has_details))
        self.method_heading.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self.method_heading.setToolTip(
            "Collapse the node's method explanation."
            if expanded
            else "Expand the node's method explanation."
        )

    def set_analysis_result(self, result) -> None:
        """Supply the visible image result for node-specific diagnostics."""

        self._analysis_result = result
        self._update_background_gamut()

    def refresh_status(self) -> None:
        if self._node is not None:
            self.status_label.setText(
                f"Status: {self._node.status.value}. {self._node.status_detail}"
            )

    def _clear_parameters(self) -> None:
        while self.parameter_form.rowCount():
            self.parameter_form.removeRow(0)
        self._parameter_widgets.clear()

    def _update_background_gamut(self) -> None:
        node_id = None if self._node is None else self._node.identifier
        visible = node_id in {"background_likelihood", "foreground_segmentation"}
        self.gamut_heading.setVisible(visible)
        self.gamut_widget.setVisible(visible)
        self.gamut_caption.setVisible(visible)
        if not visible:
            return
        class_name = (
            "foreground" if node_id == "foreground_segmentation" else "background"
        )
        self.gamut_heading.setText(f"Accepted {class_name} colours")
        layers = getattr(self._analysis_result, "layers", None)
        profile = getattr(layers, f"{class_name}_colour_profile", None)
        self.gamut_widget.set_profile(
            profile,
            self._node.parameters,
            class_name=class_name,
        )
        if profile is None:
            self.gamut_caption.setText(
                "No fitted distribution is available for the current image."
            )
            return
        low = "/".join(str(value) for value in profile.bgr_low)
        high = "/".join(str(value) for value in profile.bgr_high)
        reference_count = (
            getattr(self._analysis_result, "foreground_reference_count", 0)
            if class_name == "foreground"
            else getattr(layers, "background_reference_count", 0)
        )
        source_description = (
            "painted-reference modes"
            if reference_count
            else "automatic high-confidence modes"
        )
        exclusion_modes = len(
            getattr(profile, "excluded_component_centres_lab", ())
        )
        exclusion_description = (
            f" {exclusion_modes} negative-evidence mode(s) from painted exclusions "
            "downweight matching colours without overriding painted coordinates."
            if exclusion_modes
            else ""
        )
        self.gamut_caption.setText(
            f"Colour beneath the contours is an HSV hue/tint/shade projection "
            f"(white through pure colour to black). Contours project the fitted "
            f"Lab model across saturation and show 25/50/75/90% membership; "
            f"circles are learned "
            f"{source_description} labelled by frequency. Observed 5–95% BGR: "
            f"{low}–{high}.{exclusion_description}"
        )

    def _enabled_toggled(self, enabled: bool) -> None:
        if self._node is not None:
            self.enabled_changed.emit(self._node.identifier, enabled)

    def _parameter_edited(self, key: str, value) -> None:
        if self._node is not None:
            self.parameter_changed.emit(self._node.identifier, key, value)

    def _reset_parameters(self) -> None:
        if self._node is not None:
            self.parameters_reset.emit(self._node.identifier)


def _adaptive_text_colour(widget: QWidget, *, emphasized: bool) -> str:
    """Choose readable explanatory text for the active light or dark palette."""

    background = widget.palette().color(QPalette.ColorRole.Window)
    is_dark = background.lightnessF() < 0.50
    if emphasized:
        return "#d2d9e1" if is_dark else "#303b46"
    return "#bcc6d1" if is_dark else "#4a5865"
