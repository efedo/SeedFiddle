"""Property inspector for the currently selected pipeline node."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QSignalBlocker, QRectF, QSize, Slot, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPalette, QPen
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
    """Render fitted colour probabilities in HSV coordinates."""

    NEUTRAL_COLUMNS = 24
    CONTOURS = (
        (0.25, QColor("#11181d")),
        (0.50, QColor("#41d9ff")),
        (0.75, QColor("#ffe05a")),
        (0.90, QColor("#ff63d8")),
    )

    @classmethod
    def dominant_hsv_value(cls, profile) -> float | None:
        """Return the Value coordinate of the strongest visible fitted mode."""

        if profile is None:
            return None
        import cv2

        centres = np.asarray(
            profile.component_centres_lab or (profile.centre_lab,),
            dtype=np.float32,
        ).reshape(-1, 3)
        if not len(centres):
            return None
        scales = np.asarray(
            profile.component_scales_lab or (profile.scale_lab,),
            dtype=np.float32,
        ).reshape(-1, 3)
        if len(scales) != len(centres):
            scales = np.repeat(
                np.asarray(profile.scale_lab, np.float32)[None],
                len(centres),
                axis=0,
            )
        weights = np.asarray(
            profile.component_weights or (1.0,), dtype=np.float32
        ).reshape(-1)
        if len(weights) != len(centres):
            weights = np.full(
                len(centres), 1.0 / max(1, len(centres)), np.float32
            )
        chromatic = cls._chromatic_components(
            centres, np.maximum(scales, 0.25)
        )
        candidates = np.flatnonzero(chromatic)
        if not len(candidates):
            candidates = np.arange(len(centres))
        index = int(candidates[np.argmax(weights[candidates])])
        centre_lab = np.clip(
            np.rint(centres[index]), 0, 255
        ).astype(np.uint8)[None, None, :]
        centre_rgb = cv2.cvtColor(centre_lab, cv2.COLOR_LAB2RGB)[0, 0]
        return float(centre_rgb.max()) / 255.0

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
        projected_lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        chroma_weight = max(
            0.0,
            float(
                parameters.get(
                    f"{class_name}_chroma_weight",
                    1.25,
                )
            ),
        )
        distance_weights = np.asarray(
            (1.0, chroma_weight, chroma_weight), np.float32
        )
        positive_combination = (
            "maximum" if class_name == "background" else "sum"
        )
        probability = np.zeros((height, width), np.float32)
        excluded_probability = np.zeros((height, width), np.float32)

        # Achromatic colours have no meaningful hue. Evaluate them only in the
        # dedicated neutral strip; otherwise saturation=0 makes the same grey
        # match every hue and produces a misleading horizontal probability band.
        neutral_width = min(self.NEUTRAL_COLUMNS, max(1, width // 4))
        probability[:, :neutral_width] = self._mixture_probability(
            projected_lab[:, :neutral_width],
            centres,
            effective_scales,
            adjusted_weights,
            distance_weights,
            combine_modes=positive_combination,
        )
        excluded_probability[:, :neutral_width] = self._mixture_probability(
            projected_lab[:, :neutral_width],
            excluded_centres,
            excluded_scales,
            excluded_adjusted_weights,
            distance_weights,
        )

        # A hue/tone map necessarily hides saturation. Evaluate each chromatic
        # fitted mode at its own measured saturation. Maximizing the complete
        # mixture over every possible saturation allowed a nearly neutral dark
        # mode to match the low-saturation candidate at every hue, producing a
        # false horizontal band across the entire picker.
        chromatic = self._chromatic_components(centres, effective_scales)
        excluded_chromatic = self._chromatic_components(
            excluded_centres, excluded_scales
        )
        hue_width = width - neutral_width
        hue_probability = self._hue_tone_probability(
            centres[chromatic],
            effective_scales[chromatic],
            adjusted_weights[chromatic],
            distance_weights,
            hue_width,
            height,
            combine_modes=positive_combination,
        )
        hue_excluded_probability = self._hue_tone_probability(
            excluded_centres[excluded_chromatic],
            excluded_scales[excluded_chromatic],
            excluded_adjusted_weights[excluded_chromatic],
            distance_weights,
            hue_width,
            height,
        )
        probability[:, neutral_width:] = hue_probability
        excluded_probability[:, neutral_width:] = hue_excluded_probability
        probability = self._apply_contrastive_negative_evidence(
            np.clip(probability, 0.0, 1.0),
            np.clip(excluded_probability, 0.0, 1.0),
            float(profile.exclusion_strength),
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
            centres, effective_scales, width, height
        )
        self._weights = weights
        self.update()

    @classmethod
    def render_hsv_value_slice(
        cls,
        profile,
        parameters: dict[str, object],
        *,
        class_name: str,
        value: float,
        width: int = 960,
        height: int = 640,
    ) -> tuple[QImage, np.ndarray, np.ndarray]:
        """Render a full-size hue/saturation slice at one HSV value.

        HSV is three-dimensional. A value slider is more honest than the old
        compact hue/tint/shade projection because every displayed pixel now
        represents one exact RGB colour. Achromatic fitted modes are evaluated
        in a dedicated neutral swatch so undefined hue cannot create a false
        horizontal contour across every hue.
        """

        import cv2

        value = float(np.clip(value, 0.0, 1.0))
        width = max(640, int(width))
        height = max(440, int(height))
        plot_left, plot_top = 100, 82
        plot_width = width - plot_left - 34
        plot_height = height - plot_top - 112

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
            profile.excluded_component_centres_lab, dtype=np.float32
        ).reshape(-1, 3)
        excluded_scales = np.asarray(
            profile.excluded_component_scales_lab, dtype=np.float32
        ).reshape(-1, 3)
        excluded_weights = np.asarray(
            profile.excluded_component_weights, dtype=np.float32
        ).reshape(-1)
        if len(scales) != len(centres):
            scales = np.repeat(
                np.asarray(profile.scale_lab, np.float32)[None],
                len(centres),
                axis=0,
            )
        if len(weights) != len(centres):
            weights = np.full(
                len(centres), 1.0 / max(1, len(centres)), np.float32
            )
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
        scales = np.maximum(scales, 0.25) * scale_multiplier
        excluded_scales = np.maximum(excluded_scales, 0.25) * scale_multiplier
        adjusted_weights = np.maximum(weights, 1e-6) ** frequency_power
        adjusted_weights /= max(float(adjusted_weights.max()), 1e-6)
        excluded_adjusted_weights = np.maximum(excluded_weights, 1e-6)
        if len(excluded_adjusted_weights):
            excluded_adjusted_weights /= max(
                float(excluded_adjusted_weights.max()), 1e-6
            )
        chroma_weight = max(
            0.0,
            float(parameters.get(f"{class_name}_chroma_weight", 1.25)),
        )
        distance_weights = np.asarray(
            (1.0, chroma_weight, chroma_weight), np.float32
        )
        positive_combination = (
            "maximum" if class_name == "background" else "sum"
        )

        hsv = np.empty((plot_height, plot_width, 3), np.uint8)
        hsv[:, :, 0] = np.uint8(
            np.rint(np.linspace(0.0, 179.0, plot_width, dtype=np.float32))
        )[None, :]
        hsv[:, :, 1] = np.uint8(
            np.rint(np.linspace(0.0, 255.0, plot_height, dtype=np.float32))
        )[:, None]
        hsv[:, :, 2] = np.uint8(round(value * 255.0))
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        projected_lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        chromatic = cls._chromatic_components(centres, scales)
        excluded_chromatic = cls._chromatic_components(
            excluded_centres, excluded_scales
        )
        probability = cls._mixture_probability(
            projected_lab,
            centres[chromatic],
            scales[chromatic],
            adjusted_weights[chromatic],
            distance_weights,
            combine_modes=positive_combination,
        )
        excluded_probability = cls._mixture_probability(
            projected_lab,
            excluded_centres[excluded_chromatic],
            excluded_scales[excluded_chromatic],
            excluded_adjusted_weights[excluded_chromatic],
            distance_weights,
        )
        probability = cls._apply_contrastive_negative_evidence(
            probability,
            excluded_probability,
            float(profile.exclusion_strength),
        )
        # At zero saturation hue is undefined. Keep the exact neutral result in
        # the swatch below instead of repeating it across all hue columns.
        probability[0] = 0.0

        display = rgb.copy()
        for level, colour in cls.CONTOURS:
            edge = cls._contour_edge(probability, level)
            display[edge] = (colour.red(), colour.green(), colour.blue())
        display = np.ascontiguousarray(display)
        plot_image = QImage(
            display.data,
            plot_width,
            plot_height,
            int(display.strides[0]),
            QImage.Format.Format_RGB888,
        ).copy()

        canvas = QImage(width, height, QImage.Format.Format_RGB32)
        canvas.fill(QColor("#20262d"))
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QColor("#f2f6fa"))
        base_font = QFont("Segoe UI")
        base_font.setPointSize(10)
        title_font = QFont(base_font)
        title_font.setPointSize(17)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.drawText(
            QRectF(24, 15, width - 48, 30),
            Qt.AlignmentFlag.AlignCenter,
            f"Accepted {class_name} colours — HSV value {value:.0%}",
        )
        painter.setFont(base_font)
        painter.setPen(QColor("#c7d0d9"))
        painter.drawText(
            QRectF(24, 48, width - 48, 24),
            Qt.AlignmentFlag.AlignCenter,
            "Every point is one exact colour; mode markers appear near their native brightness.",
        )
        plot_rect = QRectF(plot_left, plot_top, plot_width, plot_height)
        painter.drawImage(plot_rect, plot_image)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor("#e4e9ee"), 1.5))
        painter.drawRect(plot_rect)

        axis_font = QFont(painter.font())
        axis_font.setPointSize(10)
        painter.setFont(axis_font)
        for hue_degrees in range(0, 361, 60):
            x = plot_left + min(hue_degrees, 359) / 359.0 * plot_width
            painter.drawText(
                QRectF(x - 25, plot_rect.bottom() + 6, 50, 20),
                Qt.AlignmentFlag.AlignCenter,
                f"{hue_degrees}°",
            )
        painter.drawText(
            QRectF(plot_left, plot_rect.bottom() + 28, plot_width, 22),
            Qt.AlignmentFlag.AlignCenter,
            "Hue",
        )
        for saturation in (0, 25, 50, 75, 100):
            y = plot_top + saturation / 100.0 * plot_height
            painter.drawText(
                QRectF(38, y - 10, 52, 20),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                f"{saturation}%",
            )
        painter.save()
        painter.translate(18, plot_top + plot_height / 2)
        painter.rotate(-90)
        painter.drawText(
            QRectF(-plot_height / 2, -10, plot_height, 22),
            Qt.AlignmentFlag.AlignCenter,
            "Saturation",
        )
        painter.restore()

        centre_points = np.empty((0, 2), np.float32)
        if len(centres):
            centre_lab = np.clip(np.rint(centres), 0, 255).astype(np.uint8)[
                :, None, :
            ]
            centre_rgb = cv2.cvtColor(centre_lab, cv2.COLOR_LAB2RGB)[:, 0]
            centre_hsv = cv2.cvtColor(
                centre_rgb[:, None, :], cv2.COLOR_RGB2HSV
            )[:, 0].astype(np.float32)
            centre_points = np.column_stack(
                (
                    centre_hsv[:, 0] / 179.0 * max(1, plot_width - 1),
                    centre_hsv[:, 1] / 255.0 * max(1, plot_height - 1),
                )
            ).astype(np.float32)
            native_values = centre_hsv[:, 2] / 255.0
            displayed_modes = [
                index
                for index in np.argsort(weights)[::-1]
                if chromatic[index]
                and abs(float(native_values[index]) - value) <= 0.05
                and weights[index]
                >= max(0.005, float(weights.max()) * 0.03)
            ][:8]
            for index in displayed_modes:
                x = plot_left + float(centre_points[index, 0])
                y = plot_top + float(centre_points[index, 1])
                painter.setPen(QPen(QColor("#101418"), 4.0))
                painter.setBrush(QColor("#ffffff"))
                painter.drawEllipse(QRectF(x - 5, y - 5, 10, 10))
                label = f"{weights[index]:.0%} · V {native_values[index]:.0%}"
                label_bounds = painter.fontMetrics().boundingRect(label)
                label_width = label_bounds.width() + 10
                label_height = label_bounds.height() + 4
                label_x = min(x + 8, plot_rect.right() - label_width - 3)
                label_y = max(
                    plot_rect.top() + 3,
                    min(
                        y - label_height / 2,
                        plot_rect.bottom() - label_height - 3,
                    ),
                )
                label_rect = QRectF(
                    label_x,
                    label_y,
                    label_width,
                    label_height,
                )
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(22, 27, 32, 215))
                painter.drawRoundedRect(label_rect, 3, 3)
                painter.setPen(QColor("#ffffff"))
                painter.drawText(
                    label_rect.adjusted(5, 0, -5, 0),
                    Qt.AlignmentFlag.AlignVCenter,
                    label,
                )

        neutral_rgb = np.full((1, 1, 3), round(value * 255.0), np.uint8)
        neutral_lab = cv2.cvtColor(neutral_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        neutral_probability = cls._mixture_probability(
            neutral_lab,
            centres,
            scales,
            adjusted_weights,
            distance_weights,
            combine_modes=positive_combination,
        )
        neutral_excluded = cls._mixture_probability(
            neutral_lab,
            excluded_centres,
            excluded_scales,
            excluded_adjusted_weights,
            distance_weights,
        )
        neutral_probability = cls._apply_contrastive_negative_evidence(
            neutral_probability,
            neutral_excluded,
            float(profile.exclusion_strength),
        )
        legend_y = height - 43
        neutral_colour = QColor.fromRgb(
            int(neutral_rgb[0, 0, 0]),
            int(neutral_rgb[0, 0, 1]),
            int(neutral_rgb[0, 0, 2]),
        )
        painter.setBrush(neutral_colour)
        painter.setPen(QPen(QColor("#e4e9ee"), 1.5))
        painter.drawRect(QRectF(26, legend_y - 3, 28, 28))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawText(
            QRectF(62, legend_y - 3, 190, 28),
            Qt.AlignmentFlag.AlignVCenter,
            f"Neutral P({class_name}) {float(neutral_probability[0, 0]):.0%}",
        )
        x = 275.0
        painter.drawText(
            QRectF(x, legend_y - 3, 150, 28),
            Qt.AlignmentFlag.AlignVCenter,
            f"P({class_name}) contours:",
        )
        x += 148.0
        for level, colour in cls.CONTOURS:
            painter.setPen(QPen(colour, 4.0))
            painter.drawLine(round(x), round(legend_y + 11), round(x + 22), round(legend_y + 11))
            painter.setPen(QColor("#f2f6fa"))
            painter.drawText(
                QRectF(x + 27, legend_y - 3, 48, 28),
                Qt.AlignmentFlag.AlignVCenter,
                f"{level:.0%}",
            )
            x += 76.0
        painter.end()
        return canvas, probability, centre_points

    @staticmethod
    def _mixture_probability(
        lab: np.ndarray,
        centres: np.ndarray,
        scales: np.ndarray,
        weights: np.ndarray,
        distance_weights: np.ndarray,
        *,
        combine_modes: str = "sum",
    ) -> np.ndarray:
        probability = np.zeros(lab.shape[:2], np.float32)
        for centre, scale, weight in zip(
            centres, scales, weights, strict=True
        ):
            delta = (lab - centre) / scale
            distance = np.sum((delta * delta) * distance_weights, axis=2)
            membership = np.exp(-0.5 * distance) * weight
            if combine_modes == "maximum":
                probability = np.maximum(probability, membership)
            else:
                probability += membership
        return np.clip(probability, 0.0, 1.0)

    @staticmethod
    def _apply_contrastive_negative_evidence(
        positive: np.ndarray,
        negative: np.ndarray,
        strength: float,
    ) -> np.ndarray:
        """Mirror the CUDA class-relative Other-evidence calculation."""

        contradiction = np.maximum(negative - positive, 0.0) / (
            negative + positive + 1e-6
        )
        return positive * (
            1.0 - np.clip(float(strength), 0.0, 1.0) * contradiction
        )

    @classmethod
    def _hue_tone_probability(
        cls,
        centres: np.ndarray,
        scales: np.ndarray,
        weights: np.ndarray,
        distance_weights: np.ndarray,
        width: int,
        height: int,
        *,
        combine_modes: str = "sum",
    ) -> np.ndarray:
        """Project modes at measured saturation onto hue and tone.

        Each component has its own colour plane. The caller chooses the same
        sum or strongest-mode rule used by the corresponding raster estimator,
        while this projection avoids maximizing over unrelated, nearly neutral
        saturation slices.
        """

        import cv2

        probability = np.zeros((height, width), np.float32)
        if not len(centres):
            return probability
        centre_lab = np.clip(np.rint(centres), 0, 255).astype(np.uint8)[:, None, :]
        centre_hls = cv2.cvtColor(
            cv2.cvtColor(centre_lab, cv2.COLOR_LAB2RGB),
            cv2.COLOR_RGB2HLS,
        )[:, 0]
        hue = np.uint8(
            np.rint(np.linspace(0.0, 179.0, width, dtype=np.float32))
        )[None, :]
        lightness = np.uint8(
            np.rint(np.linspace(255.0, 0.0, height, dtype=np.float32))
        )[:, None]
        for centre, scale, weight, hls_centre in zip(
            centres, scales, weights, centre_hls, strict=True
        ):
            hls = np.empty((height, width, 3), np.uint8)
            hls[:, :, 0] = hue
            hls[:, :, 1] = lightness
            hls[:, :, 2] = hls_centre[2]
            candidate_rgb = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)
            candidate_lab = cv2.cvtColor(
                candidate_rgb, cv2.COLOR_RGB2LAB
            ).astype(np.float32)
            delta = (candidate_lab - centre) / scale
            distance = np.sum((delta * delta) * distance_weights, axis=2)
            membership = np.exp(-0.5 * distance) * weight
            # At black and white every hue collapses to the same RGB value.
            # Keep those achromatic endpoints in the neutral strip instead of
            # drawing a semantically meaningless all-hue contour.
            visible_chroma = (
                candidate_rgb.max(axis=2).astype(np.int16)
                - candidate_rgb.min(axis=2).astype(np.int16)
            ) >= 4
            visible_membership = membership * visible_chroma
            if combine_modes == "maximum":
                probability = np.maximum(probability, visible_membership)
            else:
                probability += visible_membership
        return np.clip(probability, 0.0, 1.0)

    @staticmethod
    def _chromatic_components(
        centres: np.ndarray, scales: np.ndarray
    ) -> np.ndarray:
        if not len(centres):
            return np.zeros((0,), dtype=bool)
        import cv2

        chroma = np.hypot(centres[:, 1] - 128.0, centres[:, 2] - 128.0)
        tolerance = np.maximum(
            2.5, 0.75 * np.minimum(scales[:, 1], scales[:, 2])
        )
        lab = np.clip(np.rint(centres), 0, 255).astype(np.uint8)[:, None, :]
        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)[:, 0]
        hls = cv2.cvtColor(rgb[:, None, :], cv2.COLOR_RGB2HLS)[:, 0]
        rgb_range = (
            rgb.max(axis=1).astype(np.int16) - rgb.min(axis=1).astype(np.int16)
        )
        # Tiny channel differences in a dark or pale neutral sample can have a
        # nominal Lab hue even though hue is not visually meaningful. Keep such
        # modes in the neutral strip. A pale tint remains chromatic when either
        # its HLS saturation or its visible channel separation is substantial.
        hue_is_visible = (hls[:, 2] >= 32) | (rgb_range >= 12)
        return (chroma >= tolerance) & hue_is_visible

    def _update_tooltip(self) -> None:
        self.setToolTip(
            "A neutral-value strip followed by an HSV hue/tint/shade slice. "
            f"Contour lines show fitted {self._class_name} membership in the "
            "model's Lab space. Neutral modes are evaluated only in the neutral "
            "strip; each chromatic mode is projected at its measured saturation "
            "in the hue/tone area so pale tints remain visible without spreading "
            "grey evidence or near-neutral evidence "
            "across unrelated hues. "
            "Circle labels show the leading learned colour-mode occurrences in "
            "painted references, isolated reference seeds, or the labelled fallback."
        )

    @classmethod
    def _hsv_projection(cls, width: int, height: int) -> np.ndarray:
        """Return a neutral strip plus white-to-pure-to-black HSV hues."""

        neutral_width = min(cls.NEUTRAL_COLUMNS, max(1, width // 4))
        hue_width = max(1, width - neutral_width)
        hue = np.linspace(0.0, 179.0, hue_width, dtype=np.float32)
        row = np.linspace(0.0, 1.0, height, dtype=np.float32)
        saturation = np.where(row <= 0.5, row * 2.0, 1.0) * 255.0
        value = np.where(row <= 0.5, 1.0, (1.0 - row) * 2.0) * 255.0
        hsv = np.empty((height, width, 3), np.uint8)
        hsv[:, neutral_width:, 0] = np.uint8(np.rint(hue))[None, :]
        hsv[:, neutral_width:, 1] = np.uint8(np.rint(saturation))[:, None]
        hsv[:, neutral_width:, 2] = np.uint8(np.rint(value))[:, None]
        hsv[:, :neutral_width, 0] = 0
        hsv[:, :neutral_width, 1] = 0
        hsv[:, :neutral_width, 2] = np.uint8(
            np.rint(np.linspace(255.0, 0.0, height, dtype=np.float32))
        )[:, None]
        return hsv

    @classmethod
    def _project_centres(
        cls,
        centres: np.ndarray,
        scales: np.ndarray,
        width: int,
        height: int,
    ) -> np.ndarray:
        """Place fitted modes in the neutral strip or hue/tone projection."""

        import cv2

        if not len(centres):
            return np.empty((0, 2), np.float32)
        lab = np.clip(np.rint(centres), 0, 255).astype(np.uint8)[:, None, :]
        hls = cv2.cvtColor(
            cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), cv2.COLOR_RGB2HLS
        )[:, 0].astype(np.float32)
        neutral_width = min(cls.NEUTRAL_COLUMNS, max(1, width // 4))
        hue_width = max(1, width - neutral_width)
        chromatic = cls._chromatic_components(centres, scales)
        columns = neutral_width + hls[:, 0] / 179.0 * max(1, hue_width - 1)
        columns[~chromatic] = neutral_width * 0.5
        rows = (1.0 - hls[:, 1] / 255.0) * max(1, height - 1)
        return np.column_stack((columns, rows)).astype(np.float32)

    @staticmethod
    def _contour_edge(probability: np.ndarray, level: float) -> np.ndarray:
        import cv2

        mask = np.pad(
            np.uint8(probability >= level), 1, mode="constant"
        )
        edge = cv2.morphologyEx(
            mask,
            cv2.MORPH_GRADIENT,
            np.ones((3, 3), np.uint8),
        )
        return edge[1:-1, 1:-1] > 0

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
            "neutral  |  HSV hue",
        )

        image_width = max(1, self._image.width() - 1)
        image_height = max(1, self._image.height() - 1)
        # Dense painted references can retain dozens of colour modes. Drawing
        # every circle and percentage makes the diagnostic unreadable, while
        # the contours already include every fitted mode. Label only the twelve
        # most frequent modes.
        displayed_modes = np.argsort(self._weights)[::-1][:12]
        for index in displayed_modes:
            point = self._centre_points[index]
            weight = self._weights[index]
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
        self.foreground_start_heading = QLabel("Starting automatic colours", self)
        self.foreground_start_heading.setStyleSheet(
            "font-weight: 600; margin-top: 6px;"
        )
        self.foreground_start_heading.setVisible(False)
        self.foreground_start_label = QLabel("", self)
        self.foreground_start_label.setWordWrap(True)
        self.foreground_start_label.setTextFormat(Qt.TextFormat.RichText)
        self.foreground_start_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.foreground_start_label.setToolTip(
            "The highest-frequency colour modes detected in the isolated reference "
            "seeds above the ruler. Percentages are their proportions in those "
            "reference pixels; painted foreground references replace this automatic start."
        )
        self.foreground_start_label.setVisible(False)
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
        layout.addWidget(self.foreground_start_heading)
        layout.addWidget(self.foreground_start_label)
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
        self._update_colour_summary()

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
        self._update_colour_summary()

    def refresh_status(self) -> None:
        if self._node is not None:
            self.status_label.setText(
                f"Status: {self._node.status.value}. {self._node.status_detail}"
            )

    def _clear_parameters(self) -> None:
        while self.parameter_form.rowCount():
            self.parameter_form.removeRow(0)
        self._parameter_widgets.clear()

    def _update_colour_summary(self) -> None:
        node_id = None if self._node is None else self._node.identifier
        foreground_visible = node_id == "foreground_segmentation"
        self.foreground_start_heading.setVisible(foreground_visible)
        self.foreground_start_label.setVisible(foreground_visible)
        if not foreground_visible:
            return
        layers = getattr(self._analysis_result, "layers", None)
        profile = getattr(layers, "foreground_colour_profile", None)
        self._update_foreground_starting_colours(profile)

    def _update_foreground_starting_colours(self, profile) -> None:
        """Render the automatic foreground anchors as explicit colour swatches."""

        if profile is None:
            self.foreground_start_label.setText(
                "Run the foreground node to detect starting colours."
            )
            return
        source = getattr(profile, "source", "unknown")
        if source == "painted":
            self.foreground_start_label.setText(
                "Bypassed — applied foreground references supply the colour modes."
            )
            return
        centres = np.asarray(
            profile.component_centres_lab or (profile.centre_lab,),
            dtype=np.float32,
        ).reshape(-1, 3)
        weights = np.asarray(
            profile.component_weights or (1.0,), dtype=np.float32
        ).reshape(-1)
        if len(weights) != len(centres):
            weights = np.full(
                len(centres), 1.0 / max(1, len(centres)), np.float32
            )
        order = np.argsort(-weights)[:8]
        import cv2

        bgr = cv2.cvtColor(
            centres[order]
            .round()
            .clip(0, 255)
            .astype(np.uint8)
            .reshape(-1, 1, 3),
            cv2.COLOR_LAB2BGR,
        ).reshape(-1, 3)
        swatches = []
        for colour_bgr, weight in zip(bgr, weights[order], strict=True):
            colour = QColor(
                int(colour_bgr[2]), int(colour_bgr[1]), int(colour_bgr[0])
            ).name().upper()
            swatches.append(
                f'<span style="color:{colour}; font-size:18px;">■</span> '
                f'{colour} ({float(weight):.0%})'
            )
        source_text = (
            "Isolated reference seeds"
            if source == "isolated_reference_seeds"
            else "Fallback from high-confidence image pixels"
        )
        sample_count = int(getattr(profile, "source_sample_count", 0))
        self.foreground_start_label.setText(
            f"{source_text}; {sample_count:,} source pixels:<br>"
            + " &nbsp; ".join(swatches)
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
