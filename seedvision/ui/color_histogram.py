"""Native, compact hue-chart display; selection never changes analysis."""
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QComboBox, QLabel, QVBoxLayout, QWidget

from seedvision.benchmarking.color_histogram import HueHistogramChart


class _HuePlot(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.series = None
        self.setMinimumSize(240, 160)

    def set_series(self, series):
        self.series = series
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        palette = self.palette()
        painter.fillRect(self.rect(), palette.base())
        painter.setPen(palette.text().color())
        area = QRectF(44, 10, max(1, self.width()-56), max(1, self.height()-62))
        painter.drawLine(area.bottomLeft(), area.bottomRight())
        painter.drawLine(area.bottomLeft(), area.topLeft())
        for tick in (0, 90, 180, 270, 360):
            x = area.left()+tick/360.0*area.width()
            painter.drawLine(x, area.bottom(), x, area.bottom()+3)
            left = min(self.width()-36, max(0, x-18))
            painter.drawText(QRectF(left, area.bottom()+4, 36, 18), Qt.AlignmentFlag.AlignCenter, str(tick))
        painter.drawText(QRectF(area.left(), area.bottom()+25, area.width(), 20),
            Qt.AlignmentFlag.AlignCenter, 'Hue (degrees)')
        if self.series is None:
            painter.drawText(area, Qt.AlignmentFlag.AlignCenter, 'No hue distribution was returned.')
            return
        maximum = max(self.series.frequencies, default=0.0)
        upper = maximum if maximum > 0 else 1.0
        painter.drawText(QRectF(0, 2, 40, 20), Qt.AlignmentFlag.AlignRight, f'{upper:.3g}%')
        painter.drawText(QRectF(0, area.bottom()-12, 40, 20), Qt.AlignmentFlag.AlignRight, '0%')
        path = QPainterPath(area.bottomLeft())
        for bin_value, frequency in zip(self.series.bins, self.series.frequencies):
            x = area.left()+float(bin_value)/360.0*area.width()
            y = area.bottom()-float(frequency)/upper*area.height()
            path.lineTo(x, y)
        path.lineTo(area.bottomRight())
        path.closeSubpath()
        colour = QColor('#8056bd')
        colour.setAlpha(175)
        painter.fillPath(path, colour)
        painter.setPen(QPen(palette.text().color(), 0.7))
        painter.drawPath(path)


class HueHistogramView(QWidget):
    """Choose a returned sample for display; all original chart rows are retained."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.chart = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.description = QLabel('Returned hue distribution — native display')
        self.description.setTextFormat(Qt.TextFormat.PlainText)
        self.description.setWordWrap(True)
        layout.addWidget(self.description)
        self.samples = QComboBox()
        self.samples.setObjectName('hueHistogramDisplaySample')
        self.samples.setToolTip('Choose a sample for display; analysis is unchanged.')
        layout.addWidget(self.samples)
        self.plot = _HuePlot(self)
        layout.addWidget(self.plot, 1)
        self.samples.currentIndexChanged.connect(self._selected)
        self.set_chart(None)

    def set_chart(self, chart):
        if chart is not None and not isinstance(chart, HueHistogramChart):
            raise TypeError('chart must be HueHistogramChart or None')
        previous = self.samples.currentData()
        self.chart = chart
        self.samples.blockSignals(True)
        self.samples.clear()
        if chart is not None:
            for item in chart.series:
                self.samples.addItem(item.sample, item.label_id)
        index = self.samples.findData(previous)
        if index >= 0:
            self.samples.setCurrentIndex(index)
        self.samples.blockSignals(False)
        self.samples.setEnabled(bool(chart and chart.series))
        self.description.setText('Returned hue distribution — native display' if chart and chart.series
            else 'The selected colour space returned no hue-chart data.' if chart is not None
            else 'No colour-chart result is available.')
        self._selected(self.samples.currentIndex())

    def _selected(self, index):
        self.plot.set_series(self.chart.series[index]
            if self.chart is not None and 0 <= index < len(self.chart.series) else None)
