"""Compact native chart data for the pinned PlantCV analyze.color return.

Only already-computed observation metadata is consumed. No source image or
label raster is transferred to CPU. Rendering is native Qt; agreement claims
apply to the returned hue-chart data, not Altair/Qt pixel rendering.
"""
from dataclasses import dataclass
import math

from seedvision.cuda.plantcv_color import ColorAnalysisResult, MAX_LABELS


MAX_CHART_POINTS = MAX_LABELS*180


@dataclass(frozen=True, slots=True)
class HueHistogramSeries:
    label_id: int
    sample: str
    bins: tuple[float, ...]
    frequencies: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class HueHistogramChart:
    status: str
    series: tuple[HueHistogramSeries, ...]
    variable: str = 'hue_frequencies'
    x_title: str = 'Hue bin centre (degrees)'
    y_title: str = 'Pixels (%)'
    rendering: str = 'native Qt; pinned PlantCV returned data, independent rendering'

    def data_records(self):
        """Return the exact sample/value/label rows used by Outputs.plot_dists."""
        return [{'sample': item.sample, 'value': value, 'label': label}
            for item in self.series for label, value in zip(item.bins, item.frequencies)]


def hue_histogram_chart(observations):
    """Build the public hue-chart data, including empty RGB/LAB charts.

    PlantCV always calls plot_dists('hue_frequencies'), even when the selected
    colour space does not publish hue frequencies. In a fresh observation store
    RGB/LAB therefore return a valid chart with no rows, not fabricated hue data.
    Failed analysis does not yield a complete chart.
    """
    if not isinstance(observations, ColorAnalysisResult) or observations.status != 'available':
        raise ValueError('A complete available colour-observation result is required')
    if observations.colorspace not in ('rgb', 'lab', 'hsv', 'all'):
        raise ValueError('Unsupported colour-space declaration')
    if len(observations.labels) > MAX_LABELS:
        raise ValueError('Hue-chart sample count exceeds its compact metadata limit')
    series = []
    for item in observations.labels:
        histograms = [histogram for histogram in item.histograms if histogram.variable == 'hue_frequencies']
        if not histograms:
            if observations.colorspace in ('hsv', 'all'):
                raise ValueError('An expected hue distribution is absent')
            continue
        if len(histograms) != 1 or observations.colorspace not in ('hsv', 'all'):
            raise ValueError('Hue distributions disagree with the selected colour space')
        histogram = histograms[0]
        bins, values = tuple(histogram.bin_labels), tuple(histogram.values)
        if bins != tuple(range(1, 360, 2)) or len(values) != 180:
            raise ValueError('Require the pinned 180 hue bins with odd-degree centres')
        if any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 100
                for value in values):
            raise ValueError('Hue frequencies must be finite percentages in [0,100]')
        if type(item.label_id) is not int or not 1 <= item.label_id <= MAX_LABELS or not isinstance(item.sample, str):
            raise ValueError('Require a bounded label ID and sample string')
        series.append(HueHistogramSeries(item.label_id, item.sample, bins, values))
    if len({item.label_id for item in series}) != len(series) or len({item.sample for item in series}) != len(series):
        raise ValueError('Hue-chart sample and label identities must be unique')
    if sum(len(item.bins) for item in series) > MAX_CHART_POINTS:
        raise ValueError('Hue-chart points exceed their compact metadata limit')
    return HueHistogramChart('available' if series else 'empty', tuple(series))
