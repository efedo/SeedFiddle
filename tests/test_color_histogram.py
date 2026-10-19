from dataclasses import replace
import json
from pathlib import Path
import subprocess
import unittest

import numpy as np
import torch
from PySide6.QtWidgets import QApplication

from seedvision.benchmarking.color_histogram import hue_histogram_chart
from seedvision.cuda.plantcv_color import analyze_color_cuda, ColorAnalysisResult
from seedvision.ui.color_histogram import HueHistogramView


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT/'artifacts'/'plantcv-reference-env'/'Scripts'/'python.exe'
SCRIPT = r'''
import hashlib, importlib, json, sys
from pathlib import Path
from importlib.metadata import version
import numpy as np
from plantcv import plantcv as pcv
assert version('plantcv') == '4.11.3'
assert version('numpy') == '2.2.6'
assert version('scipy') == '1.15.3'
assert version('opencv-python') == '4.11.0.86'
assert version('altair') == '5.5.0'
module = importlib.import_module('plantcv.plantcv.analyze.color')
assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == '1a28a86a82af612f65ffd55a142df8d23673cd5ba80ec9a2fbfda10606f7b6b8'
classes = importlib.import_module('plantcv.plantcv.classes')
assert hashlib.sha256(Path(classes.__file__).read_bytes()).hexdigest() == '8ee436f54c1bf10bdf6eaada1692205d61a209d65fb3687a636e87e97dc8b957'
pcv.params.debug = None
payload = json.loads(sys.stdin.read())
chart = pcv.analyze.color(np.asarray(payload['image'], dtype=np.uint8),
    np.asarray(payload['mask'], dtype=np.int32), n_labels=2,
    colorspaces=payload['colorspace'], label=['first', 'second'])
print(json.dumps(chart.data.to_dict('records')))
'''


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for actual colour chart observations')
class ColorHistogramTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def observations(self, space='all'):
        image = np.random.default_rng(1717).integers(1, 256, (5, 7, 3), dtype=np.uint8)
        mask = np.ones((5, 7), dtype=np.int32)
        mask[:, 4:] = 2
        result = analyze_color_cuda(torch.as_tensor(image, device='cuda'),
            torch.as_tensor(mask, device='cuda'), n_labels=2, colorspaces=space, label=['first', 'second'])
        return image, mask, result


    def test_native_sample_selection_and_empty_transition_preserve_all_observations(self):
        _, _, result = self.observations()
        chart = hue_histogram_chart(result)
        before = chart.data_records()
        view = HueHistogramView()
        view.resize(420, 280)
        view.set_chart(chart)
        view.show()
        self.app.processEvents()
        self.assertEqual(view.samples.count(), 2)
        self.assertEqual(view.plot.series.sample, 'first_1')
        view.samples.setCurrentIndex(1)
        self.app.processEvents()
        self.assertEqual(view.plot.series.sample, 'second_2')
        self.assertFalse(view.grab().isNull())
        self.assertEqual(chart.data_records(), before)
        self.assertEqual(result.labels[0].sample, 'first_1')
        view.set_chart(chart)
        self.assertEqual(view.plot.series.label_id, 2)
        _, _, empty_result = self.observations('rgb')
        view.set_chart(hue_histogram_chart(empty_result))
        self.assertFalse(view.samples.isEnabled())
        self.assertIsNone(view.plot.series)
        view.set_chart(None)
        self.assertIn('No colour-chart', view.description.text())
        view.close()

    def test_invalid_or_incomplete_observations_never_form_a_complete_chart(self):
        _, _, result = self.observations('hsv')
        with self.assertRaises(ValueError):
            hue_histogram_chart(ColorAnalysisResult('reference_error', 'hsv', (), error_type='UnboundLocalError'))
        first = result.labels[0]
        histogram = first.histograms[0]
        for altered in (replace(histogram, values=(float('nan'), *histogram.values[1:])),
                        replace(histogram, bin_labels=tuple(range(180))),
                        replace(histogram, values=histogram.values[:-1])):
            with self.assertRaises(ValueError):
                hue_histogram_chart(replace(result, labels=(replace(first, histograms=(altered,)),)))
        with self.assertRaises(ValueError):
            hue_histogram_chart(replace(result, labels=(first, first)))
        with self.assertRaises(ValueError):
            hue_histogram_chart(replace(result, labels=(replace(first, histograms=()),)))


if __name__ == '__main__':
    unittest.main()
