"""CUDA Naive Bayes masks versus the pinned PlantCV public function."""
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from seedvision.cuda import plantcv_naive_bayes as native

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT/'artifacts'/'plantcv-reference-env'/'Scripts'/'python.exe'
REFERENCE_SCRIPT = r'''
import hashlib, importlib, json, sys
from importlib.metadata import version
from pathlib import Path
import numpy as np
from plantcv import plantcv as pcv
assert version('plantcv') == '4.11.3'
assert version('numpy') == '2.2.6'
assert version('opencv-python') == '4.11.0.86'
module = importlib.import_module('plantcv.plantcv.naive_bayes_classifier')
assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == 'd3f40429ead7aa5deb298fa78d484d0f6d89c9cf05e706e07fdf7b97fedc3049'
image = np.asarray(json.loads(sys.stdin.read())['image'], dtype=np.uint8)
pcv.params.debug = None
masks = module.naive_bayes_classifier(rgb_img=image, pdf_file=sys.argv[1])
print(json.dumps({name: mask.tolist() for name, mask in masks.items()}))
'''


def _row(name, channel, values):
    assert len(values) == 256
    return '\t'.join((name, channel, *(repr(float(value)) for value in values)))


def _pdf(duplicate=False):
    indexes = np.arange(256, dtype=np.float64)
    rows = ['class\tchannel\t'+'\t'.join(map(str, range(256)))]
    for index, name in enumerate(('leaf', 'soil', 'shadow')):
        shift = index*0.31
        tables = (
            0.02+((indexes*(0.013+index*0.002)+shift) % 1.0),
            0.04+((indexes*(0.007+index*0.003)+shift*2) % 0.94),
            0.03+((indexes*(0.011+index*0.001)+shift*3) % 0.96),
        )
        rows.extend(_row(name, channel, values)
            for channel, values in zip(native.CHANNELS, tables))
        if duplicate and index == 0:
            rows.append(_row(name, 'hue', np.full(256, 0.37)))
    return ('\n'.join(rows)+'\n').encode('utf-8')


def _constant_pdf(values):
    rows = ['class\tchannel\t'+'\t'.join(map(str, range(256)))]
    for name, value in values.items():
        rows.extend(_row(name, channel, np.full(256, value)) for channel in native.CHANNELS)
    return ('\n'.join(rows)+'\n').encode('utf-8')


def _image():
    return np.random.default_rng(771).integers(0, 256, (23, 29, 3), dtype=np.uint8)


def _reference(image, model_bytes):
    if not REFERENCE.is_file():
        raise AssertionError('Pinned PlantCV reference environment is required')
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/'frozen-synthetic.pdf.txt'
        path.write_bytes(model_bytes)
        result = subprocess.run([str(REFERENCE), '-c', REFERENCE_SCRIPT, str(path)],
            input=json.dumps({'image': image.tolist()}), capture_output=True, text=True, timeout=90)
    if result.returncode:
        raise AssertionError(result.stderr)
    return {name: np.asarray(mask, dtype=np.uint8)
        for name, mask in json.loads(result.stdout).items()}


class PlantCVNaiveBayesModelTests(unittest.TestCase):
    def test_parsing_duplicate_channels_newlines_unknown_channels_and_model_hash(self):
        raw = _pdf(duplicate=True)
        model = native.parse_plantcv_pdf(raw)
        self.assertEqual(model.class_names, ('leaf', 'soil', 'shadow'))
        self.assertEqual(model.source_sha256, hashlib.sha256(raw).hexdigest())
        self.assertEqual(model.probabilities[0][0], (0.37,)*256)
        crlf = raw.replace(b'\n', b'\r\n')
        self.assertEqual(native.parse_plantcv_pdf(crlf).probabilities, model.probabilities)
        extra = raw+(_row('leaf', 'ignored', np.ones(256))+'\n').encode()
        self.assertEqual(native.parse_plantcv_pdf(extra).probabilities, model.probabilities)

    def test_malformed_nonfinite_negative_missing_and_oversized_models_are_explicit(self):
        raw = _pdf()
        for invalid in (b'', b'header\nclass\thue\t1\n',
                raw.replace(b'0.02', b'nan', 1), raw.replace(b'0.02', b'-0.02', 1),
                b'header\n'+(_row('class', 'hue', np.ones(256))+'\n').encode(),
                _constant_pdf({'overflow': 1e200}), b'\xff', '\ud800'):
            with self.subTest(invalid=repr(invalid)[:40]), self.assertRaises(native.NaiveBayesError):
                native.parse_plantcv_pdf(invalid)
        for kwargs in ({'max_model_bytes': 16}, {'max_classes': 2},
                {'max_classes': True}, {'max_classes': native.MAX_CLASSES+1},
                {'max_model_bytes': native.MAX_MODEL_BYTES+1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(native.NaiveBayesError):
                native.parse_plantcv_pdf(raw, **kwargs)

    def test_file_loading_is_bounded_before_parse_even_when_file_size_changes(self):
        class TrackedStream(io.BytesIO):
            def read(self, size=-1):
                self.requested = size
                return super().read(size)
        stream = TrackedStream(b'x'*1024)
        with patch.object(Path, 'open', return_value=stream):
            with self.assertRaises(native.NaiveBayesError):
                native.load_plantcv_pdf('unread-placeholder', max_model_bytes=16)
        self.assertEqual(stream.requested, 17)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'valid.tsv'
            path.write_bytes(_pdf())
            self.assertEqual(native.load_plantcv_pdf(path), native.parse_plantcv_pdf(_pdf()))
        with patch.object(Path, 'open', side_effect=AssertionError('opened before limits')):
            with self.assertRaises(native.NaiveBayesError):
                native.load_plantcv_pdf('unread-placeholder', max_model_bytes=0)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for Naive Bayes inference')
class PlantCVNaiveBayesTests(unittest.TestCase):
    def assert_masks(self, actual, expected, device):
        self.assertEqual(tuple(actual.masks), tuple(expected))
        for name, mask in actual.masks.items():
            self.assertEqual(mask.device, device)
            self.assertEqual(mask.dtype, torch.uint8)
            self.assertTrue(mask.is_contiguous())
            np.testing.assert_array_equal(mask.cpu().numpy(), expected[name], err_msg=name)

    def test_public_masks_class_order_duplicate_channel_rows_and_input_preservation(self):
        image, raw = _image(), _pdf(duplicate=True)
        bgr = torch.as_tensor(image, device='cuda')
        before = bgr.clone()
        result = native.naive_bayes_masks(bgr, native.parse_plantcv_pdf(raw), tile_pixels=113)
        self.assert_masks(result, _reference(image, raw), bgr.device)
        self.assertEqual(result.class_names, ('leaf', 'soil', 'shadow'))
        self.assertEqual(result.model_sha256, hashlib.sha256(raw).hexdigest())
        self.assertEqual(result.reference_version, '4.11.3')
        self.assertEqual(result.reference_source_sha256, native.PLANTCV_SOURCE_SHA256)
        torch.testing.assert_close(bgr, before, rtol=0, atol=0)

    def test_first_class_ties_zero_underflow_and_large_finite_products_match_public(self):
        image = _image()
        for values in ({'first': 1.0, 'second': 1.0}, {'first': 0.0, 'second': 0.0},
                {'first': 1e-200, 'second': 1e-200}, {'first': 1e100, 'second': 1e-100},
                {'first': 1e-100, 'second': np.nextafter(1e-100, np.inf)}):
            with self.subTest(values=values):
                raw = _constant_pdf(values)
                bgr = torch.as_tensor(image, device='cuda')
                result = native.naive_bayes_masks(bgr, native.parse_plantcv_pdf(raw))
                self.assert_masks(result, _reference(image, raw), bgr.device)

    def test_custom_stream_and_tiled_likelihoods_match_public_call(self):
        image, raw = _image(), _pdf()
        bgr = torch.as_tensor(image, device='cuda')
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            result = native.naive_bayes_masks(bgr, native.parse_plantcv_pdf(raw), tile_pixels=13)
        stream.synchronize()
        self.assert_masks(result, _reference(image, raw), bgr.device)

    def test_invalid_frozen_models_preflight_and_cancellation_publish_no_partial_masks(self):
        model = native.parse_plantcv_pdf(_pdf())
        altered = tuple(tuple(tuple(value+0.01 if i == 0 else value
            for i, value in enumerate(channel)) for channel in channels)
            for channels in model.probabilities)
        bgr = torch.as_tensor(_image(), device='cuda')
        for forged in (replace(model, probabilities=altered), replace(model, source_bytes='invalid'),
                replace(model, class_names=None), replace(model, source_sha256=None),
                replace(model, probabilities=()), replace(model, class_names=('\ud800', 'soil', 'shadow'))):
            with self.subTest(forged=repr(forged)[:40]), self.assertRaises(native.NaiveBayesError):
                native.naive_bayes_masks(bgr, forged)
        for parameter in ('MAX_PIXELS', 'MAX_OUTPUT_BYTES', 'MAX_SCRATCH_BYTES', 'MAX_CUDA_LIVE_BYTES'):
            with self.subTest(cap=parameter), patch.object(native, parameter, 1):
                with patch.object(native.torch, 'tensor', side_effect=AssertionError('allocated before cap')):
                    with self.assertRaises(native.NaiveBayesError):
                        native.naive_bayes_masks(bgr, model)
        calls = 0
        def cancel():
            nonlocal calls
            calls += 1
            return calls >= 5
        before = bgr.clone()
        with self.assertRaises(native.NaiveBayesCancelled):
            native.naive_bayes_masks(bgr, model, tile_pixels=5, cancel=cancel)
        torch.testing.assert_close(bgr, before, rtol=0, atol=0)

    def test_largest_class_count_multi_tile_actual_allocation_stays_below_preflight(self):
        model = native.parse_plantcv_pdf(_constant_pdf({f'class{i}': 1.0 for i in range(native.MAX_CLASSES)}))
        bgr = torch.zeros((1, native.TILE_PIXELS*3+17, 3), dtype=torch.uint8, device='cuda')
        native.naive_bayes_masks(bgr[:, :1].contiguous(), model)
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        result = native.naive_bayes_masks(bgr, model)
        torch.cuda.synchronize()
        actual = torch.cuda.max_memory_allocated()-baseline
        declared = (bgr.shape[0]*bgr.shape[1]*native.MAX_CLASSES
            + native.MAX_CLASSES*3*256*8
            + native.TILE_PIXELS*native.SCRATCH_ENVELOPE_BYTES_PER_PIXEL)
        self.assertLessEqual(actual, declared)
        self.assertLessEqual(actual, native.MAX_CUDA_LIVE_BYTES)
        self.assertEqual(len(result.masks), native.MAX_CLASSES)
        self.assertTrue(bool(torch.all(result.masks['class0'] == 255).item()))
        self.assertTrue(bool(torch.all(result.masks[f'class{native.MAX_CLASSES-1}'] == 0).item()))


if __name__ == '__main__':
    unittest.main()
