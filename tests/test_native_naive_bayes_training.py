"""Public sampled-RGB trainer, TSV handoff and frozen-model prediction gates."""
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

from seedvision.benchmarking import native_naive_bayes_training as native
from seedvision.cuda import plantcv_naive_bayes as inference
from seedvision.cuda.plantcv_naive_bayes_training import BayesTrainingError, BayesTrainingCancelled
from tests.test_plantcv_naive_bayes import _reference as reference_masks
from tests.test_plantcv_naive_bayes_training import REFERENCE

SCRIPT = r'''
import hashlib, importlib, json, sys
from importlib.metadata import version
from pathlib import Path
import cv2, numpy as np
module=importlib.import_module('plantcv.learn.naive_bayes')
assert version('plantcv')=='4.11.3'
assert version('numpy')=='2.2.6'
assert version('scipy')=='1.15.3'
assert version('opencv-python')=='4.11.0.86'
assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()=='f59c95c493d189b6110d47a60a97c86e306ab87583a2f128eff8c389aa6b56eb'
try:
    module.naive_bayes_multiclass(sys.argv[1],sys.argv[2])
    print(json.dumps(dict(tsv=Path(sys.argv[2]).read_text())))
except Exception as e:
    print(json.dumps(dict(error=type(e).__name__)))
'''


def public(raw):
    if not REFERENCE.is_file():
        raise AssertionError('pinned PlantCV environment required')
    with tempfile.TemporaryDirectory() as directory:
        sample, output = Path(directory)/'samples.tsv', Path(directory)/'model.tsv'
        sample.write_bytes(raw)
        run = subprocess.run([str(REFERENCE), '-c', SCRIPT, str(sample), str(output)],
            capture_output=True, text=True, timeout=120)
    if run.returncode:
        raise AssertionError(run.stderr)
    return json.loads(run.stdout)


def sample_bytes(count=25):
    rng = np.random.default_rng(991)
    arrays = [rng.integers(0, 256, (count, 3), dtype=np.uint8) for _ in range(3)]
    lines = ['seed\tbackground\tseed']
    for row in zip(*arrays):
        lines.append('\t'.join('"'+','.join(map(str, triple))+'"' for triple in row))
    return ('\r\n'.join(lines)+'\r\n\r\n').encode('utf-8')


def constant_value_samples(count=6, value=255):
    rows = ['seed']
    for index in range(count):
        rgb = [value,30+index*7,80+index*11]
        rgb = rgb[index%3:]+rgb[:index%3]
        rows.append(','.join(map(str,rgb)))
    return ('\n'.join(rows)+'\n').encode('utf-8')


class BayesSampleParsingTests(unittest.TestCase):
    def test_duplicate_headers_quoted_cells_whitespace_missing_and_extra_empty_columns(self):
        raw = b'"seed"\tsoil\t"seed"\r\n" 1, 2,3"\t4,5,6\t7,8,9\t\r\n10,11,12\r\n\r\n'
        table = native.parse_bayes_samples(raw)
        self.assertEqual(table.headers, ('"seed"', 'soil', '"seed"'))
        self.assertEqual(table.class_names, ('"seed"', 'soil'))
        self.assertEqual(table.rgb_bytes, (bytes((1,2,3,7,8,9,10,11,12)), bytes((4,5,6))))
        self.assertEqual(table.sample_counts, (3, 1))
        self.assertEqual(table.source_sha256, hashlib.sha256(raw).hexdigest())
        self.assertEqual(table.source_bytes, len(raw))
        self.assertEqual(table, native.parse_bayes_samples(raw.decode('utf-8')))
        lf = native.parse_bayes_samples(raw.replace(b'\r\n', b'\n'))
        self.assertEqual(table.parsed_sha256, lf.parsed_sha256)
        self.assertNotEqual(table.source_sha256, lf.source_sha256)

    def test_malformed_uint8_and_unsupported_empty_labels_are_explicit(self):
        bad = [b'', b'\xff', b'\n1,2,3\n', b'a\n1,2\n', b'a\n1,2,3,4\n', b'a\n \n',
            b'a\n1.5,2,3\n', b'a\n-1,2,3\n', b'a\n256,2,3\n', b'a\n1,2,3\t4,5,6\n']
        for raw in bad:
            with self.subTest(raw=raw), self.assertRaises(BayesTrainingError):
                native.parse_bayes_samples(raw)
        for cap, value in (('MAX_SAMPLE_BYTES', 4), ('MAX_LINE_BYTES', 4), ('MAX_LABEL_BYTES', 2)):
            with patch.object(native, cap, value), self.assertRaises(BayesTrainingError):
                native.parse_bayes_samples(b'seed\n1,2,3\n')
        with patch.object(native._fit, 'MAX_SAMPLES', 1), self.assertRaises(BayesTrainingError):
            native.parse_bayes_samples(b'a\n1,2,3\n4,5,6\n')

    def test_bounded_file_read_same_bytes_distinct_origins_and_cancellation(self):
        class Tracked(io.BytesIO):
            def read(self, size=-1):
                self.requested = size
                return super().read(size)
        stream = Tracked(b'x'*20)
        with patch.object(native, 'MAX_SAMPLE_BYTES', 16), patch.object(Path, 'open', return_value=stream):
            with self.assertRaises(BayesTrainingError):
                native.read_bayes_samples('placeholder')
        self.assertEqual(stream.requested, 17)
        with self.assertRaises(BayesTrainingCancelled):
            native.parse_bayes_samples(sample_bytes(), cancel=lambda: True)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required')
class NativeBayesSampleTrainingTests(unittest.TestCase):
    def test_actual_public_constant_value_success_and_true_singular_failures(self):
        from seedvision.model_io.naive_bayes_provenance import parse_training_record,CONSTANT_NUMERICAL_PROFILE
        for n,value in ((3,200),(6,255)):
            with self.subTest(n=n,value=value):
                raw = constant_value_samples(n,value)
                expected = public(raw)
                self.assertNotIn('error',expected)
                source = inference.parse_plantcv_pdf(expected['tsv'])
                result = native.train_bayes_samples(native.parse_bayes_samples(raw))
                self.assertTrue(all(x.device.type=='cuda' for x in result.samples))
                record = parse_training_record(result.provenance_json,result.model)
                fits = record.metadata['channel_fits'][0]
                self.assertNotIn('constant_value',fits[0])
                self.assertNotIn('constant_value',fits[1])
                self.assertEqual(fits[2]['constant_value'],value)
                self.assertEqual(fits[2]['numerical_profile'],CONSTANT_NUMERICAL_PROFILE)
                if value == 255:
                    self.assertGreater(fits[2]['mean'],255)
                for c,fit in enumerate(result.fit.channels[0]):
                    np.testing.assert_allclose(result.model.probabilities[0][c],source.probabilities[0][c],
                        rtol=0,atol=fit.pdf_absolute_budget)
        raw = b'seed\n0,0,0\n0,0,0\n0,0,0\n'
        self.assertIn('error',public(raw))
        with self.assertRaisesRegex(BayesTrainingError,'singular'):
            native.train_bayes_samples(native.parse_bayes_samples(raw))

    def test_actual_public_trainer_pdf_order_export_and_downstream_masks(self):
        raw = sample_bytes()
        expected = public(raw)
        self.assertNotIn('error', expected)
        source = inference.parse_plantcv_pdf(expected['tsv'])
        table = native.parse_bayes_samples(raw)
        result = native.train_bayes_samples(table)
        self.assertEqual(result.model, inference.parse_plantcv_pdf(result.tsv_bytes))
        self.assertEqual(result.model.class_names, source.class_names)
        self.assertEqual(result.model.class_names, ('seed', 'background'))
        self.assertTrue(all(x.device.type == 'cuda' for x in result.samples))
        self.assertEqual([line.split(b'\t')[:2] for line in result.tsv_bytes.splitlines()[1:]],
            [[name.encode(), channel.encode()] for channel in inference.CHANNELS for name in source.class_names])
        for index, channels in enumerate(result.fit.channels):
            for c, fit in enumerate(channels):
                np.testing.assert_allclose(result.model.probabilities[index][c], source.probabilities[index][c],
                    rtol=0, atol=fit.pdf_absolute_budget)
        image = np.random.default_rng(222).integers(0,256,(19,31,3),dtype=np.uint8)
        predicted = inference.naive_bayes_masks(torch.tensor(image, device='cuda'), result.model)
        # Applying the *actual native bytes* must match the public classifier exactly.
        for name, mask in reference_masks(image, result.tsv_bytes).items():
            np.testing.assert_array_equal(predicted.masks[name].cpu().numpy(), mask)
        # On this fixture the independently trained source model selects the same classes.
        for name, mask in reference_masks(image, expected['tsv'].encode()).items():
            np.testing.assert_array_equal(predicted.masks[name].cpu().numpy(), mask)
        provenance = json.loads(result.provenance_json)
        self.assertEqual(provenance['sample_sha256'], hashlib.sha256(raw).hexdigest())
        self.assertEqual(provenance['tables_sha256'], result.model.tables_sha256)
        self.assertEqual(provenance['model_sha256'], hashlib.sha256(result.tsv_bytes).hexdigest())
        self.assertFalse(provenance['scientific_validation'])
        self.assertIn('empirical', provenance['channel_fits'][0][0]['numerical_profile'])

    def test_public_single_class_success_and_parser_failures_are_not_invented(self):
        raw = b'a\n2,50,180\n140,3,80\n220,100,13\n'
        self.assertNotIn('error', public(raw))
        self.assertEqual(native.train_bayes_samples(native.parse_bayes_samples(raw)).model.class_names, ('a',))
        for invalid in (b'a\n256,2,3\n4,5,6\n', b'a\n1,2\n', b'a\n1,2,3\t4,5,6\n'):
            self.assertIn('error', public(invalid))
            with self.assertRaises(BayesTrainingError):
                native.train_bayes_samples(native.parse_bayes_samples(invalid))

    def test_same_byte_files_same_model_identity_distinct_provenance_and_inference_without_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = (Path(directory)/'first.tsv', Path(directory)/'second.tsv')
            for path in paths:
                path.write_bytes(sample_bytes(9))
            first, second = [native.train_bayes_sample_file(path) for path in paths]
            self.assertEqual(first.tsv_bytes, second.tsv_bytes)
            self.assertEqual(first.model.tables_sha256, second.model.tables_sha256)
            records = [json.loads(r.provenance_json) for r in (first, second)]
            self.assertNotEqual(records[0]['sample_origin_path'], records[1]['sample_origin_path'])
            self.assertEqual(records[0]['parsed_samples_sha256'], records[1]['parsed_samples_sha256'])
            output = Path(directory)/'trained-model.tsv'
            output.write_bytes(first.tsv_bytes)
            for path in paths:
                path.unlink()
            self.assertEqual(inference.load_plantcv_pdf(output), first.model)

    def test_final_high_level_callbacks_guard_samples_pdf_storage_and_raised_callbacks(self):
        table = native.parse_bayes_samples(sample_bytes(9))
        calls = 0
        def count():
            nonlocal calls
            calls += 1
            return False
        native.train_bayes_samples(table, cancel=count)
        last_checkpoint = calls
        empty, zeros = torch.empty, torch.zeros
        for kind in ('sample', 'pdf_storage', 'pdf_metadata', 'sample_and_raise'):
            with self.subTest(kind=kind):
                allocated = {}
                calls = 0
                def tracked_empty(*args, **kwargs):
                    tensor = empty(*args, **kwargs)
                    if tensor.dtype == torch.uint8 and tensor.ndim == 2 and tensor.shape[1] == 3:
                        allocated.setdefault('sample', tensor)
                    return tensor
                def tracked_zeros(*args, **kwargs):
                    tensor = zeros(*args, **kwargs)
                    if tensor.dtype == torch.float64 and tuple(tensor.shape) == (2,3,256):
                        allocated['pdf'] = tensor
                    return tensor
                def mutate_last():
                    nonlocal calls
                    calls += 1
                    if calls == last_checkpoint:
                        if kind == 'pdf_storage':
                            allocated['pdf'].data = allocated['pdf'].clone()
                        elif kind == 'pdf_metadata':
                            allocated['pdf'].transpose_(1, 2)
                        else:
                            allocated['sample'].add_(1)
                            if kind == 'sample_and_raise':
                                raise RuntimeError('caller failure after mutation')
                    return False
                with patch.object(torch, 'empty', side_effect=tracked_empty), patch.object(torch, 'zeros', side_effect=tracked_zeros):
                    with self.assertRaisesRegex(BayesTrainingError, 'changed before publication'):
                        native.train_bayes_samples(table, cancel=mutate_last)

    def test_combined_preflight_forged_parsed_bytes_cancellation_and_changed_file(self):
        table = native.parse_bayes_samples(sample_bytes(9))
        with patch.object(native.torch, 'empty', side_effect=AssertionError('allocated before preflight')):
            with patch.object(native._fit, 'MAX_WORK', 1), self.assertRaises(BayesTrainingError):
                native.train_bayes_samples(table)
            changed = replace(table, rgb_bytes=(b'\x00'*len(table.rgb_bytes[0]), table.rgb_bytes[1]))
            with self.assertRaises(BayesTrainingError):
                native.train_bayes_samples(changed)
        calls = 0
        def cancel():
            nonlocal calls
            calls += 1
            return calls > 5
        with self.assertRaises(BayesTrainingCancelled):
            native.train_bayes_samples(table, cancel=cancel)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'samples.tsv'
            path.write_bytes(sample_bytes(9))
            calls = 0
            def change_file():
                nonlocal calls
                calls += 1
                if calls == 4:  # After the selected bytes were parsed.
                    path.write_bytes(sample_bytes(10))
                return False
            with self.assertRaisesRegex(BayesTrainingError, 'changed'):
                native.train_bayes_sample_file(path, cancel=change_file)
            path.write_bytes(sample_bytes(9))
            calls = 0
            def count():
                nonlocal calls
                calls += 1
                return False
            native.train_bayes_sample_file(path, cancel=count)
            last_checkpoint = calls
            calls = 0
            def change_at_last_checkpoint():
                nonlocal calls
                calls += 1
                if calls == last_checkpoint:
                    path.write_bytes(sample_bytes(10))
                return False
            with self.assertRaisesRegex(BayesTrainingError, 'changed'):
                native.train_bayes_sample_file(path, cancel=change_at_last_checkpoint)


if __name__ == '__main__':
    unittest.main()
