
from tests.functional_units import functional_units
from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from seedvision.cuda import plantcv_kmeans_model as native

ROOT = Path(__file__).resolve().parents[1]


@functional_units
class KMeansModelTests(unittest.TestCase):
    def payload(self, **changes):
        value = dict(schema=native.SCHEMA, preprocessing=native.PREPROCESSING,
                     dtype='float64', patch_size=1, channels=1, centres=[[0], [2.5], [0]])
        value.update(changes)
        return json.dumps(value)

    def check_test_frozen_rows_order_duplicates_and_raw_vs_numeric_identity(self):
        raw = self.payload()
        model = native.parse_kmeans_model(raw)
        self.assertEqual(model.centres, ((0.0,), (2.5,), (0.0,)))
        self.assertEqual(model.source_sha256, sha256(raw.encode()).hexdigest())
        self.assertEqual(model.source_bytes, len(raw.encode()))
        native.validate_kmeans_model(model)
        formatted = native.parse_kmeans_model(json.dumps(json.loads(raw), indent=4))
        self.assertNotEqual(model.source_sha256, formatted.source_sha256)
        self.assertEqual(model.centres_sha256, formatted.centres_sha256)
        reordered = native.model_from_centres(((2.5,), (0.0,), (0.0,)), 1, 1)
        self.assertNotEqual(reordered.centres_sha256, model.centres_sha256)
        with self.assertRaises(FrozenInstanceError):
            model.patch_size = 2

    def check_test_canonical_encoding_preserves_ordered_numeric_and_binary64_identity(self):
        centres = ((9.25, 1.5, 255.0, -0.0), (2.125, 8.75, 0.5, 17.0))
        model = native.model_from_centres(centres, 2, 1)
        encoded = native.encode_kmeans_model(model)
        self.assertEqual(native.parse_kmeans_model(encoded), model)
        self.assertIn(b'-0.0', encoded)
        payload = json.loads(encoded)
        self.assertEqual(list(payload), ['schema', 'preprocessing', 'dtype', 'patch_size', 'channels', 'centres'])
        formatted = json.dumps(payload, indent=2)+'\n'
        loaded = native.parse_kmeans_model(formatted)
        canonical = native.parse_kmeans_model(native.encode_kmeans_model(loaded))
        self.assertEqual(canonical.centres, loaded.centres)
        self.assertEqual(canonical.centres_sha256, loaded.centres_sha256)
        self.assertNotEqual(canonical.source_sha256, loaded.source_sha256)
        self.assertEqual(canonical.source_sha256, sha256(encoded).hexdigest())

    def check_test_encoding_rejects_forged_models_before_creating_an_encoder(self):
        model = native.model_from_centres(((1.0,),), 1, 1)
        for invalid in (replace(model, centres=((2.0,),)), replace(model, source_bytes=0),
                        replace(model, centres_sha256='0'*64), None):
            with self.subTest(value=type(invalid).__name__), patch.object(native.json, 'JSONEncoder',
                    side_effect=AssertionError('encoded before validating model')):
                with self.assertRaises(native.KMeansModelError):
                    native.encode_kmeans_model(invalid)

    def check_test_actual_large_numeric_table_stops_at_the_incremental_encoding_cap(self):
        centres = tuple((0.12345678901234568,)*(31*31*3) for _ in range(256))
        model = native.KMeansModel(31, 3, centres, '0'*64, 1,
            native._centres_digest(31, 3, centres))
        native.validate_kmeans_model(model)
        with self.assertRaisesRegex(native.KMeansModelError, 'byte limit'):
            native.encode_kmeans_model(model)

    def check_test_constructor_normalizes_numbers_and_rejects_before_encoding(self):
        integer = native.model_from_centres([[0], [2]], 1, 1)
        floating = native.model_from_centres(((0.0,), (2.0,)), 1, 1)
        self.assertEqual(integer, floating)
        for value in (True, '0', float('nan'), float('inf'), -1, 256, 10 ** 1000):
            with self.subTest(value=type(value).__name__), patch.object(native.json, 'dumps',
                    side_effect=AssertionError('encoded before numeric validation')):
                with self.assertRaises(native.KMeansModelError):
                    native.model_from_centres([[value]], 1, 1)

    def check_test_wrong_schema_dtype_shapes_nonfinite_and_duplicate_fields_are_explicit(self):
        invalid = [self.payload(schema='other'), self.payload(preprocessing='sigma0'),
                   self.payload(dtype='float32'), self.payload(patch_size=True),
                   self.payload(channels=2), self.payload(centres=[]),
                   self.payload(centres=[[0, 1]]), self.payload(centres=[[True]]),
                   self.payload(centres=[['0']]), self.payload(centres=[[float('nan')]]),
                   self.payload(centres=[[1e309]]), self.payload(centres=[[-1]]),
                   self.payload(extra='unsupported'), '[]', '{}', '', b'\xff']
        raw = self.payload()
        invalid.append(raw[:-1] + ',"channels":1}')
        invalid.append(raw.replace('2.5', '1e309'))
        for source in invalid:
            with self.subTest(source=str(source)[:50]):
                with self.assertRaises(native.KMeansModelError):
                    native.parse_kmeans_model(source)

    def check_test_model_and_table_caps_preflight_and_loading_reads_cap_plus_one(self):
        source = self.payload()
        with patch.object(native, 'MAX_MODEL_BYTES', 10), patch.object(native.json, 'loads',
                side_effect=AssertionError('parsed before byte cap')):
            with self.assertRaises(native.KMeansModelError):
                native.parse_kmeans_model(source)
        with patch.object(native, 'MAX_CENTRE_VALUES', 1):
            with self.assertRaises(native.KMeansModelError):
                native.parse_kmeans_model(source)
        class Reader(io.BytesIO):
            count = None
            def read(self, count=-1):
                self.count = count
                return super().read(count)
        reader = Reader(b'x' * 40)
        with patch.object(native, 'MAX_MODEL_BYTES', 17), patch.object(native, 'Path') as path:
            path.return_value.open.return_value.__enter__.return_value = reader
            with self.assertRaises(native.KMeansModelError):
                native.load_kmeans_model('changed-after-selection')
        self.assertEqual(reader.count, 18)

    def check_test_actual_load_changed_source_missing_asset_and_forged_model_tables(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'artifacts') as directory:
            path = Path(directory) / 'centres.json'
            path.write_text(self.payload(), encoding='utf-8')
            original = native.load_kmeans_model(path)
            path.write_text(self.payload(centres=[[0], [3.5], [0]]), encoding='utf-8')
            changed = native.load_kmeans_model(path)
            self.assertNotEqual(original.source_sha256, changed.source_sha256)
            self.assertNotEqual(original.centres_sha256, changed.centres_sha256)
            with self.assertRaises(native.KMeansModelError):
                native.load_kmeans_model(Path(directory) / 'absent.json')
        for forged in (replace(original, centres=((0.0,), (99.0,), (0.0,))),
                       replace(original, centres=[[0.0], [2.5], [0.0]]),
                       replace(original, source_bytes=0), replace(original, source_sha256='invalid'),
                       replace(original, centres_sha256='invalid'), replace(original, patch_size=2),
                       replace(original, centres=((0,), (2.5,), (0.0,)))):
            with self.assertRaises(native.KMeansModelError):
                native.validate_kmeans_model(forged)

    def test_workflow_lifecycle(self):
        """2 related KMeansModelTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_frozen_rows_order_duplicates_and_raw_vs_numeric_identity)
        self.scenario(self.check_test_canonical_encoding_preserves_ordered_numeric_and_binary64_identity)

    def test_validation_and_recovery(self):
        """6 related KMeansModelTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_encoding_rejects_forged_models_before_creating_an_encoder)
        self.scenario(self.check_test_actual_large_numeric_table_stops_at_the_incremental_encoding_cap)
        self.scenario(self.check_test_constructor_normalizes_numbers_and_rejects_before_encoding)
        self.scenario(self.check_test_wrong_schema_dtype_shapes_nonfinite_and_duplicate_fields_are_explicit)
        self.scenario(self.check_test_model_and_table_caps_preflight_and_loading_reads_cap_plus_one)
        self.scenario(self.check_test_actual_load_changed_source_missing_asset_and_forged_model_tables)
