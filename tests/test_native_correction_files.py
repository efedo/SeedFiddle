"""Four-file correction identity, bounded preparation and actual output archives."""

from tests.catalogue_fixtures import CatalogueTestCase

from tests.functional_units import functional_units
from dataclasses import replace
from hashlib import sha256
import io
import inspect
import json
from pathlib import Path
import tempfile
import threading
import unittest
import warnings
from unittest.mock import patch

import cv2
import numpy as np
import torch

from seedvision.benchmarking import native_correction_inputs as inputs
from seedvision.benchmarking.native_correction_recipe import (
    NativeCorrectionCancelled, NativeCorrectionError, RECIPE_ID, STAGE_ID,
    build_native_correction_recipe,
)
from seedvision.benchmarking.recipe_catalogue import build_recipe_catalogue
from seedvision.benchmarking.recipes import RecipeSpec, RecipeProvenanceMode
from seedvision.export import native_results as exports
from seedvision.segmentation.baseline import PipelineAnalysisCache
from tests.test_native_correction_dialog import _write_inputs, _fixture
from tests.test_native_result_export import _output


class CorrectionContractTests(CatalogueTestCase):
    def test_native_builder_preserves_historical_recipe_and_exact_four_ports(self):
        historical = build_recipe_catalogue().recipe(RECIPE_ID)
        before = historical.to_json()
        recipe = build_native_correction_recipe(historical)
        self.assertEqual(historical.to_json(), before)
        self.assertEqual(recipe.provenance.mode, RecipeProvenanceMode.NATIVE_COMPATIBLE)
        self.assertEqual(recipe.provenance.backend_id, 'seedvision.benchmarking.native_correction_recipe')
        self.assertNotEqual(recipe.identity, historical.identity)
        self.assertEqual(recipe.stages, historical.stages)
        self.assertEqual(recipe.dependencies, historical.dependencies)
        self.assertEqual(RecipeSpec.from_json(recipe.to_json()), recipe)
        self.assertIs(build_native_correction_recipe(recipe), recipe)

    def test_stream_read_bounds_hold_even_if_file_grows(self):
        class Stream(io.BytesIO):
            def read(self, size=-1):
                self.requests.append(size)
                return super().read(size)
        for mode in ('hash', 'snapshot'):
            stream = Stream(b'x'*100)
            stream.requests = []
            with self.subTest(mode=mode), patch.object(Path, 'open', return_value=stream), \
                    patch.object(inputs, 'MAX_INPUT_FILE_BYTES', 8):
                with self.assertRaises(NativeCorrectionError):
                    if mode == 'hash':
                        inputs._hash_file(Path('growing'))
                    else:
                        inputs._read_decode_bytes(Path('growing'), 8, sha256(b'x'*8).hexdigest(), None)
                self.assertEqual(stream.requests, [9])
                self.assertEqual(stream.tell() if not stream.closed else 9, 9)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
@functional_units
class CorrectionFileTests(CatalogueTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.paths = _write_inputs(self.root)
        self.recipe = build_native_correction_recipe()
        self.cache = PipelineAnalysisCache()

    def tearDown(self):
        self.temp.cleanup()

    def run_files(self, **options):
        return inputs.run_native_correction_files(self.recipe, self.paths, cache=self.cache, **options)


    def check_test_every_raw_file_change_recomputes_only_correction(self):
        result, prepared = self.run_files()
        self.cache.values['unrelated'] = sentinel = object()
        for name, path in self.paths.items():
            with self.subTest(role=name):
                before = result.node(STAGE_ID).product_key
                path.write_bytes(path.read_bytes()+b' ')
                result, current = self.run_files(previous=prepared)
                self.assertIsNot(current, prepared)
                self.assertFalse(result.node(STAGE_ID).reused)
                self.assertNotEqual(result.node(STAGE_ID).product_key, before)
                self.assertEqual(result.input_sources[name]['sha256'], sha256(path.read_bytes()).hexdigest())
                self.assertEqual(self.cache.last_computed_nodes, (STAGE_ID,))
                self.assertIs(self.cache.values['unrelated'], sentinel)
                prepared = current



    def check_test_decode_uses_selected_snapshot_despite_temporary_path_replacement(self):
        original = cv2.imdecode
        calls = 0
        def replace_during_decode(raw, flags):
            nonlocal calls
            calls += 1
            path = self.paths['target_rgb']
            saved = path.read_bytes()
            path.write_bytes(b'changed while decoding')
            try:
                return original(raw, flags)
            finally:
                path.write_bytes(saved)
        with patch.object(inputs.cv2, 'imdecode', side_effect=replace_during_decode):
            prepared = inputs.prepare_native_correction_inputs(self.paths)
        self.assertEqual(calls, 4)
        for name, expected in zip(self.paths, _fixture()):
            np.testing.assert_array_equal(prepared.tensors[name].cpu(), expected)



    def check_test_prepared_identity_and_decoder_versions_are_required_for_reuse(self):
        _, prepared = self.run_files()
        self.assertRegex(prepared.preparation_sha256, '^[0-9a-f]{64}$')
        for previous in (replace(prepared, preparation_sha256='0'*64),
                         replace(prepared, decoder_versions=dict(prepared.decoder_versions) | {'PySide6': 'old'})):
            with self.subTest(previous=previous.preparation_sha256):
                result, fresh = self.run_files(previous=previous)
                self.assertIsNot(fresh, previous)
                self.assertFalse(result.node(STAGE_ID).reused)
                self.assertEqual(result.runtime['correction_input_preparation'], fresh.preparation_sha256)
                self.assertEqual(result.runtime['PySide6'], fresh.decoder_versions['PySide6'])
                self.assertEqual(result.implementation_sha256, inputs.correction_file_implementation_sha256(result.runtime))




    def check_test_actual_four_input_export_roundtrip_matches_pinned_public_correction(self):
        from tests.test_native_correction_recipe import NativeCorrectionRecipeTests
        expected = NativeCorrectionRecipeTests().reference(_fixture())
        result, _ = self.run_files()
        output = exports.export_native_recipe_result(self.root/'run', self.recipe, result)
        report, arrays = exports.load_native_recipe_result(output.parent)
        self.assertEqual(report['version'], 5)
        self.assertEqual(report['source_sha256'], result.source_sha256)
        self.assertEqual(report['correction_inputs'], inputs.validate_correction_input_sources(result.input_sources))
        self.assertEqual(report['correction_implementation_sha256'], result.implementation_sha256)
        self.assertEqual(set(report['preparation']), set(self.paths))
        self.assertFalse(report['scientifically_validated'])
        for port in ('target_matrix', 'source_matrix', 'transformation_matrix'):
            array = arrays[_output(report, STAGE_ID, port)['name']]
            np.testing.assert_array_equal(array, result.node(STAGE_ID).outputs[port])
            np.testing.assert_allclose(array, expected[port], rtol=1e-8, atol=1e-8)
        corrected = arrays[_output(report, STAGE_ID, 'corrected_image')['name']]
        difference = np.abs(corrected.astype(np.int16)-expected['corrected_rgb'].astype(np.int16))
        self.assertLessEqual(int(difference.max()), 1)
        # A durable output archive is independent of the continued availability of input paths.
        for path in self.paths.values():
            path.unlink()
        restored, _ = exports.load_native_recipe_result(output.parent)
        self.assertEqual(restored['correction_inputs'], report['correction_inputs'])




    def check_test_report_identity_tampering_is_rejected_before_any_array_load(self):
        result, _ = self.run_files()
        path = exports.export_native_recipe_result(self.root/'run', self.recipe, result)
        original = path.read_bytes()
        for mutation in ('hash', 'mask-frame', 'implementation', 'preparation', 'missing'):
            report = json.loads(original)
            if mutation == 'hash':
                report['correction_inputs']['source_mask']['sha256'] = '0'*64
            elif mutation == 'mask-frame':
                report['correction_inputs']['target_mask']['frame']['frame_id'] = 'source.decoded'
            elif mutation == 'implementation':
                report['correction_implementation_sha256'] = '0'*64
            elif mutation == 'preparation':
                report['preparation'] = {'gray': 'fake', 'rgb': 'fake'}
            else:
                report.pop('correction_inputs')
            report.pop('result_revision')
            report['result_revision'] = sha256(exports._json_bytes(report)).hexdigest()
            path.write_bytes(exports._json_bytes(report))
            with self.subTest(mutation=mutation), patch.object(exports, 'load_checkpoint') as load:
                with self.assertRaises(exports.NativeResultExportError):
                    exports.load_native_recipe_result(path.parent)
                load.assert_not_called()



    def test_workflow_lifecycle(self):
        """4 related CorrectionFileTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_every_raw_file_change_recomputes_only_correction)
        self.scenario(self.check_test_decode_uses_selected_snapshot_despite_temporary_path_replacement)
        self.scenario(self.check_test_prepared_identity_and_decoder_versions_are_required_for_reuse)
        self.scenario(self.check_test_actual_four_input_export_roundtrip_matches_pinned_public_correction)

    def test_validation_and_recovery(self):
        """1 related CorrectionFileTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_report_identity_tampering_is_rejected_before_any_array_load)


if __name__ == '__main__':
    unittest.main()
