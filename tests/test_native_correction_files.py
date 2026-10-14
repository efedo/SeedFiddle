"""Four-file correction identity, bounded preparation and actual output archives."""
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


class CorrectionContractTests(unittest.TestCase):
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
class CorrectionFileTests(unittest.TestCase):
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

    def test_four_records_are_complete_deeply_immutable_and_cached_on_cuda(self):
        first, prepared = self.run_files()
        self.assertEqual(set(first.input_sources), set(self.paths))
        for name, path in self.paths.items():
            row = first.input_sources[name]
            self.assertEqual(row['path'], str(path.resolve()))
            self.assertEqual(row['sha256'], sha256(path.read_bytes()).hexdigest())
            self.assertEqual(row['bytes'], path.stat().st_size)
            self.assertEqual(row['decode'], first.preparation[name])
            self.assertEqual(tuple(row['frame']['shape']), prepared.coordinate_frames[name].shape)
            with self.assertRaises(TypeError):
                row['frame']['shape'] = (1, 1)
            self.assertIsInstance(row['frame']['shape'], tuple)
            self.assertEqual(prepared.tensors[name].device.type, 'cuda')
        second, retained = self.run_files(previous=prepared)
        self.assertIs(retained, prepared)
        self.assertTrue(second.node(STAGE_ID).reused)
        self.assertIs(first.node(STAGE_ID).outputs['corrected_image'],
                      second.node(STAGE_ID).outputs['corrected_image'])
        self.assertEqual(first.input_sources, second.input_sources)

    def test_every_raw_file_change_recomputes_only_correction(self):
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

    def test_reuse_checks_lowered_limits_before_decode_or_transfer(self):
        _, prepared = self.run_files()
        with patch.object(inputs.cv2, 'imdecode') as decode, patch.object(inputs.torch, 'as_tensor') as transfer:
            for kwargs in ({'max_combined_pixels': 849}, {'max_prepared_cuda_bytes': prepared.cuda_bytes-1}):
                with self.subTest(kwargs=kwargs), self.assertRaisesRegex(NativeCorrectionError, 'Cached correction inputs'):
                    inputs.prepare_native_correction_inputs(self.paths, previous=prepared, **kwargs)
            reused = inputs.prepare_native_correction_inputs(self.paths, previous=prepared,
                max_combined_pixels=850, max_prepared_cuda_bytes=prepared.cuda_bytes)
            self.assertIs(reused, prepared)
            decode.assert_not_called()
            transfer.assert_not_called()
        for kwargs in ({'max_combined_pixels': True}, {'max_combined_pixels': 64_000_001},
                       {'max_prepared_cuda_bytes': 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(NativeCorrectionError):
                inputs.prepare_native_correction_inputs(self.paths, previous=prepared, **kwargs)

    def test_byte_and_header_limits_precede_decode_and_cuda_transfer(self):
        for bound, maximum in (('MAX_INPUT_FILE_BYTES', 1),):
            with self.subTest(bound=bound), patch.object(inputs, bound, maximum), \
                    patch.object(inputs.cv2, 'imdecode') as decode, patch.object(inputs.torch, 'as_tensor') as transfer:
                with self.assertRaises(NativeCorrectionError):
                    inputs.prepare_native_correction_inputs(self.paths)
                decode.assert_not_called()
                transfer.assert_not_called()
        # Each header fits the requested pixel cap; the pair of oversized masks
        # cannot allocate a fourth decoded array beyond the aggregate CPU cap.
        for role in ('target_mask', 'source_mask'):
            self.assertTrue(cv2.imwrite(str(self.paths[role]), np.zeros((25, 34), np.uint8)))
        with patch.object(inputs.cv2, 'imdecode', wraps=cv2.imdecode) as decode, \
                patch.object(inputs.torch, 'as_tensor') as transfer:
            with self.assertRaisesRegex(NativeCorrectionError, 'aggregate decoded CPU-byte bound'):
                inputs.prepare_native_correction_inputs(self.paths, max_combined_pixels=850)
            self.assertEqual(decode.call_count, 3)
            transfer.assert_not_called()
        _write_inputs(self.root)
        self.assertTrue(cv2.imwrite(str(self.paths['source_mask']), np.zeros((17, 25), np.uint16)))
        with patch.object(inputs.cv2, 'imdecode', wraps=cv2.imdecode) as decode, \
                patch.object(inputs.torch, 'as_tensor') as transfer:
            with self.assertRaisesRegex(NativeCorrectionError, 'eight-bit raster header'):
                inputs.prepare_native_correction_inputs(self.paths)
            self.assertEqual(decode.call_count, 3)
            transfer.assert_not_called()

    def test_decode_uses_selected_snapshot_despite_temporary_path_replacement(self):
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

    def test_mutation_between_hash_and_snapshot_fails_before_decode(self):
        original, calls = Path.open, 0
        def changed(path, *args, **kwargs):
            nonlocal calls
            if path in self.paths.values() and args and args[0] == 'rb':
                calls += 1
                if calls == 5:
                    path.write_bytes(path.read_bytes()+b' ')
            return original(path, *args, **kwargs)
        with patch.object(Path, 'open', changed), \
                patch.object(inputs.cv2, 'imdecode') as decode:
            with self.assertRaisesRegex(NativeCorrectionError, 'changed while OpenCV'):
                self.run_files()
            decode.assert_not_called()

    def test_after_run_changes_and_cancellation_clear_correction_cache(self):
        for role in (*self.paths, 'tensor', 'cancel'):
            prepared = inputs.prepare_native_correction_inputs(self.paths)
            changed = False
            self.cache.values['unrelated'] = sentinel = object()
            self.cache.last_computed_nodes = ()
            def change():
                nonlocal changed
                if not changed and self.cache.last_computed_nodes == (STAGE_ID,):
                    changed = True
                    if role in self.paths:
                        path = self.paths[role]
                        path.write_bytes(path.read_bytes()+b' ')
                    elif role == 'tensor':
                        prepared.tensors['target_rgb'][0, 0, 0] ^= 1
                return changed and role == 'cancel'
            with self.subTest(role=role):
                with self.assertRaises((NativeCorrectionError, NativeCorrectionCancelled)):
                    self.run_files(previous=prepared, cancel=change)
            self.assertTrue(changed)
            self.assertNotIn(inputs._CORRECTION_CACHE_KEY, self.cache.values)
            self.assertIs(self.cache.values['unrelated'], sentinel)

    def test_prepared_identity_and_decoder_versions_are_required_for_reuse(self):
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

    def test_changed_loaded_preparation_helper_cannot_reuse_previous_bundle(self):
        _, prepared = self.run_files()
        with patch.object(inputs, '_decode_header'), patch.object(inputs.cv2, 'imdecode') as decode, \
                patch.object(inputs.torch, 'as_tensor') as transfer:
            with self.assertRaisesRegex(NativeCorrectionError, 'verified source'):
                self.run_files(previous=prepared)
            decode.assert_not_called()
            transfer.assert_not_called()
        self.assertNotIn(inputs._CORRECTION_CACHE_KEY, self.cache.values)

    def test_final_cancel_callback_mutation_is_checked_before_return(self):
        for mode in ('file', 'tensor', 'output', 'matrix'):
            prepared = inputs.prepare_native_correction_inputs(self.paths)
            changed = False
            def mutate_at_final_poll():
                nonlocal changed
                caller = inspect.currentframe().f_back.f_back
                if caller.f_code.co_name == 'run_native_correction_files' and 'sources' in caller.f_locals and not changed:
                    changed = True
                    if mode == 'file':
                        path = self.paths['target_mask']
                        path.write_bytes(path.read_bytes()+b' ')
                    elif mode == 'tensor':
                        prepared.tensors['source_rgb'][0, 0, 0] ^= 1
                    elif mode == 'output':
                        self.cache.values[inputs._CORRECTION_CACHE_KEY].node.outputs['corrected_image'][0, 0, 0] ^= 1
                    else:
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore', DeprecationWarning)
                            self.cache.values[inputs._CORRECTION_CACHE_KEY].node.outputs['source_matrix'].shape = (4, 24)
                return False
            with self.subTest(mode=mode), self.assertRaisesRegex(NativeCorrectionError, 'changed'):
                self.run_files(previous=prepared, cancel=mutate_at_final_poll)
            self.assertTrue(changed)
            self.assertNotIn(inputs._CORRECTION_CACHE_KEY, self.cache.values)

    def test_cancellation_before_decode_and_before_upload_returns_no_bundle(self):
        with patch.object(inputs.cv2, 'imdecode') as decode:
            with self.assertRaises(NativeCorrectionCancelled):
                self.run_files(cancel=lambda: True)
            decode.assert_not_called()
        stop = threading.Event()
        original = cv2.imdecode
        calls = 0
        def stop_after_last(raw, flags):
            nonlocal calls
            calls += 1
            result = original(raw, flags)
            if calls == 4:
                stop.set()
            return result
        with patch.object(inputs.cv2, 'imdecode', side_effect=stop_after_last), \
                patch.object(inputs.torch, 'as_tensor') as transfer:
            with self.assertRaises(NativeCorrectionCancelled):
                self.run_files(cancel=stop.is_set)
            transfer.assert_not_called()

    def test_actual_four_input_export_roundtrip_matches_pinned_public_correction(self):
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

    def test_invalid_source_records_never_publish_or_transfer(self):
        result, _ = self.run_files()
        sources = inputs.validate_correction_input_sources(result.input_sources)
        variants = []
        for role in self.paths:
            for field, value in (('sha256', 'bad'), ('bytes', True), ('decode', ''), ('path', '')):
                changed = json.loads(json.dumps(sources))
                changed[role][field] = value
                variants.append(changed)
        variants.extend(({key: value for key, value in sources.items() if key != 'target_mask'},
                         sources | {'unknown': sources['target_rgb']}))
        for path in ('relative.png', r'C:\folder\..\image.png', '/folder/../image.png',
                     str(self.paths['target_rgb'])+' '):
            changed = json.loads(json.dumps(sources))
            changed['target_rgb']['path'] = path
            variants.append(changed)
        changed = json.loads(json.dumps(sources))
        changed['target_mask']['frame']['shape'] = [18, 25]
        variants.append(changed)
        for index, changed in enumerate(variants):
            with self.subTest(index=index), patch.object(exports, 'publish_checkpoint') as publish:
                with self.assertRaises(ValueError):
                    exports.export_native_recipe_result(self.root/f'bad-{index}', self.recipe,
                                                        replace(result, input_sources=changed))
                publish.assert_not_called()
        with self.assertRaises(ValueError):
            exports.export_native_recipe_result(self.root/'wrong-source', self.recipe,
                                                replace(result, source_sha256='0'*64))

    def test_foreign_absolute_source_records_load_without_path_probing(self):
        result, _ = self.run_files()
        sources = inputs.validate_correction_input_sources(result.input_sources)
        for name, row in sources.items():
            row['path'] = f'/foreign-host/correction/{name}.png'
        with patch.object(Path, 'resolve') as resolve:
            checked = inputs.validate_correction_input_sources(sources)
            resolve.assert_not_called()
        changed = replace(result, input_sources=checked)
        path = exports.export_native_recipe_result(self.root/'foreign', self.recipe, changed)
        for origin in self.paths.values():
            origin.unlink()
        report, _ = exports.load_native_recipe_result(path.parent)
        self.assertEqual(report['correction_inputs'], sources)

    def test_matrix_domain_and_immutable_backing_precede_checkpoint_transfer(self):
        result, _ = self.run_files()
        node = result.node(STAGE_ID)
        def immutable(value):
            return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
        for port, value in (('source_matrix', np.zeros((24, 4), np.float64)),
                ('source_matrix', immutable(np.zeros((256, 4), np.float64))),
                ('source_matrix', immutable(np.zeros((24, 4), np.float32))),
                ('source_matrix', immutable(np.full((24, 4), np.nan))),
                ('transformation_matrix', immutable(np.zeros((8, 9), np.float64))),
                ('corrected_image', np.zeros((17, 25, 3), np.uint8))):
            changed = replace(result, nodes=(replace(node, outputs=dict(node.outputs) | {port: value}),))
            with self.subTest(port=port, shape=value.shape), patch.object(exports, 'publish_checkpoint') as publish:
                with self.assertRaises(ValueError):
                    exports.export_native_recipe_result(self.root/'bad', self.recipe, changed)
                publish.assert_not_called()

    def test_report_identity_tampering_is_rejected_before_any_array_load(self):
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

    def test_verify_only_archive_still_rejects_rehashed_nonfinite_auxiliary_fit_matrix(self):
        from seedvision.benchmarking import checkpoints
        result, _ = self.run_files()
        path = exports.export_native_recipe_result(self.root/'nonfinite',self.recipe,result)
        report, arrays = exports.load_native_recipe_result(path.parent)
        name = _output(report,STAGE_ID,'source_matrix')['name']
        group = next(item for item in report['groups'] if name in item['arrays'])
        directory = path.parent/'checkpoints'/f'{group["checkpoint_sha256"]}.checkpoint'
        values = {item:arrays[item].copy() for item in group['arrays']}
        values[name][0,0] = np.nan
        with (directory/'arrays.npz').open('wb') as stream: np.savez(stream,**values)
        manifest_path = directory/'manifest.json'; manifest = json.loads(manifest_path.read_text())
        next(item for item in manifest['arrays'] if item['name']==name)['data_sha256'] = checkpoints._array_data_sha256(values[name])
        manifest['checkpoint_sha256'] = checkpoints._content_sha256(manifest)
        manifest_path.write_bytes(exports._json_bytes(manifest))
        group['checkpoint_sha256'] = manifest['checkpoint_sha256']
        directory.rename(directory.parent/f'{group["checkpoint_sha256"]}.checkpoint')
        report.pop('result_revision'); report['result_revision'] = sha256(exports._json_bytes(report)).hexdigest()
        path.write_bytes(exports._json_bytes(report))
        with self.assertRaisesRegex(exports.NativeResultExportError,'fit matrices must be finite'):
            exports.load_native_recipe_result(path.parent,array_names=())

    def test_export_cancellation_and_output_mutation_leave_no_final_artifact(self):
        result, _ = self.run_files()
        stop = threading.Event()
        publish = exports.publish_checkpoint
        for mode in ('cancel', 'mutate'):
            stop.clear()
            def interrupted(*args, **kwargs):
                ref = publish(*args, **kwargs)
                if mode == 'cancel':
                    stop.set()
                else:
                    result.node(STAGE_ID).outputs['corrected_image'][0, 0, 0] ^= 1
                return ref
            with self.subTest(mode=mode), patch.object(exports, 'publish_checkpoint', side_effect=interrupted):
                with self.assertRaises((exports.NativeResultExportError, exports.NativeRecipeCancelled)):
                    exports.export_native_recipe_result(self.root/mode, self.recipe, result, cancel=stop.is_set)
            self.assertFalse((self.root/mode).exists())
            self.assertFalse(list(self.root.glob('.native-result-*')))


if __name__ == '__main__':
    unittest.main()
