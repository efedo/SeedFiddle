"""Pinned correction execution, origin archives and actual batch companions."""
from copy import deepcopy
from dataclasses import replace
import gc
from hashlib import sha256
import inspect
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
import weakref

import cv2
import numpy as np
import torch

from seedvision.benchmarking import project_correction as adapter
from seedvision.benchmarking.native_correction_inputs import run_native_correction_files
from seedvision.benchmarking.native_correction_recipe import STAGE_ID, RECIPE_ID
from seedvision.benchmarking.native_recipe import NativeRecipeCancelled
from seedvision.export import native_results as exports
from seedvision.export import project_native_graphs as graphs
from seedvision.persistence import correction_inputs as assets
from seedvision.persistence.project_analysis import ProjectImageSpec, ProjectAnalysisStore
from seedvision.segmentation.baseline import PipelineAnalysisCache
from tests import test_project_correction_inputs as fixtures
from tests.test_native_result_export import _attachment, _output


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class ProjectCorrectionRunnerTests(unittest.TestCase):
    setUp = fixtures.ProjectCorrectionInputTests.setUp
    tearDown = fixtures.ProjectCorrectionInputTests.tearDown
    make_binding = fixtures.ProjectCorrectionInputTests.make_binding
    project = fixtures.ProjectCorrectionInputTests.project

    def run_binding(self, document, **kwargs):
        return adapter.run_project_correction(self.recipe, self.workspace, document.correction_bindings[0],
                                             document.images[0], self.source, **kwargs)

    def test_actual_transport_run_preserves_origins_and_verified_producer_identity(self):
        _, document = self.project()
        result, prepared = self.run_binding(document)
        binding = document.correction_bindings[0]
        self.assertEqual(result.input_origins.binding, binding)
        self.assertEqual(result.input_origins.project_source_path, document.images[0].source.path)
        self.assertEqual(result.source_sha256, document.images[0].source.sha256)
        self.assertEqual(result.coordinate_frame.shape, document.images[0].source_shape)
        for asset in binding.inputs:
            actual = result.input_sources[asset.role]
            self.assertEqual(actual['path'], str((self.workspace/asset.path).resolve()))
            self.assertNotEqual(actual['path'], asset.origin_path)
            self.assertEqual(actual['sha256'], asset.source_sha256)
            self.assertEqual(tuple(actual['frame']['shape']), asset.frame.shape)
            self.assertIn('path='+actual['path'], result.preparation[asset.role])
            self.assertEqual(prepared.tensors[asset.role].device.type, 'cuda')
        self.assertEqual({name: result.runtime[name] for name in assets.PROJECT_RUNTIME_KEYS}, adapter._implementation())
        self.assertEqual(result.input_origins, assets.CorrectionRunOrigins.from_dict(result.input_origins.to_dict()))

    def test_exact_image_locator_and_selection_reject_same_byte_substitution_before_execution(self):
        _, document = self.project()
        other = self.workspace/'same-bytes.png'
        shutil.copyfile(self.source, other)
        second = self.store.capture(analysis_settings=self.profile, native_recipes=(self.recipe,),
                                   images=(ProjectImageSpec(other, (17, 25)),)).images[0]
        for record, path in ((second, other), (document.images[0], other),
                             (replace(document.images[0], identifier='a'*64), self.source),
                             (replace(document.images[0], source_shape=(17, 24)), self.source)):
            with self.subTest(image_id=record.identifier), patch.object(adapter, 'run_native_correction_files') as run:
                with self.assertRaises(ValueError):
                    adapter.run_project_correction(self.recipe, self.workspace, document.correction_bindings[0],
                                                   record, path)
                run.assert_not_called()

    def test_target_may_differ_in_width_but_project_association_uses_source_frame(self):
        inputs = []
        for asset in self.binding.inputs:
            if asset.role.startswith('target_'):
                path = Path(asset.origin_path)
                image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                image = cv2.copyMakeBorder(image, 0, 0, 0, 2, cv2.BORDER_CONSTANT, value=0)
                self.assertTrue(cv2.imwrite(str(path), image))
                digest = sha256(path.read_bytes()).hexdigest()
                asset = replace(asset, source_sha256=digest, path='native-correction/'+digest+'.image',
                    source_bytes=path.stat().st_size, frame=replace(asset.frame, shape=(17, 27)))
            inputs.append(asset)
        binding = replace(self.binding, inputs=tuple(inputs))
        _, document = self.project((binding,))
        result, _ = self.run_binding(document)
        self.assertEqual(result.target_coordinate_frame.shape, (17, 27))
        self.assertEqual(result.coordinate_frame.shape, (17, 25))
        self.assertNotEqual(result.target_source_sha256, result.source_sha256)
        self.assertEqual(result.input_origins.binding.image_id, document.images[0].identifier)

    def test_cache_hit_attaches_current_origin_metadata_without_recomputing_outputs(self):
        _, document = self.project()
        cache = PipelineAnalysisCache()
        first, prepared = self.run_binding(document, cache=cache)
        binding = document.correction_bindings[0]
        changed = replace(binding, inputs=tuple(replace(asset, origin_path=asset.origin_path+'.archived',
                                                       decode=asset.decode+'; prior runtime') for asset in binding.inputs))
        second, retained = adapter.run_project_correction(self.recipe, self.workspace, changed,
            document.images[0], self.source, cache=cache, previous=prepared)
        self.assertIs(retained, prepared)
        self.assertTrue(second.node(STAGE_ID).reused)
        self.assertEqual(first.node(STAGE_ID).product_key, second.node(STAGE_ID).product_key)
        self.assertIs(first.node(STAGE_ID).outputs['corrected_image'], second.node(STAGE_ID).outputs['corrected_image'])
        self.assertEqual(second.input_origins.binding, changed)
        self.assertNotEqual(first.input_origins, second.input_origins)
        self.assertNotEqual(exports._identity(self.recipe, first), exports._identity(self.recipe, second))

    def test_each_corrupt_pin_fails_without_fallback_and_evicts_only_correction(self):
        _, document = self.project()
        cache = PipelineAnalysisCache()
        sentinel = object()
        cache.values['unrelated'] = sentinel
        for asset in document.correction_bindings[0].inputs:
            self.run_binding(document, cache=cache)
            path = self.workspace/asset.path
            raw = path.read_bytes()
            path.write_bytes(bytes([raw[0]^1])+raw[1:])
            with self.subTest(role=asset.role), self.assertRaises(ValueError):
                self.run_binding(document, cache=cache)
            self.assertIs(cache.values['unrelated'], sentinel)
            self.assertNotIn(adapter._CACHE_KEY, cache.values)
            self.assertTrue(Path(asset.origin_path).exists())
            path.write_bytes(raw)

    def test_final_callback_mutations_and_cancellation_withhold_results(self):
        _, document = self.project()
        source_raw = self.source.read_bytes()
        asset_path = self.workspace/document.correction_bindings[0].inputs[0].path
        asset_raw = asset_path.read_bytes()
        for mode in ('project-image', 'asset', 'input', 'output', 'matrix', 'implementation',
                     'metadata-helper', 'metadata-constant', 'cancel'):
            cache = PipelineAnalysisCache()
            sentinel = object()
            cache.values['unrelated'] = sentinel
            changed, active_patch = False, None
            def cancel():
                nonlocal changed, active_patch
                caller = inspect.currentframe().f_back.f_back
                if caller.f_code.co_name != 'run_project_correction' or 'signatures' not in caller.f_locals:
                    return False
                changed = True
                if mode == 'project-image':
                    self.source.write_bytes(bytes([source_raw[0]^1])+source_raw[1:])
                elif mode == 'asset':
                    asset_path.write_bytes(bytes([asset_raw[0]^1])+asset_raw[1:])
                elif mode == 'input':
                    caller.f_locals['prepared'].tensors['source_mask'][0, 0] ^= 1
                elif mode == 'output':
                    caller.f_locals['result'].node(STAGE_ID).outputs['corrected_image'][0, 0, 0] ^= 1
                elif mode == 'matrix':
                    caller.f_locals['result'].node(STAGE_ID).outputs['source_matrix'].shape = (12, 8)
                elif mode == 'implementation':
                    active_patch = patch.object(adapter, '_project_source', lambda *args: self.source)
                    active_patch.start()
                elif mode == 'metadata-helper':
                    active_patch = patch.object(adapter._project_metadata, '_exact_int', lambda value: True)
                    active_patch.start()
                elif mode == 'metadata-constant':
                    active_patch = patch.object(adapter._project_metadata, '_SHA256_LENGTH', 63)
                    active_patch.start()
                return mode == 'cancel'
            try:
                with self.subTest(mode=mode), self.assertRaises((ValueError, NativeRecipeCancelled)):
                    self.run_binding(document, cache=cache, cancel=cancel)
                self.assertTrue(changed)
                self.assertNotIn(adapter._CACHE_KEY, cache.values)
                self.assertIs(cache.values['unrelated'], sentinel)
            finally:
                if active_patch is not None:
                    active_patch.stop()
                self.source.write_bytes(source_raw)
                asset_path.write_bytes(asset_raw)

    def test_schema4_archive_keeps_actual_and_original_inputs_without_live_probes(self):
        _, document = self.project()
        result, _ = self.run_binding(document)
        path = exports.export_native_recipe_result(self.root/'archive', self.recipe, result)
        report = json.loads(path.read_bytes())
        self.assertEqual(report['version'], 5)
        self.assertEqual(report['correction_input_origins'], result.input_origins.to_dict())
        self.assertEqual(report['correction_inputs']['source_rgb']['path'],
                         str((self.workspace/document.correction_bindings[0].inputs[2].path).resolve()))
        for path in {*self.paths.values(), self.source,
                     *(self.workspace/asset.path for asset in document.correction_bindings[0].inputs)}:
            path.unlink()
        with patch.object(assets, 'copy_verified_input', side_effect=AssertionError('No input file reads')), \
                patch.object(assets, 'resolve_correction_asset_path', side_effect=AssertionError('No transport probes')):
            restored, arrays = exports.load_native_recipe_result(self.root/'archive')
        self.assertEqual(restored['correction_input_origins'], report['correction_input_origins'])
        expected = result.node(STAGE_ID).outputs['corrected_image'].cpu().numpy()
        np.testing.assert_array_equal(arrays[_output(restored, STAGE_ID, 'corrected_image')['name']], expected)

    def test_origin_tampering_is_rejected_before_array_loading(self):
        _, document = self.project()
        result, _ = self.run_binding(document)
        path = exports.export_native_recipe_result(self.root/'archive', self.recipe, result)
        original = path.read_bytes()
        mutations = ('image-id', 'locator', 'recipe', 'hash', 'bytes', 'frame', 'origin-path',
                     'transport-path', 'order', 'decode', 'missing', 'producer')
        for mode in mutations:
            report = json.loads(original)
            origin = report['correction_input_origins']
            binding = origin['binding']
            if mode == 'image-id':
                binding['image_id'] = '0'*64
            elif mode == 'locator':
                origin['project_source_path'] = 'different.png'
            elif mode == 'recipe':
                binding['recipe_sha256'] = '0'*64
            elif mode == 'hash':
                binding['inputs'][1]['source_sha256'] = '0'*64
            elif mode == 'bytes':
                binding['inputs'][1]['source_bytes'] += 1
            elif mode == 'frame':
                binding['inputs'][0]['frame']['shape'] = [17, 24]
            elif mode == 'origin-path':
                binding['inputs'][0]['origin_path'] += '.changed'
            elif mode == 'transport-path':
                binding['inputs'][0]['path'] = '../escape.image'
            elif mode == 'order':
                binding['inputs'].reverse()
            elif mode == 'decode':
                binding['inputs'][0]['decode'] += ' changed'
            elif mode == 'missing':
                report.pop('correction_input_origins')
            else:
                report['runtime'].pop(assets.PROJECT_RUNTIME_KEYS[0])
            report.pop('result_revision')
            report['result_revision'] = sha256(exports._json_bytes(report)).hexdigest()
            path.write_bytes(exports._json_bytes(report))
            with self.subTest(mode=mode), patch.object(exports, 'load_checkpoint') as load:
                with self.assertRaises(ValueError):
                    exports.load_native_recipe_result(path.parent)
                load.assert_not_called()

    def test_origin_and_project_producer_fields_must_be_present_together(self):
        _, document = self.project()
        project, _ = self.run_binding(document)
        local, _ = run_native_correction_files(self.recipe, self.paths)
        for result in (replace(project, input_origins=None), replace(local, input_origins=project.input_origins)):
            with patch.object(exports, 'publish_checkpoint') as publish:
                with self.assertRaises(ValueError):
                    exports.export_native_recipe_result(self.root/'bad', self.recipe, result)
                publish.assert_not_called()

    def test_legacy_schema3_local_correction_identity_and_arrays_remain_readable(self):
        from seedvision.benchmarking.checkpoints import load_checkpoint, publish_checkpoint
        local, _ = run_native_correction_files(self.recipe, self.paths)
        path = exports.export_native_recipe_result(self.root/'v4', self.recipe, local)
        report = json.loads(path.read_bytes())
        self.assertIsNone(report.pop('correction_input_origins'))
        report.pop('result_revision')
        groups = report.pop('groups')
        report['version'] = 3
        for node in report['nodes']:
            # Calculation metadata belongs to schema5, not the legacy wire format.
            node.pop('calculation')
        self.assertNotIn('correction_input_origins', report['checkpoint_identity']['image_semantics'])
        binding = sha256(exports._json_bytes(report)).hexdigest()
        destination = self.root/'legacy3'
        destination.mkdir()
        saved = []
        for index, group in enumerate(groups):
            checkpoint = load_checkpoint(path.parent/'checkpoints', group['checkpoint_sha256'])
            reference = publish_checkpoint(destination/'checkpoints', checkpoint.identity, checkpoint.arrays,
                execution=checkpoint.execution, measurement=checkpoint.measurement, raster_roles=checkpoint.raster_roles,
                metadata={'format': exports.FORMAT, 'version': 3, 'run_binding_sha256': binding,
                          'group_index': index, 'arrays': group['arrays']}, allow_cpu_transfer=True)
            saved.append({'checkpoint_sha256': reference.checkpoint_sha256, 'arrays': group['arrays']})
        report['groups'] = saved
        report['result_revision'] = sha256(exports._json_bytes(report)).hexdigest()
        (destination/'native-results.json').write_bytes(exports._json_bytes(report))
        restored, arrays = exports.load_native_recipe_result(destination)
        self.assertEqual(restored['version'], 3)
        self.assertNotIn('correction_input_origins', restored)
        self.assertEqual(restored['checkpoint_identity'], exports._identity(self.recipe, local).to_dict())
        np.testing.assert_array_equal(arrays[_output(restored, STAGE_ID, 'corrected_image')['name']],
                                      local.node(STAGE_ID).outputs['corrected_image'].cpu())

    def test_adjacent_correction_reuses_inputs_then_releases_them_before_generic_graph(self):
        from seedvision.benchmarking.native_extended_threshold_recipes import build_native_extended_threshold_recipes
        generic = build_native_extended_threshold_recipes()['plantcv_threshold_dual_channels']
        _, document = self.project()
        refs = []
        def cancel():
            caller = inspect.currentframe().f_back.f_back
            if caller.f_code.co_name == 'run_project_correction' and 'signatures' in caller.f_locals:
                refs.extend(weakref.ref(value) for value in caller.f_locals['prepared'].tensors.values())
            return False
        run_generic = graphs.run_native_recipe_path
        def run(*args, **kwargs):
            gc.collect()
            self.assertTrue(refs)
            self.assertTrue(all(reference() is None for reference in refs))
            return run_generic(*args, **kwargs)
        history = (_attachment(self.recipe),)
        before = history[0].to_dict()
        with patch.object(graphs, 'run_native_recipe_path', side_effect=run):
            path = graphs.export_project_native_graphs(history, self.source, self.root/'companions',
                expected_source_sha256=document.images[0].source.sha256, project_root=self.workspace,
                project_image_record=document.images[0], correction_bindings=document.correction_bindings,
                native_recipes=(self.recipe, generic), cancel=cancel)
        report = json.loads(path.read_bytes())
        self.assertEqual(report['version'], 3)
        self.assertEqual([row['status'] for row in report['graphs']], ['executed']*3)
        self.assertTrue(report['graphs'][1]['nodes'][0]['reused'])
        self.assertEqual(history[0].to_dict(), before)
        self.assertEqual(report['project_image_id'], document.images[0].identifier)

    def duplicate_project(self):
        other = self.workspace/'same-rgb.png'
        shutil.copyfile(self.source, other)
        images = self.store.capture(analysis_settings=self.profile,
            images=(ProjectImageSpec(self.source, (17, 25)), ProjectImageSpec(other, (17, 25))),
            native_recipes=(self.recipe,)).images
        paths = dict(self.paths)
        paths['source_rgb'] = other
        mask = cv2.imread(str(paths['source_mask']), cv2.IMREAD_UNCHANGED)
        mask[0, 0] = 10
        paths['source_mask'] = self.root/'distinct-mask.png'
        self.assertTrue(cv2.imwrite(str(paths['source_mask']), mask))
        second = self.make_binding(images[1].identifier, paths)
        master, document = self.project((self.binding, second), images)
        return master, document, (self.source, other)

    def test_duplicate_rgb_images_use_their_own_masks_and_missing_binding_is_unavailable(self):
        _, document, paths = self.duplicate_project()
        reports = []
        for index in (1, 0):
            path = graphs.export_project_native_graphs((), paths[index], self.root/f'row{index}',
                expected_source_sha256=document.images[index].source.sha256, project_root=self.workspace,
                project_image_record=document.images[index], correction_bindings=document.correction_bindings,
                native_recipes=(self.recipe,))
            row = json.loads(path.read_bytes())['graphs'][0]
            self.assertEqual(row['status'], 'executed')
            self.assertEqual(row['correction_binding']['image_id'], document.images[index].identifier)
            report, arrays = exports.load_native_recipe_result(path.parent/Path(row['report']).parent)
            self.assertEqual(report['correction_inputs']['source_mask']['sha256'],
                             document.correction_bindings[index].inputs[3].source_sha256)
            reports.append(arrays[_output(report, STAGE_ID, 'source_matrix')['name']])
        self.assertFalse(np.array_equal(*reports))
        path = graphs.export_project_native_graphs((), paths[0], self.root/'missing',
            expected_source_sha256=document.images[0].source.sha256, project_root=self.workspace,
            project_image_record=document.images[0], correction_bindings=(document.correction_bindings[1],),
            native_recipes=(self.recipe,))
        row = json.loads(path.read_bytes())['graphs'][0]
        self.assertEqual(row['status'], 'unavailable')
        self.assertIsNone(row['report'])
        self.assertEqual(list(path.parent.iterdir()), [path])

    def test_project_batch_keeps_baselines_and_other_images_when_one_correction_pin_is_bad(self):
        from seedvision.export.project_runner import run_project
        from tests.test_seed_crop_artifacts import _analysis
        master, document, paths = self.duplicate_project()
        # First image has no selection; second has an explicitly selected bad pin.
        document = replace(document, correction_bindings=(document.correction_bindings[1],))
        (self.workspace/'config').mkdir()
        shutil.copyfile(Path(__file__).parents[1]/'config'/'traits.json', self.workspace/'config'/'traits.json')
        mask_path = self.workspace/document.correction_bindings[0].inputs[3].path
        mask_raw = mask_path.read_bytes()
        for mode in ('corrupt-mask', 'wrong-source-byte-count'):
            selected = document
            if mode == 'corrupt-mask':
                mask_path.write_bytes(b'bad mask')
            else:
                binding = document.correction_bindings[0]
                source = binding.inputs[2]
                # Structurally valid metadata, with a distinct transport path so
                # deduplicated RGB roles do not create a metadata conflict.
                source = replace(source, source_bytes=source.source_bytes+1,
                                 path='native-correction/wrong-source-size.image')
                (self.workspace/source.path).write_bytes(self.source.read_bytes())
                binding = replace(binding, inputs=binding.inputs[:2]+(source,)+binding.inputs[3:])
                selected = replace(document, correction_bindings=(binding,))
            self.store.save(selected, master)
            baseline_refs = []
            def analyze(path, **kwargs):
                result = _analysis(path)
                baseline_refs.append(weakref.ref(result.calibration.gpu_corrected_bgr))
                return result
            with self.subTest(mode=mode), patch('seedvision.export.project_runner.analyze_path', side_effect=analyze):
                outputs = run_project(self.workspace, master, self.root/('batch-'+mode))
                self.assertEqual(len(outputs), 2)
                self.assertTrue(all((path/'results.json').is_file() for path in outputs))
                rows = [json.loads((path/'native_graphs'/'native-graphs.json').read_bytes())['graphs'][0] for path in outputs]
                self.assertEqual([row['status'] for row in rows], ['unavailable', 'failed'])
                self.assertTrue(all(row['report'] is None for row in rows))
                self.assertEqual(rows[1]['correction_binding'], selected.correction_bindings[0].to_dict())
                self.assertTrue(all(reference() is None for reference in baseline_refs))
            mask_path.write_bytes(mask_raw)

    def test_ledger_final_callback_cannot_publish_changed_transport_or_cancellation(self):
        _, document = self.project()
        asset = document.correction_bindings[0].inputs[0]
        path = self.workspace/asset.path
        raw = path.read_bytes()
        source_raw = self.source.read_bytes()
        for mode in ('change', 'grow-project', 'cancel'):
            active_patch = None
            def cancel():
                nonlocal active_patch
                if list(self.root.glob('.native-graphs-*/native-graphs.json')):
                    if mode == 'cancel':
                        return True
                    if mode == 'grow-project':
                        self.source.write_bytes(source_raw+b'excess bytes')
                        active_patch = patch.object(graphs, 'file_sha256', side_effect=AssertionError('Unbounded image read'))
                        active_patch.start()
                    else:
                        path.write_bytes(bytes([raw[0]^1])+raw[1:])
                return False
            try:
                with self.subTest(mode=mode), self.assertRaises((ValueError, NativeRecipeCancelled)):
                    graphs.export_project_native_graphs((), self.source, self.root/'withheld',
                        expected_source_sha256=document.images[0].source.sha256, project_root=self.workspace,
                        project_image_record=document.images[0], correction_bindings=document.correction_bindings,
                        native_recipes=(self.recipe,), cancel=cancel)
            finally:
                if active_patch is not None:
                    active_patch.stop()
                self.source.write_bytes(source_raw)
            self.assertFalse((self.root/'withheld').exists())
            self.assertFalse(list(self.root.glob('.native-graphs-*')))
            path.write_bytes(raw)

    def test_bound_project_image_guards_never_use_the_unbounded_reader(self):
        _, document = self.project()
        with patch.object(graphs, 'file_sha256', side_effect=AssertionError('Unbounded image read')):
            path = graphs.export_project_native_graphs((), self.source, self.root/'bounded',
                expected_source_sha256=document.images[0].source.sha256, project_root=self.workspace,
                project_image_record=document.images[0], correction_bindings=document.correction_bindings,
                native_recipes=(self.recipe,))
        self.assertEqual(json.loads(path.read_bytes())['graphs'][0]['status'], 'executed')
