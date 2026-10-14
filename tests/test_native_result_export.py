"""Actual native product persistence and saved-project batch bindings."""
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time
import unittest
import weakref
from unittest.mock import patch

import cv2
import numpy as np
import torch

from seedvision.benchmarking import native_kmeans_recipe as kmeans
from seedvision.benchmarking.native_recipe import (
    NativeRecipeCancelled, build_native_seed_recipe, run_native_recipe_path,
)
from seedvision.benchmarking.checkpoints import CheckpointLimits
from seedvision.benchmarking.recipes import initial_stage_run
from seedvision.export import native_results as exports
from seedvision.export.project_native_graphs import export_project_native_graphs
from seedvision.model_io import plantcv_kmeans_model as models
from seedvision.persistence.reference_graphs import ReferenceGraphAttachment
from seedvision.segmentation.baseline import PipelineAnalysisCache
from tests.test_native_kmeans_recipe import _parameter
from tests.test_native_recipe import fixture, selected_recipe


def _attachment(recipe):
    return ReferenceGraphAttachment(recipe, tuple(initial_stage_run(recipe, stage.stage_id)
                                                for stage in recipe.stages))


def _output(report, stage, port):
    return dict(next(node for node in report['nodes'] if node['stage_id'] == stage)['outputs'])[port]


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class NativeResultExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root/'seeds.png'
        image = np.zeros((13, 17, 3), np.uint8)
        image[:, 8:] = 180
        self.assertTrue(cv2.imwrite(str(self.source), image))
        self.model_path = self.root/'centres.json'
        self.recipe = self.make_recipe(2)

    def tearDown(self):
        self.temp.cleanup()

    def make_recipe(self, count):
        self.model_path.write_text(json.dumps({'schema': models.SCHEMA,
            'preprocessing': models.PREPROCESSING, 'dtype': 'float64',
            'patch_size': 1, 'channels': 1,
            'centres': [[float(value)] for value in (range(256) if count == 256 else (0, 180))]}),
            encoding='utf-8')
        return _parameter(kmeans.build_native_kmeans_recipe(), kmeans.MODEL_STAGE_ID,
                          'model_source', kmeans.model_source_for_path(self.model_path))

    def run_recipe(self, recipe=None):
        return run_native_recipe_path(recipe or self.recipe, self.source)

    def test_actual_model_labels_masks_and_source_bytes_roundtrip(self):
        result = self.run_recipe()
        before = tuple((node.stage_id, node.product_key, node.reused) for node in result.nodes)
        report_path = exports.export_native_recipe_result(self.root/'run', self.recipe, result)
        report, arrays = exports.load_native_recipe_result(report_path.parent)
        self.assertEqual(report['source_sha256'], sha256(self.source.read_bytes()).hexdigest())
        self.assertEqual(report['recipe_identity'], result.recipe_identity.to_dict())
        self.assertEqual(report['runtime'], dict(result.runtime))
        self.assertFalse(report['scientifically_validated'])
        labels = _output(report, kmeans.PREDICT_STAGE_ID, 'labels')
        np.testing.assert_array_equal(arrays[labels['name']], result.node(kmeans.PREDICT_STAGE_ID).outputs['labels'].cpu())
        masks = _output(report, kmeans.MASK_STAGE_ID, 'class_masks')
        self.assertEqual([name for name, _ in masks['items']], ['0', '1'])
        for name, binding in masks['items']:
            np.testing.assert_array_equal(arrays[binding['name']],
                result.node(kmeans.MASK_STAGE_ID).outputs['class_masks'][name].cpu())
        model = _output(report, kmeans.MODEL_STAGE_ID, 'model')
        self.assertEqual(arrays[model['bytes']['name']].tobytes(), self.model_path.read_bytes())
        self.assertEqual(before, tuple((node.stage_id, node.product_key, node.reused) for node in result.nodes))
        self.assertTrue(all(value.device.type == 'cuda' for node in result.nodes
                            for value in node.outputs.values() if torch.is_tensor(value)))

    def test_256_ordered_class_masks_cross_checkpoint_groups(self):
        recipe = self.make_recipe(256)
        result = self.run_recipe(recipe)
        exports.export_native_recipe_result(self.root/'many', recipe, result)
        report, arrays = exports.load_native_recipe_result(self.root/'many')
        self.assertGreater(len(report['groups']), 1)
        masks = _output(report, kmeans.MASK_STAGE_ID, 'class_masks')['items']
        self.assertEqual([name for name, _ in masks], list(map(str, range(256))))
        for name, binding in masks:
            np.testing.assert_array_equal(arrays[binding['name']],
                result.node(kmeans.MASK_STAGE_ID).outputs['class_masks'][name].cpu())

    def test_inventory_and_selected_load_keep_full_verification_and_subset_semantics(self):
        result = self.run_recipe()
        exports.export_native_recipe_result(self.root/'selected', self.recipe, result)
        with patch.object(exports, 'load_checkpoint') as load:
            inventory, specs = exports.read_native_recipe_inventory(self.root/'selected')
            load.assert_not_called()
        binding = _output(inventory, kmeans.PREDICT_STAGE_ID, 'labels')
        name = binding['name']
        self.assertIn(name, specs)
        with patch('seedvision.benchmarking.checkpoints.np.load', side_effect=AssertionError('full decode')):
            report, arrays = exports.load_native_recipe_result(self.root/'selected',
                array_names=(name,), expected_revision=inventory['result_revision'])
            verified, empty = exports.load_native_recipe_result(self.root/'selected', array_names=())
        self.assertEqual(report, inventory)
        self.assertEqual(verified, inventory)
        self.assertEqual(set(arrays), {name})
        self.assertEqual(empty, {})
        np.testing.assert_array_equal(arrays[name], result.node(kmeans.PREDICT_STAGE_ID).outputs['labels'].cpu())
        with self.assertRaises(ValueError): arrays[name].setflags(write=True)

    def test_selected_load_rejects_revision_names_and_cancellation_before_arrays(self):
        result = self.run_recipe()
        exports.export_native_recipe_result(self.root/'selected-invalid', self.recipe, result)
        with patch.object(exports, 'load_checkpoint') as load:
            with self.assertRaises(exports.NativeResultExportError):
                exports.load_native_recipe_result(self.root/'selected-invalid', array_names=(), expected_revision='0'*64)
            for names in (('missing',), ['array_0000'], ('array_0000','array_0000')):
                with self.assertRaises(exports.NativeResultExportError):
                    exports.load_native_recipe_result(self.root/'selected-invalid', array_names=names)
            with self.assertRaises(NativeRecipeCancelled):
                exports.load_native_recipe_result(self.root/'selected-invalid', array_names=(), cancel=lambda:True)
            load.assert_not_called()

    def test_changed_recipe_or_model_never_publishes_old_result(self):
        result = self.run_recipe()
        changed = _parameter(self.recipe, kmeans.MASK_STAGE_ID, 'cat_list', [1])
        with self.assertRaises(exports.NativeResultExportError):
            exports.export_native_recipe_result(self.root/'stale', changed, result)
        self.assertFalse((self.root/'stale').exists())
        self.model_path.write_text(self.model_path.read_text()+' ')
        with self.assertRaises(ValueError):
            exports.export_native_recipe_result(self.root/'changed-model', self.recipe, result)
        self.assertFalse((self.root/'changed-model').exists())

    def test_all_resource_preflights_precede_checkpoint_transfer(self):
        result = self.run_recipe()
        limits = CheckpointLimits(max_array_bytes=1, max_total_array_bytes=1)
        with patch.object(exports, 'publish_checkpoint') as publish:
            with self.assertRaises(ValueError):
                exports.export_native_recipe_result(self.root/'too-big', self.recipe, result, limits=limits)
            publish.assert_not_called()
        self.assertFalse((self.root/'too-big').exists())
        with patch.object(exports, 'MAX_REPORT_BYTES', 32), patch.object(exports, 'publish_checkpoint') as publish:
            with self.assertRaises(exports.NativeResultExportError):
                exports.export_native_recipe_result(self.root/'too-much-metadata', self.recipe, result)
            publish.assert_not_called()

    def test_cancel_failure_and_tensor_mutation_remove_staging(self):
        result = self.run_recipe()
        cancelled = threading.Event()
        publish = exports.publish_checkpoint
        def stop_after_first(*args, **kwargs):
            reference = publish(*args, **kwargs)
            cancelled.set()
            return reference
        with patch.object(exports, 'publish_checkpoint', side_effect=stop_after_first):
            with self.assertRaises(NativeRecipeCancelled):
                exports.export_native_recipe_result(self.root/'cancelled', self.recipe, result,
                                                     cancel=cancelled.is_set)
        self.assertFalse((self.root/'cancelled').exists())
        self.assertFalse(list(self.root.glob('.native-result-*')))
        with patch.object(exports, 'publish_checkpoint', side_effect=OSError('write failed')):
            with self.assertRaises(OSError):
                exports.export_native_recipe_result(self.root/'failed', self.recipe, result)
        self.assertFalse((self.root/'failed').exists())
        def mutate(*args, **kwargs):
            reference = publish(*args, **kwargs)
            result.node(kmeans.PREDICT_STAGE_ID).outputs['labels'].add_(1)
            return reference
        with patch.object(exports, 'publish_checkpoint', side_effect=mutate):
            with self.assertRaises(exports.NativeResultExportError):
                exports.export_native_recipe_result(self.root/'mutated', self.recipe, result)
        self.assertFalse((self.root/'mutated').exists())
        self.assertFalse(list(self.root.glob('.native-result-*')))

    def test_tampered_report_or_array_archive_is_rejected(self):
        result = self.run_recipe()
        path = exports.export_native_recipe_result(self.root/'run', self.recipe, result)
        original = path.read_bytes()
        report = json.loads(original)
        report['status'] = 'partial'
        path.write_text(json.dumps(report))
        with self.assertRaises(exports.NativeResultExportError):
            exports.load_native_recipe_result(path.parent)
        path.write_bytes(original)
        archive = next((path.parent/'checkpoints').glob('*/arrays.npz'))
        with archive.open('r+b') as stream:
            stream.seek(-20, 2)
            stream.write(b'corrupt result bytes')
        with self.assertRaises(ValueError):
            exports.load_native_recipe_result(path.parent)

    def test_recomputed_report_hash_cannot_change_semantic_claims_or_bindings(self):
        result = self.run_recipe()
        path = exports.export_native_recipe_result(self.root/'run', self.recipe, result)
        original = path.read_bytes()
        def change_model_identity(report):
            model_node = next(node for node in report['nodes'] if node['stage_id'] == kmeans.PREDICT_STAGE_ID)
            dict(model_node['outputs'])['model_identity']['items'][0][1] = 'different model'
        def put_asset_on_label_port(report):
            model = _output(report, kmeans.MODEL_STAGE_ID, 'model')
            node = next(node for node in report['nodes'] if node['stage_id'] == kmeans.PREDICT_STAGE_ID)
            next(item for item in node['outputs'] if item[0] == 'labels')[1] = model
        changes = [lambda report: report.update(scientifically_validated=True),
            lambda report: report.update(status='partial'),
            lambda report: report['runtime'].update(torch='different'),
            lambda report: report.update(runtime=list(report['runtime'].items())),
            lambda report: report['recipe_identity'].update(backend_version='different'),
            lambda report: report['preparation'].update(gray='different'),
            lambda report: report['nodes'][0].update(status='failed'),
            lambda report: report['nodes'][0]['outputs'][0][1].update(name='array_0999'),
            lambda report: report.update(correction_input_origins={}),
            change_model_identity, put_asset_on_label_port]
        for change in changes:
            report = json.loads(original)
            change(report)
            report.pop('result_revision')
            report['result_revision'] = sha256(exports._json_bytes(report)).hexdigest()
            path.write_bytes(exports._json_bytes(report))
            with patch.object(exports, 'load_checkpoint') as load:
                with self.assertRaises(ValueError):
                    exports.load_native_recipe_result(path.parent)
                load.assert_not_called()

    def test_final_cancel_poll_mutation_is_withheld_before_publication(self):
        result = self.run_recipe()
        def mutate_on_final_poll():
            if list(self.root.glob('.native-result-*/native-results.json')):
                result.node(kmeans.PREDICT_STAGE_ID).outputs['labels'].add_(1)
            return False
        with self.assertRaises(exports.NativeResultExportError):
            exports.export_native_recipe_result(self.root/'mutated', self.recipe, result,
                                                 cancel=mutate_on_final_poll)
        self.assertFalse((self.root/'mutated').exists())
        self.assertFalse(list(self.root.glob('.native-result-*')))
    def test_loader_aggregate_preflight_happens_before_any_array_load(self):
        result = self.run_recipe()
        exports.export_native_recipe_result(self.root/'run', self.recipe, result,
                                            limits=CheckpointLimits(max_arrays=1))
        with patch.object(exports, 'load_checkpoint') as load:
            with self.assertRaises(ValueError):
                exports.load_native_recipe_result(self.root/'run',
                    limits=CheckpointLimits(max_array_bytes=700, max_total_array_bytes=700))
            load.assert_not_called()

    def test_actual_seed_size_colour_histograms_and_partial_products(self):
        gray, _ = fixture()
        cv2.imwrite(str(self.source), np.repeat(gray[..., None], 3, axis=2))
        recipe = build_native_seed_recipe(selected_recipe(**{'analysis.color': {'colorspaces': 'hsv'}}))
        result = self.run_recipe(recipe)
        exports.export_native_recipe_result(self.root/'seed', recipe, result)
        report, arrays = exports.load_native_recipe_result(self.root/'seed')
        size = next(stage for stage in recipe.stages if stage.operation_id == 'analysis.size')
        colour = next(stage for stage in recipe.stages if stage.operation_id == 'analysis.color')
        self.assertEqual(_output(report, size.stage_id, 'observations')['kind'], 'mapping')
        self.assertEqual(_output(report, colour.stage_id, 'observations')['kind'], 'record')
        self.assertEqual(_output(report, colour.stage_id, 'histogram')['kind'], 'record')
        partial_node = replace(result.node(size.stage_id), status='partial',
            outputs={'observations': result.node(size.stage_id).outputs['observations']},
            reason='Unavailable outputs: diagnostic')
        partial = replace(result, status='partial', nodes=tuple(partial_node if node.stage_id == size.stage_id
                                                              else node for node in result.nodes))
        exports.export_native_recipe_result(self.root/'partial', recipe, partial)
        partial_report, _ = exports.load_native_recipe_result(self.root/'partial')
        self.assertEqual(partial_report['status'], 'partial')
        self.assertNotIn('diagnostic', dict(next(node for node in partial_report['nodes']
                                               if node['stage_id'] == size.stage_id)['outputs']))

    def test_actual_bayes_model_bytes_and_ordered_masks(self):
        from tests.test_plantcv_naive_bayes import _constant_pdf
        from tests.test_native_naive_bayes_recipe import _recipe_with_model
        model_path = self.root/'pdf.tsv'
        model_path.write_bytes(_constant_pdf({'first': 1., 'second': 2.}))
        recipe, _ = _recipe_with_model(model_path)
        result = self.run_recipe(recipe)
        exports.export_native_recipe_result(self.root/'bayes', recipe, result)
        report, arrays = exports.load_native_recipe_result(self.root/'bayes')
        model_stage = next(node.stage_id for node in result.nodes if 'model' in node.outputs)
        model = _output(report, model_stage, 'model')
        self.assertEqual(arrays[model['bytes']['name']].tobytes(), model_path.read_bytes())
        stage = next(node for node in result.nodes if 'class_masks' in node.outputs)
        masks = _output(report, stage.stage_id, 'class_masks')['items']
        self.assertEqual([name for name, _ in masks], list(stage.outputs['class_masks']))
        for name, binding in masks:
            np.testing.assert_array_equal(arrays[binding['name']], stage.outputs['class_masks'][name].cpu())

    def test_project_graphs_execute_exact_recipe_preserve_history_and_report_failures(self):
        attachments = (_attachment(self.recipe), _attachment(selected_recipe()))
        historical = tuple(item.to_dict() for item in attachments)
        cache = PipelineAnalysisCache()
        source_sha = sha256(self.source.read_bytes()).hexdigest()
        with patch('seedvision.export.project_native_graphs.run_native_recipe_path',
                   wraps=run_native_recipe_path) as run:
            path = export_project_native_graphs(attachments, self.source, self.root/'graphs',
                                                expected_source_sha256=source_sha, cache=cache)
        self.assertIs(run.call_args.args[0], self.recipe)
        self.assertIs(run.call_args.kwargs['cache'], cache)
        self.assertEqual(run.call_count, 1)
        report = json.loads(path.read_bytes())
        self.assertEqual([row['status'] for row in report['graphs']], ['executed', 'unavailable'])
        self.assertEqual(historical, tuple(item.to_dict() for item in attachments))
        self.assertEqual(report['graphs'][0]['historical_checkpoint_data'], 'unavailable')
        exports.load_native_recipe_result(path.parent/Path(report['graphs'][0]['report']).parent)
        self.model_path.unlink()
        failed = export_project_native_graphs(attachments, self.source, self.root/'failed-graphs',
                                              expected_source_sha256=source_sha, cache=cache)
        self.assertEqual(json.loads(failed.read_bytes())['graphs'][0]['status'], 'failed')

    def test_project_graph_cancel_or_changed_source_never_publishes(self):
        attachments = (_attachment(self.recipe),)
        with self.assertRaises(ValueError):
            export_project_native_graphs(attachments, self.source, self.root/'wrong-source',
                                          expected_source_sha256='0'*64)
        with self.assertRaises(NativeRecipeCancelled):
            export_project_native_graphs(attachments, self.source, self.root/'cancelled-graphs',
                expected_source_sha256=sha256(self.source.read_bytes()).hexdigest(), cancel=lambda: True)
        self.assertFalse((self.root/'cancelled-graphs').exists())
        self.assertFalse(list(self.root.glob('.native-graphs-*')))

    def test_project_releases_exported_graph_products_before_next_run(self):
        from tests.test_plantcv_naive_bayes import _constant_pdf
        from tests.test_native_naive_bayes_recipe import _recipe_with_model
        model_path = self.root/'second-pdf.tsv'
        model_path.write_bytes(_constant_pdf({'first': 1., 'second': 2.}))
        second, _ = _recipe_with_model(model_path)
        cache = PipelineAnalysisCache()
        references = []
        def run_selected(recipe, image_path, **kwargs):
            # Simulate normal cache eviction before the next graph allocates.
            cache.values.clear()
            self.assertTrue(all(reference() is None for reference in references))
            result = run_native_recipe_path(recipe, image_path, **kwargs)
            tensor = next(value for node in result.nodes for value in node.outputs.values()
                          if torch.is_tensor(value))
            references.append(weakref.ref(tensor))
            return result
        with patch('seedvision.export.project_native_graphs.run_native_recipe_path', side_effect=run_selected):
            path = export_project_native_graphs((_attachment(self.recipe), _attachment(second)),
                self.source, self.root/'released-graphs',
                expected_source_sha256=sha256(self.source.read_bytes()).hexdigest(), cache=cache)
        self.assertEqual([row['status'] for row in json.loads(path.read_bytes())['graphs']],
                         ['executed', 'executed'])
        cache.values.clear()
        self.assertTrue(all(reference() is None for reference in references))

    def test_project_final_source_recheck_and_failed_export_cleanup(self):
        attachments = (_attachment(self.recipe),)
        source_sha = sha256(self.source.read_bytes()).hexdigest()
        def partial_failure(destination, *args, **kwargs):
            destination.mkdir()
            (destination/'partial.bin').write_bytes(b'incomplete')
            raise OSError('injected export failure')
        with patch('seedvision.export.project_native_graphs.export_native_recipe_result', side_effect=partial_failure):
            ledger = export_project_native_graphs(attachments, self.source, self.root/'failed-export',
                                                  expected_source_sha256=source_sha)
        self.assertEqual(json.loads(ledger.read_bytes())['graphs'][0]['status'], 'failed')
        self.assertEqual(list(ledger.parent.iterdir()), [ledger])
        original = self.source.read_bytes()
        publish = exports.export_native_recipe_result
        def change_source(*args, **kwargs):
            report = publish(*args, **kwargs)
            self.source.write_bytes(original+b'changed')
            return report
        with patch('seedvision.export.project_native_graphs.export_native_recipe_result', side_effect=change_source):
            with self.assertRaises(ValueError):
                export_project_native_graphs(attachments, self.source, self.root/'changed-source',
                                              expected_source_sha256=source_sha)
        self.assertFalse((self.root/'changed-source').exists())
        self.assertFalse(list(self.root.glob('.native-graphs-*')))

    def test_saved_project_batch_produces_baseline_and_native_outputs(self):
        from seedvision.persistence import ProjectAnalysisStore, ProjectImageSpec, analysis_settings_profile_from_graph
        from seedvision.pipeline import build_default_pipeline
        from seedvision.export.project_runner import run_project
        from tests.test_seed_crop_artifacts import _analysis
        (self.root/'config').mkdir()
        shutil.copyfile(Path(__file__).parents[1]/'config'/'traits.json', self.root/'config'/'traits.json')
        master = ProjectAnalysisStore(self.root).capture_and_save(
            analysis_settings=analysis_settings_profile_from_graph(build_default_pipeline()),
            images=(ProjectImageSpec(self.source, (13, 17)),), reference_graphs=(_attachment(self.recipe),),
            destination=self.root/'test.seedfiddle-project.json')
        project_bytes = master.read_bytes()
        baseline_references = []
        def analyze(*args, **kwargs):
            result = _analysis(self.source)
            baseline_references.append(weakref.ref(result.calibration.gpu_corrected_bgr))
            return result
        def native_graphs(*args, **kwargs):
            self.assertTrue(all(reference() is None for reference in baseline_references))
            return export_project_native_graphs(*args, **kwargs)
        with patch('seedvision.export.project_runner.analyze_path', side_effect=analyze), \
                patch('seedvision.export.project_runner.export_project_native_graphs', side_effect=native_graphs):
            paths = run_project(self.root, master, self.root/'output')
        self.assertEqual(len(paths), 1)
        baseline = json.loads((paths[0]/'results.json').read_bytes())
        self.assertEqual(baseline['schema_version'], 3)
        self.assertEqual(baseline['identity']['revision_contract'], 'seedfiddle.result-revision-v7')
        ledger = json.loads((paths[0]/'native_graphs'/'native-graphs.json').read_bytes())
        self.assertEqual(ledger['graphs'][0]['status'], 'executed')
        self.assertEqual(master.read_bytes(), project_bytes)


class NativeMetadataTests(unittest.TestCase):
    def test_ordered_mappings_tuples_and_nonfinite_values_are_unambiguous(self):
        from seedvision.benchmarking.checkpoints import CoordinateFrame
        collector = exports._Collector(CoordinateFrame('metadata', (2, 3), 'top-left', 'xy', 'pixel'),
                                       CheckpointLimits(), None)
        encoded = collector.encode({'kind': ('first', [float('nan'), float('inf'), -float('inf')]),
                                    'second': {'value': 2., 'available': False}})
        self.assertEqual([name for name, _ in encoded['items']], ['kind', 'second'])
        values = encoded['items'][0][1]['items'][1]['items']
        self.assertEqual([value['value'] for value in values], ['nan', 'positive_infinity', 'negative_infinity'])
        self.assertEqual(json.loads(exports._json_bytes(encoded)), encoded)
        self.assertEqual(collector.arrays, {})

    def test_metadata_depth_cycles_cpu_tensors_and_visit_caps_reject(self):
        from seedvision.benchmarking.checkpoints import CoordinateFrame
        def collector():
            return exports._Collector(CoordinateFrame('metadata', (2, 3), 'top-left', 'xy', 'pixel'),
                                      CheckpointLimits(), None)
        cycle = []
        cycle.append(cycle)
        for value in (cycle, torch.zeros((2, 3)), np.zeros((2, 3), np.uint8), 2**257):
            with self.assertRaises(exports.NativeResultExportError):
                collector().encode(value)
        with patch.object(exports, 'MAX_METADATA_ITEMS', 2):
            with self.assertRaises(exports.NativeResultExportError):
                collector().encode([1, 2, 3])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_ambiguous_layout_and_unversioned_tensors_are_explicit(self):
        from seedvision.benchmarking.checkpoints import CoordinateFrame
        collector = exports._Collector(CoordinateFrame('square', (3, 3), 'top-left', 'xy', 'pixel'),
                                       CheckpointLimits(), None)
        binding = collector.array(torch.zeros((3, 3, 3), dtype=torch.uint8, device='cuda'))
        self.assertNotIn(binding['name'], collector.roles)
        with torch.inference_mode():
            value = torch.zeros((3, 3), dtype=torch.uint8, device='cuda')
        with self.assertRaises(exports.NativeResultExportError):
            collector.array(value)

    def test_full_128_label_colour_metadata_domain_and_escaped_text_bound(self):
        from seedvision.benchmarking.checkpoints import CoordinateFrame
        from seedvision.cuda.plantcv_color import ColorAnalysisResult, ColorLabelResult, ColorHistogram
        histograms = tuple(ColorHistogram('channel'+str(index), 'frequency', 'method', 'nominal',
            'percent', 'float', tuple(map(float, range(256))), (0.,)*256) for index in range(12))
        observations = ColorAnalysisResult('available', 'all', tuple(
            ColorLabelResult(index, f'seed_{index}', 1, histograms, ()) for index in range(1, 129)))
        collector = exports._Collector(CoordinateFrame('metadata', (2, 3), 'top-left', 'xy', 'pixel'),
                                       CheckpointLimits(), None)
        encoded = collector.encode(observations)
        self.assertLess(len(exports._json_bytes(encoded)), exports.MAX_REPORT_BYTES)
        with patch.object(exports, 'MAX_REPORT_BYTES', 256):
            with self.assertRaises(exports.NativeResultExportError):
                collector.encode('\u0001'*100)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class NativeDesktopExportTests(unittest.TestCase):
    setUp = NativeResultExportTests.setUp
    tearDown = NativeResultExportTests.tearDown
    make_recipe = NativeResultExportTests.make_recipe
    run_recipe = NativeResultExportTests.run_recipe

    def test_export_button_worker_persists_actual_run_and_drains_on_close(self):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        from PySide6.QtWidgets import QApplication
        from seedvision.ui.native_kmeans_dialog import NativeKMeansDialog
        app = QApplication.instance() or QApplication([])
        dialog = NativeKMeansDialog(self.recipe)
        dialog.set_source_path(self.source)
        result = self.run_recipe()
        dialog._completed(result)
        self.assertTrue(dialog.export_button.isEnabled())
        with patch('seedvision.ui.native_recipe_dialog.QFileDialog.getExistingDirectory', return_value=str(self.root)):
            dialog.export_button.click()
        self.assertIsNotNone(dialog._worker)
        self.assertFalse(dialog.choose_model_button.isEnabled())
        self.assertFalse(dialog.train_model_button.isEnabled())
        deadline = time.monotonic()+30
        while dialog._worker is not None and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.01)
        self.assertIsNone(dialog._worker, dialog.status_label.text())
        self.assertIn('exported:', dialog.status_label.text())
        self.assertIs(dialog.result, result)
        report = next(self.root.glob('plantcv_kmeans*/native-results.json'))
        exports.load_native_recipe_result(report.parent)
        self.assertTrue(dialog.choose_model_button.isEnabled())
        dialog._parameter_changed(kmeans.MASK_STAGE_ID, 'cat_list', '[1]')
        self.assertFalse(dialog.export_button.isEnabled())
        dialog.reject()
        dialog.deleteLater()

    def test_close_cancels_and_drains_export_worker(self):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        from PySide6.QtWidgets import QApplication
        from seedvision.ui.native_kmeans_dialog import NativeKMeansDialog
        app = QApplication.instance() or QApplication([])
        dialog = NativeKMeansDialog(self.recipe)
        dialog.set_source_path(self.source)
        dialog._completed(self.run_recipe())
        dialog.show()
        entered = threading.Event()
        def blocked_export(*args, cancel, **kwargs):
            entered.set()
            deadline = time.monotonic()+10
            while not cancel() and time.monotonic() < deadline:
                time.sleep(.01)
            raise NativeRecipeCancelled('cancelled test export')
        with patch('seedvision.ui.native_recipe_dialog.QFileDialog.getExistingDirectory', return_value=str(self.root)), \
                patch('seedvision.ui.native_recipe_dialog.export_native_recipe_result', side_effect=blocked_export):
            dialog.export_button.click()
            deadline = time.monotonic()+10
            while not entered.is_set() and time.monotonic() < deadline:
                app.processEvents()
                time.sleep(.01)
            self.assertTrue(entered.is_set())
            dialog.reject()
            self.assertTrue(dialog._close_requested)
            self.assertIsNotNone(dialog._worker)
            deadline = time.monotonic()+10
            while dialog._worker is not None and time.monotonic() < deadline:
                app.processEvents()
                time.sleep(.01)
            self.assertIsNone(dialog._worker)
            self.assertFalse(dialog.isVisible())
        self.assertFalse(list(self.root.glob('plantcv_kmeans*/native-results.json')))
        dialog.deleteLater()
