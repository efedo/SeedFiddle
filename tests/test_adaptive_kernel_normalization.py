"""Pinned public adaptive-kernel normalization, cache identity and saved replay."""
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

import cv2
import numpy as np
import torch

from seedvision.benchmarking import native_recipe as runner
from seedvision.benchmarking import native_stage_replay as replay
from seedvision.benchmarking import native_threshold_stage as bridge
from seedvision.benchmarking import reference_stage as reference_bridge
from seedvision.benchmarking.checkpoints import load_checkpoint
from seedvision.benchmarking.recipe_catalogue import build_recipe_catalogue
from seedvision.cuda import plantcv_thresholds as mean
from seedvision.cuda import plantcv_extended_thresholds as extended
from seedvision.export import native_results as archives
from seedvision.segmentation.baseline import PipelineAnalysisCache
from tests.test_native_recipe import fixture, selected_recipe, FRAME, SOURCE, PREPARATION
from tests.test_native_threshold_stage import _fixture as native_fixture
from tests.test_reference_stage import _fixture as reference_fixture, _gray_image
from tests.test_plantcv_extended_thresholds import SCRIPT, REFERENCE

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')


def public_masks(test, image, operation, cases):
    test.assertTrue(REFERENCE.is_file(), 'Pinned public environment is required')
    process = subprocess.run([str(REFERENCE), '-c', SCRIPT],
        input=json.dumps(dict(image=image.tolist(), operation=operation, cases=cases)),
        capture_output=True, text=True, timeout=90)
    test.assertEqual(process.returncode, 0, process.stderr)
    return [np.asarray(value, dtype=np.uint8) for value in json.loads(process.stdout)]


def mean_recipe(kernel):
    return runner.build_native_seed_recipe(selected_recipe('mean',
        **{'threshold.mean': {'ksize': kernel}}))


class AdaptiveKernelContractTests(unittest.TestCase):
    def test_integer_requests_normalize_before_effective_cap_and_keep_other_controls(self):
        for requested, effective in ((3, 3), (4, 5), (254, 255), (255, 255)):
            self.assertEqual(mean._validate_mean_parameters(requested, -1.25), (effective, -1.25))
        for requested in (True, 2, 256, 257, 4.0):
            with self.subTest(requested=requested), self.assertRaises((TypeError, ValueError)):
                mean._validate_mean_parameters(requested, 0)
        for offset in (True, float('nan'), float('inf'), 256):
            with self.subTest(offset=offset), self.assertRaises((TypeError, ValueError)):
                mean._validate_mean_parameters(4, offset)

    def test_replay_plans_use_effective_kernel_without_rewriting_request(self):
        for requested in (4, 254):
            parameters = dict(ksize=requested, offset=.5, object_type='light')
            plan = replay._plan((17, 23), 'threshold.mean', parameters, 2*17*23)
            self.assertEqual(parameters['ksize'], requested)
            self.assertEqual(plan['work_details']['requested_kernel_size'], requested)
            self.assertEqual(plan['work_details']['effective_kernel_size'], requested+1)
        for requested in (4, 14):
            parameters = dict(ksize=requested, offset=.5, object_type='light')
            plan = replay._plan((31, 47), 'threshold.gaussian', parameters, 2*31*47)
            self.assertEqual(plan['work_details']['requested_kernel_size'], requested)
            self.assertEqual(plan['work_details']['effective_kernel_size'], requested+1)
            self.assertEqual(plan['work_details']['window_sample_visits'], 31*47*(requested+1)**2)
        maximum = extended.MAX_GAUSSIAN_FLOAT_WORK//(15*15)
        parameters = dict(ksize=14, offset=.5, object_type='light')
        admitted = replay._extended_threshold_admission(maximum, 'threshold.gaussian', parameters)
        self.assertEqual(admitted['window_sample_visits'], maximum*15*15)
        with self.assertRaisesRegex(replay.NativeStageReplayUnavailable, 'work limit'):
            replay._extended_threshold_admission(maximum+1, 'threshold.gaussian', parameters)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class AdaptiveKernelCudaTests(unittest.TestCase):
    def compare_cases(self, image, operation, cases):
        expected = public_masks(self, image, operation, cases)
        source = torch.tensor(image, device='cuda')
        version = source._version
        function = mean.threshold_mean if operation == 'mean' else extended.threshold_gaussian
        for parameters, original in zip(cases, expected):
            with self.subTest(operation=operation, shape=image.shape, parameters=parameters):
                actual = function(source, **parameters)
                self.assertEqual(actual.device, source.device)
                self.assertEqual(actual.dtype, torch.uint8)
                np.testing.assert_array_equal(actual.cpu().numpy(), original)
        self.assertEqual(source._version, version)
        np.testing.assert_array_equal(source.cpu().numpy(), image)

    def test_every_even_mean_kernel_matches_pinned_public_masks_on_structured_frame(self):
        image = np.full((17, 23), 37, dtype=np.uint8)
        image[:, 9:] = 219
        cases = [dict(ksize=k, offset=offset, object_type=polarity)
            for k in range(4, 255, 2) for offset in (-1.25, 0, .5)
            for polarity in ('light', 'dark')]
        self.assertEqual(len(cases), 756)
        self.compare_cases(image, 'mean', cases)

    def test_mean_thin_singleton_borders_and_kernel_changes_match_public(self):
        cases = [dict(ksize=k, offset=offset, object_type=polarity)
            for k in (4, 128, 254) for offset in (-1.25, 0, .5)
            for polarity in ('light', 'dark')]
        for image in (np.array([[0, 1, 127, 128, 254, 255]], dtype=np.uint8),
                np.array([[0], [1], [127], [128], [254], [255]], dtype=np.uint8),
                np.array([[127]], dtype=np.uint8)):
            self.compare_cases(image, 'mean', cases)
        image = np.full((31, 47), 37, dtype=np.uint8)
        image[:, 19:] = 219
        effects = [dict(ksize=k, offset=12, object_type='light') for k in (4, 5, 6, 7)]
        masks = public_masks(self, image, 'mean', effects)
        self.compare_cases(image, 'mean', effects)
        np.testing.assert_array_equal(masks[0], masks[1])
        np.testing.assert_array_equal(masks[2], masks[3])
        self.assertFalse(np.array_equal(masks[0], masks[2]), 'Effective kernel must affect calculation')

    def test_all_even_gaussian_kernels_match_pinned_public_masks(self):
        image = np.full((31, 47), 37, dtype=np.uint8)
        image[:, 19:] = 219
        cases = [dict(ksize=k, offset=offset, object_type=polarity)
            for k in (4, 6, 8, 10, 12, 14) for offset in (-255, -1.25, 0, .5, 12, 255)
            for polarity in ('light', 'dark')]
        self.assertEqual(len(cases), 72)
        self.compare_cases(image, 'gaussian', cases)

    def test_even_maximum_mean_keeps_precision_withholding_and_publishes_no_checkpoint(self):
        image = np.full((255, 3), 255, dtype=np.uint8)
        image[0, 0], image[1, 0], image[127, 2] = 0, 254, 253
        parameters = dict(ksize=254, offset=2, object_type='dark')
        expected = public_masks(self, image, 'mean', [parameters])[0]
        self.assertEqual(int(expected[127, 2]), 255)
        source = torch.tensor(image, device='cuda')
        with self.assertRaisesRegex(mean.AdaptiveMeanPrecisionError, 'scalar/SIMD'):
            mean.threshold_mean(source, **parameters)
        np.testing.assert_array_equal(source.cpu().numpy(), image)
        with tempfile.TemporaryDirectory() as temporary:
            args, _, _ = native_fixture(temporary, 'threshold.mean', values=image, selected=parameters)
            outcome = bridge.run_native_threshold_stage(**args)
            self.assertEqual(outcome.run_record.state.value, 'unsupported')
            self.assertIsNone(outcome.checkpoint_reference)
            self.assertIn('rounding boundary', outcome.run_record.reason)

    def test_actual_reference_and_native_checkpoints_separate_selected_and_effective_kernel(self):
        parameters = dict(ksize=4, offset=1.25, object_type='light')
        selected = selected_recipe('mean', **{'threshold.mean': parameters})
        with tempfile.TemporaryDirectory() as temporary:
            args, _, _, _, measurement = reference_fixture(temporary,
                operation_id='threshold.mean', recipe_override=selected)
            original = reference_bridge.run_reference_stage(**args)
            self.assertIsNotNone(original.checkpoint_reference, original.run_record.reason)
            saved = load_checkpoint(args['output_root'], original.checkpoint_reference.checkpoint_sha256,
                expected_identity=args['expected_output_identity'], expected_measurement=measurement)
            self.assertEqual(saved.metadata['parameters']['ksize'], 4)
            self.assertEqual(saved.metadata['effective_parameters']['ksize'], 5)
            expected = saved.arrays['mask'].copy()
        with tempfile.TemporaryDirectory() as temporary:
            args, _, _ = native_fixture(temporary, 'threshold.mean', values=_gray_image(), selected=parameters)
            outcome = bridge.run_native_threshold_stage(**args)
            self.assertIsNotNone(outcome.checkpoint_reference, outcome.run_record.reason)
            saved = load_checkpoint(args['output_root'], outcome.checkpoint_reference.checkpoint_sha256,
                expected_identity=args['expected_output_identity'], expected_measurement=args['output_measurement'])
            self.assertEqual(saved.metadata['selected_parameters']['ksize'], 4)
            self.assertEqual(saved.metadata['effective_parameters']['ksize'], 5)
            np.testing.assert_array_equal(saved.arrays['mask'], expected)

    def test_request_identity_invalidates_only_threshold_dependents_and_saved_stage_replays(self):
        history = build_recipe_catalogue().to_json()
        recipe = mean_recipe(4)
        changed = mean_recipe(5)
        self.assertNotEqual(recipe.identity, changed.identity)
        gray, rgb = fixture()
        gray, rgb = torch.tensor(gray, device='cuda'), torch.tensor(rgb, device='cuda')
        cache = PipelineAnalysisCache()
        def run(value):
            return runner.run_native_recipe(value, gray, rgb, source_sha256=SOURCE,
                coordinate_frame=FRAME, preparation=PREPARATION, cache=cache)
        first = run(recipe)
        stage = next(s.stage_id for s in recipe.stages if s.operation_id == 'threshold.mean')
        self.assertEqual(first.node(stage).status, 'executed', first.node(stage).reason)
        second = run(recipe)
        self.assertEqual(cache.last_computed_nodes, ())
        self.assertIs(second.node(stage).outputs['mask'], first.node(stage).outputs['mask'])
        calls = []
        def profile(frame, event, arg):
            if event == 'call' and frame.f_code is runner._dispatch.__code__:
                calls.append(frame.f_locals['operation'])
        sys.setprofile(profile)
        try:
            third = run(changed)
        finally:
            sys.setprofile(None)
        parents = {s.stage_id for s in recipe.stages
            if s.operation_id in ('external.image_inputs', 'filter.median_blur')}
        self.assertCountEqual(calls, [s.operation_id for s in recipe.stages if s.stage_id not in parents])
        self.assertIn(stage, cache.last_computed_nodes)
        self.assertFalse(parents & set(cache.last_computed_nodes))
        for parent in parents:
            for port, output in first.node(parent).outputs.items():
                self.assertIs(third.node(parent).outputs[port], output)
        self.assertEqual(first.node(stage).parameters['ksize'], 4)
        self.assertEqual(third.node(stage).parameters['ksize'], 5)
        self.assertTrue(torch.equal(first.node(stage).outputs['mask'], third.node(stage).outputs['mask']))
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)/'even-run'
            archives.export_native_recipe_result(directory, recipe, first)
            before = {str(p.relative_to(directory)): sha256(p.read_bytes()).hexdigest()
                for p in directory.rglob('*') if p.is_file()}
            result = replay.replay_native_stage(directory, stage)
            self.assertEqual(result.status, 'matched', result.reason)
            self.assertEqual(result.metadata['comparison']['mismatched_elements'], 0)
            self.assertEqual(result.metadata['plan']['work_details']['requested_kernel_size'], 4)
            self.assertEqual(result.metadata['plan']['work_details']['effective_kernel_size'], 5)
            self.assertTrue(torch.equal(result.outputs['mask'], first.node(stage).outputs['mask']))
            after = {str(p.relative_to(directory)): sha256(p.read_bytes()).hexdigest()
                for p in directory.rglob('*') if p.is_file()}
            self.assertEqual(after, before)
        self.assertEqual(build_recipe_catalogue().to_json(), history)

    def test_even_gaussian_async_qt_archives_replay_with_requested_and_effective_sizes(self):
        from PySide6.QtWidgets import QApplication
        from seedvision.ui.native_recipe_dialog import NativeRecipeDialog
        from seedvision.benchmarking.native_extended_threshold_recipes import build_native_extended_threshold_recipes
        application = QApplication.instance() or QApplication([])
        image = np.full((31, 47), 37, dtype=np.uint8)
        image[:, 19:] = 219
        cases = [dict(ksize=k, offset=.5, object_type='light') for k in (4, 14)]
        expected = public_masks(self, image, 'gaussian', cases)
        base = build_native_extended_threshold_recipes()['plantcv_threshold_gaussian']
        stage = base.stages[1].stage_id
        recipe = replace(base, stages=(base.stages[0], replace(base.stages[1],
            parameters=tuple(replace(p, value=cases[0].get(p.parameter_id, p.value))
                for p in base.stages[1].parameters))))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root/'gaussian.png'
            self.assertTrue(cv2.imwrite(str(path), np.repeat(image[..., None], 3, axis=2)))
            dialog = NativeRecipeDialog(recipe)
            dialog.show()
            saved = []
            try:
                dialog.set_source_path(path)
                for index, (parameters, original) in enumerate(zip(cases, expected)):
                    if index:
                        dialog._parameter_changed(stage, 'ksize', str(parameters['ksize']))
                    dialog.start_run()
                    deadline = time.monotonic()+30
                    while dialog._worker is not None and time.monotonic() < deadline:
                        application.processEvents()
                        time.sleep(.01)
                    application.processEvents()
                    self.assertIsNone(dialog._worker, 'Qt worker did not complete')
                    self.assertIsNotNone(dialog.result, dialog.status_label.text())
                    node = dialog.result.node(stage)
                    self.assertEqual(node.status, 'executed', node.reason)
                    self.assertEqual(node.parameters['ksize'], parameters['ksize'])
                    np.testing.assert_array_equal(node.outputs['mask'].cpu().numpy(), original)
                    dialog._select_stage(stage)
                    self.assertIn(f"requested {parameters['ksize']}, effective {parameters['ksize']+1} pixels",
                        dialog.facts.toPlainText())
                    self.assertFalse(dialog.preview.pixmap().isNull())
                    if index:
                        self.assertEqual(dialog.cache.last_computed_nodes, (stage,))
                    directory = root/f"kernel-{parameters['ksize']}"
                    archives.export_native_recipe_result(directory, dialog.recipe, dialog.result)
                    saved.append((directory, parameters, original))
                path.unlink()  # Replay must use the archived raster after source deletion.
                for directory, parameters, original in saved:
                    before = {str(p.relative_to(directory)): sha256(p.read_bytes()).hexdigest()
                        for p in directory.rglob('*') if p.is_file()}
                    result = replay.replay_native_stage(directory, stage)
                    self.assertEqual(result.status, 'matched', result.reason)
                    self.assertEqual(result.metadata['comparison']['mismatched_elements'], 0)
                    work = result.metadata['plan']['work_details']
                    self.assertEqual(work['requested_kernel_size'], parameters['ksize'])
                    self.assertEqual(work['effective_kernel_size'], parameters['ksize']+1)
                    self.assertEqual(work['window_sample_visits'], image.size*(parameters['ksize']+1)**2)
                    np.testing.assert_array_equal(result.outputs['mask'].cpu().numpy(), original)
                    after = {str(p.relative_to(directory)): sha256(p.read_bytes()).hexdigest()
                        for p in directory.rglob('*') if p.is_file()}
                    self.assertEqual(before, after)
            finally:
                dialog.close()
                application.processEvents()

    def test_normal_async_qt_stage_facts_report_effective_kernel(self):
        from PySide6.QtWidgets import QApplication
        from seedvision.ui.native_recipe_dialog import NativeRecipeDialog
        application = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'seed.png'
            gray, _ = fixture()
            self.assertTrue(cv2.imwrite(str(path), np.repeat(gray[..., None], 3, axis=2)))
            recipe = mean_recipe(4)
            dialog = NativeRecipeDialog(recipe)
            dialog.show()
            try:
                dialog.set_source_path(path)
                dialog.start_run()
                deadline = time.monotonic()+30
                while dialog._worker is not None and time.monotonic() < deadline:
                    application.processEvents()
                    time.sleep(.01)
                application.processEvents()
                self.assertIsNone(dialog._worker, 'Qt worker did not complete')
                self.assertIsNotNone(dialog.result, dialog.status_label.text())
                stage = next(s.stage_id for s in recipe.stages if s.operation_id == 'threshold.mean')
                dialog._select_stage(stage)
                self.assertEqual(dialog.result.node(stage).status, 'executed')
                self.assertIn('requested 4, effective 5 pixels', dialog.facts.toPlainText())
                self.assertIn("'ksize': 4", dialog.facts.toPlainText())
                spec = next(p for p in dialog.graph.nodes[stage].parameter_specs if p.key == 'ksize')
                self.assertIn('even sizes round up to odd', spec.description)
                self.assertNotIn('runner rejects', spec.description)
                self.assertFalse(dialog.preview.pixmap().isNull())
            finally:
                dialog.close()
                application.processEvents()


if __name__ == '__main__':
    unittest.main()
