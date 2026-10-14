"""Public rectangle-helper equality and integrated native ROI morphology."""
from dataclasses import replace
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from seedvision.benchmarking import native_recipe as runner
from seedvision.benchmarking import native_morphology_roi_recipe as factory
from seedvision.benchmarking import native_stage_replay as replay
from seedvision.benchmarking.recipes import RecipeSpec
from seedvision.cuda import plantcv_morphology as primitive
from seedvision.cuda.tensor_callback import TensorCallbackMutation
from seedvision.export import native_results as archives
from seedvision.segmentation.baseline import PipelineAnalysisCache
from tests.test_native_recipe import fixture, selected_recipe, FRAME, SOURCE, PREPARATION, REFERENCE

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')


def _change(recipe, operation, **values):
    return replace(recipe, stages=tuple(replace(s, parameters=tuple(
        replace(p, value=values.get(p.parameter_id, p.value)) for p in s.parameters))
        if s.operation_id == operation else s for s in recipe.stages))


def _recipe():
    return factory.build_native_morphology_roi_recipe(runner.build_native_seed_recipe(selected_recipe()))


REFERENCE_SCRIPT = r'''
import hashlib, importlib, json, sys
from importlib.metadata import version
from pathlib import Path
import numpy as np
from plantcv import plantcv as pcv
from plantcv.plantcv.classes import Objects
assert version('plantcv') == '4.11.3'
assert version('opencv-python') == '4.11.0.86'
helper=importlib.import_module('plantcv.plantcv._helpers')
assert hashlib.sha256(Path(helper.__file__).read_bytes()).hexdigest() == 'c05b0c944210657332ce307f2066caab55f2ebc8d8fea11988db3e476e7cdfba'
for name,digest in [('erode','6ae07bc734ae7226b9f39f5b970d549319e49b45ca9559b0ae3064e02a67487a'),('dilate','e0bf2fcf8ca96e8d7aa5cb05cf3b7e4dbdb163450466126bc8fb124549f2472f')]:
 module=importlib.import_module('plantcv.plantcv.'+name)
 assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()==digest
pcv.params.debug=None
source=np.load(sys.argv[1],allow_pickle=False)['gray']
cases=json.loads(Path(sys.argv[2]).read_text())
results={}
for index,case in enumerate(cases):
 bounds=case['roi'];roi=None
 if bounds is not None:
  x0,y0,x1,y1=bounds
  contour=[np.array([[[x0,y0]],[[x0,y1]],[[x1,y1]],[[x1,y0]]],np.int32)]
  hierarchy=np.array([[[-1,-1,-1,-1]]],np.int32)
  roi=Objects(contours=[contour],hierarchy=[hierarchy])
 output=getattr(pcv,case['operation'])(gray_img=source,ksize=case['ksize'],i=case['iterations'],roi=roi)
 results[str(index)]=output
 # Public rectangle uses x+w-1/y+h-1 as the helper's exclusive endpoints.
 if bounds is not None and x1<source.shape[1] and y1<source.shape[0]:
  rectangle=pcv.roi.rectangle(source,x=x0,y=y0,w=x1-x0+1,h=y1-y0+1)
  check=getattr(pcv,case['operation'])(gray_img=source,ksize=case['ksize'],i=case['iterations'],roi=rectangle)
  assert np.array_equal(output,check)
np.savez(sys.argv[3],**results)
'''


class MorphologyROIContractTests(unittest.TestCase):
    def test_historical_mask_identity_tracks_shared_callback_implementation(self):
        from seedvision.benchmarking import native_mask_stage as bridge
        from seedvision.cuda import tensor_callback
        bridge._implementation_digest()
        with patch.object(tensor_callback, 'checked_cancel', lambda *args, **kwargs: False):
            with self.assertRaises(bridge.NativeMaskStageError):
                bridge._implementation_digest()

    def test_all_fourteen_variants_roundtrip_without_changing_base_definitions(self):
        for base_id in factory.BASE_IDS:
            with self.subTest(base=base_id):
                base = factory.canonical_base(base_id)
                before = base.to_dict()
                recipe = factory.build_native_morphology_roi_recipe(base)
                runner._validate_recipe(recipe)
                restored = RecipeSpec.from_json(recipe.to_json())
                self.assertEqual(restored.identity, recipe.identity)
                self.assertEqual(base.to_dict(), before)
                self.assertEqual(runner._diagnostic_recipe.base_recipe_id(recipe.recipe_id),
                                 runner._diagnostic_recipe.base_recipe_id(base_id))
                for stage, prior in zip(recipe.stages, base.stages, strict=True):
                    if stage.operation_id in factory.OPERATIONS:
                        self.assertEqual(stage.parameters[:-1], prior.parameters)
                        self.assertIsNone(stage.parameters[-1].value)
                    else:
                        self.assertEqual(stage, prior)

    def test_bounds_and_full_canvas_replay_admission_remain_strict(self):
        parameters = {'ksize': 4, 'iterations': 2, factory.PARAMETER: [2, 3, 12, 13]}
        plan = replay._plan((20, 30), 'morphology.erode', parameters, 1200)
        self.assertEqual(plan['work_details']['roi_pixels'], 100)
        self.assertEqual(plan['work'], 100*4*2*2+600)
        for value in ([0, 0, 31, 20], [0, 0, 0, 1], [True, 0, 1, 1], [0., 0, 1, 1], '0,0,1,1'):
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                replay._plan((20, 30), 'morphology.erode', dict(parameters, roi_bounds_xyxy=value), 1200)
        with self.assertRaises(runner.NativeRecipeError):
            replay._plan((1, 7_340_033), 'morphology.erode',
                         dict(parameters, roi_bounds_xyxy=[0, 0, 1, 1]), 14_680_066)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class MorphologyROIIntegrationTests(unittest.TestCase):
    @unittest.skipUnless(REFERENCE.is_file(), 'Pinned public PlantCV environment required')
    def test_104_public_masks_exact_on_thin_border_shifted_full_and_strided_rois(self):
        rng = np.random.default_rng(1707)
        gray = rng.integers(0, 256, (17, 23), dtype=np.uint8)
        rois = (None, [0, 0, 23, 17], [0, 0, 1, 1], [22, 16, 23, 17],
                [3, 5, 18, 6], [8, 2, 9, 15], [0, 2, 8, 12], [15, 2, 23, 12],
                [3, 0, 15, 7], [3, 10, 15, 17], [2, 3, 12, 13], [3, 3, 13, 13], [10, 8, 11, 9])
        cases = [{'operation': op, 'ksize': k, 'iterations': i, 'roi': roi}
                 for op in ('erode', 'dilate') for k, i in ((2, 1), (4, 2), (5, 3), (255, 64)) for roi in rois]
        self.assertEqual(len(cases), 104)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            np.savez(root/'input.npz', gray=gray)
            (root/'cases.json').write_text(json.dumps(cases))
            result = subprocess.run([str(REFERENCE), '-c', REFERENCE_SCRIPT, str(root/'input.npz'),
                                     str(root/'cases.json'), str(root/'output.npz')],
                                    capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            with np.load(root/'output.npz', allow_pickle=False) as expected:
                # Transpose twice with an expanded column stride gives a dense noncontiguous view.
                backing = torch.empty((17, 46), dtype=torch.uint8, device='cuda')
                source = backing[:, ::2]
                source.copy_(torch.tensor(gray, device='cuda'))
                before = source.clone()
                for index, case in enumerate(cases):
                    roi = None if case['roi'] is None else tuple(case['roi'])
                    actual = getattr(primitive, case['operation'])(source, case['ksize'], case['iterations'], roi=roi)
                    np.testing.assert_array_equal(actual.cpu().numpy(), expected[str(index)], err_msg=str(case))
                    self.assertEqual(actual.device, source.device)
                    self.assertEqual(actual.dtype, torch.uint8)
                    if roi is not None:
                        outside = np.ones(gray.shape, bool)
                        x0, y0, x1, y1 = roi; outside[y0:y1, x0:x1] = False
                        np.testing.assert_array_equal(actual.cpu().numpy()[outside], gray[outside])
                torch.testing.assert_close(source, before)

    def test_crop_boundary_is_independent_and_real_controls_change_pixels(self):
        source = torch.zeros((40, 40), dtype=torch.uint8, device='cuda')
        source[12:28, 12:28] = 255
        roi = (12, 12, 28, 28)
        crop_first = primitive.erode(source, 4, 1, roi=roi)
        full_first = primitive.erode(source, 4, 1)
        self.assertFalse(torch.equal(crop_first[12:28, 12:28], full_first[12:28, 12:28]))
        roi = (16, 16, 32, 32)
        outputs = (primitive.erode(source, 2, 1, roi=roi), primitive.erode(source, 4, 1, roi=roi),
                   primitive.erode(source, 4, 2, roi=roi), primitive.dilate(source, 4, 1, roi=roi))
        self.assertEqual(len({hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest() for t in outputs}), 4)

    def test_invalid_bounds_and_cancellation_never_publish_or_change_input(self):
        source = torch.arange(24, device='cuda', dtype=torch.uint8).reshape(4, 6)
        before = source.clone()
        for roi in ((0, 0, 7, 4), (0, 0, 0, 1), (True, 0, 1, 1), [0, 0, 1, 1]):
            with self.subTest(roi=roi), patch.object(primitive, '_axis_max', side_effect=AssertionError('allocated work')):
                with self.assertRaises((TypeError, ValueError)):
                    primitive.erode(source, 3, roi=roi)
        polls = 0
        def cancel():
            nonlocal polls
            polls += 1
            return polls >= 3
        with self.assertRaises(primitive.MorphologyCancelledError):
            primitive.dilate(source, 2, 4, roi=(1, 1, 5, 4), cancel=cancel)
        with self.assertRaises(TypeError):
            primitive.dilate(source, 2, cancel=lambda: 1)
        torch.testing.assert_close(source, before)

    def test_callback_mutation_of_original_private_buffers_or_storage_withholds_output(self):
        for target, poll in (('gray', 1), ('values', 2), ('result', 3), ('output', 4)):
            for outcome in ('false', 'true', 'raise'):
                with self.subTest(target=target, outcome=outcome):
                    source = torch.ones((8, 10), dtype=torch.uint8, device='cuda')
                    polls = 0
                    def cancel():
                        nonlocal polls
                        polls += 1
                        if polls != poll:
                            return False
                        frame = inspect.currentframe().f_back
                        while frame.f_code is not primitive._morphology.__code__:
                            frame = frame.f_back
                        frame.f_locals[target].fill_(7)
                        if outcome == 'raise':
                            raise LookupError('caller callback failed')
                        return outcome == 'true'
                    with self.assertRaises(TensorCallbackMutation):
                        primitive.dilate(source, 2, roi=(1, 1, 9, 7), cancel=cancel)
        source = torch.ones((8, 10), dtype=torch.uint8, device='cuda')
        def resize_storage():
            storage = source.untyped_storage()
            storage.resize_(storage.nbytes()+1)
            raise LookupError('caller callback failed after resizing storage')
        with self.assertRaises(TensorCallbackMutation):
            primitive.erode(source, 2, cancel=resize_storage)
        def unchanged_error():
            raise LookupError('unchanged caller error')
        with self.assertRaisesRegex(LookupError, 'unchanged caller error'):
            primitive.erode(source, 2, cancel=unchanged_error)

    def test_roi_edits_recompute_only_selected_node_and_descendants(self):
        gray, rgb = fixture()
        gray, rgb = torch.tensor(gray, device='cuda'), torch.tensor(rgb, device='cuda')
        recipe = _change(_recipe(), 'morphology.erode', roi_bounds_xyxy=[20, 20, 70, 70])
        cache = PipelineAnalysisCache()
        def run(selected):
            return runner.run_native_recipe(selected, gray, rgb, source_sha256=SOURCE,
                coordinate_frame=FRAME, preparation=PREPARATION, cache=cache)
        first = run(recipe); second = run(recipe)
        self.assertEqual(len(cache.last_reused_nodes), len(recipe.stages))
        stage = next(s.stage_id for s in recipe.stages if s.operation_id=='morphology.erode')
        self.assertIs(second.node(stage).outputs['mask'], first.node(stage).outputs['mask'])
        shifted = _change(recipe, 'morphology.erode', roi_bounds_xyxy=[21, 20, 71, 70])
        changed = run(shifted)
        self.assertEqual(changed.node(stage).status, 'executed', changed.node(stage).reason)
        self.assertNotIn(stage, cache.last_reused_nodes)
        ancestors = {s.stage_id for s in recipe.stages if s.operation_id in
                     ('external.image_inputs', 'filter.median_blur', 'threshold.binary', 'morphology.fill')}
        self.assertTrue(ancestors.issubset(cache.last_reused_nodes))
        self.assertFalse(torch.equal(changed.node(stage).outputs['mask'], first.node(stage).outputs['mask']))

    def test_archived_rois_and_null_replay_only_selected_stage(self):
        gray, rgb = fixture()
        gray, rgb = torch.tensor(gray, device='cuda'), torch.tensor(rgb, device='cuda')
        for bounds in (None, [20, 20, 70, 70]):
            recipe = _change(_recipe(), 'morphology.erode', roi_bounds_xyxy=bounds)
            recipe = _change(recipe, 'morphology.dilate', roi_bounds_xyxy=bounds)
            run = runner.run_native_recipe(recipe, gray, rgb, source_sha256=SOURCE,
                coordinate_frame=FRAME, preparation=PREPARATION)
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)/'archive'
                archives.export_native_recipe_result(directory, recipe, run)
                before = {str(p.relative_to(directory)): p.read_bytes() for p in directory.rglob('*') if p.is_file()}
                for stage in (s for s in recipe.stages if s.operation_id in factory.OPERATIONS):
                    calls = []
                    def profile(frame, event, arg):
                        if event=='call' and frame.f_code is runner._dispatch.__code__:
                            calls.append(frame.f_locals['operation'])
                    sys.setprofile(profile)
                    try:
                        result = replay.replay_native_stage(directory, stage.stage_id)
                    finally:
                        sys.setprofile(None)
                    self.assertEqual(result.status, 'matched', result.reason)
                    self.assertEqual(calls, [stage.operation_id])
                    torch.testing.assert_close(result.outputs['mask'], run.node(stage.stage_id).outputs['mask'])
                self.assertEqual(before, {str(p.relative_to(directory)): p.read_bytes() for p in directory.rglob('*') if p.is_file()})

    def test_async_desktop_add_edit_run_and_saved_replay_after_source_deleted(self):
        from PySide6.QtWidgets import QApplication
        from seedvision.ui.native_recipe_dialog import NativeRecipeDialog
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); source = root/'input.png'
            gray, _ = fixture()
            self.assertTrue(cv2.imwrite(str(source), np.repeat(gray[..., None], 3, axis=2)))
            kept = []
            dialog = NativeRecipeDialog(runner.build_native_seed_recipe(selected_recipe()),
                                        keep_in_project=kept.append)
            dialog.show()
            try:
                moved = dialog.recipe.stages[0].stage_id
                dialog.canvas.node_items[moved].setPos(123, 456)
                dialog.morphology_roi_button.click()
                self.assertIn(dialog.recipe.recipe_id, factory.RECIPE_IDS)
                self.assertEqual((dialog.canvas.node_items[moved].pos().x(),
                                  dialog.canvas.node_items[moved].pos().y()), (123, 456))
                stage = next(s.stage_id for s in dialog.recipe.stages if s.operation_id=='morphology.erode')
                dialog._parameter_changed(stage, factory.PARAMETER, '[20,20,70,70]')
                dialog.set_source_path(source); dialog.start_run()
                self.assertFalse(dialog.morphology_roi_button.isEnabled())
                deadline = time.monotonic()+45
                while dialog._worker is not None and time.monotonic()<deadline:
                    app.processEvents(); time.sleep(.01)
                app.processEvents()
                self.assertIsNone(dialog._worker)
                self.assertIsNotNone(dialog.result, dialog.status_label.text())
                node = dialog.result.node(stage)
                self.assertEqual(node.status, 'executed', node.reason)
                self.assertEqual(tuple(node.parameters[factory.PARAMETER]), (20,20,70,70))
                dialog._select_stage(stage)
                self.assertFalse(dialog.preview.pixmap().isNull())
                saved = root/'saved'
                selected = dialog._selected_recipe_with_layout()
                dialog.project_button.click()
                self.assertEqual(kept, [selected])
                reopened = NativeRecipeDialog(kept[0])
                self.assertEqual(tuple(reopened.recipe.parameters_for_runner(stage)[factory.PARAMETER]),
                                 (20, 20, 70, 70))
                self.assertEqual((reopened.canvas.node_items[moved].pos().x(),
                                  reopened.canvas.node_items[moved].pos().y()), (123, 456))
                reopened.close(); reopened.deleteLater()
                archives.export_native_recipe_result(saved, selected, dialog.result)
                source.unlink()
                dialog.start_stage_replay(saved)
                self.assertFalse(dialog.morphology_roi_button.isEnabled())
                deadline = time.monotonic()+45
                while dialog._worker is not None and time.monotonic()<deadline:
                    app.processEvents(); time.sleep(.01)
                app.processEvents()
                self.assertIsNone(dialog._worker)
                self.assertEqual(dialog.replay_result.status, 'matched', dialog.status_label.text())
                self.assertIs(dialog.result.node(stage), node)
            finally:
                dialog.close(); app.processEvents()

    def test_calibrated_diagnostic_roi_project_batch_and_local_card_crop_replay(self):
        from seedvision.benchmarking import native_calibrated_seed_recipe as calibrated
        from seedvision.benchmarking import native_card_recipe as card
        from seedvision.benchmarking import native_watershed_diagnostic_recipe as diagnostic
        from seedvision.export.project_native_graphs import export_project_native_graphs
        from seedvision.persistence.project_analysis import ProjectAnalysisStore, ProjectImageSpec
        from seedvision.persistence.analysis_settings import analysis_settings_profile_from_graph
        from seedvision.pipeline import build_default_pipeline
        from tests import test_native_calibrated_seed_recipe as calibrated_tests
        from tests.test_native_result_export import _output
        bounds = [20, 20, 70, 70]
        recipe = factory.build_native_morphology_roi_recipe(
            diagnostic.build_native_watershed_diagnostic_recipe(calibrated_tests.selected_recipe()))
        recipe = _change(recipe, 'morphology.erode', roi_bounds_xyxy=bounds)
        stage = next(s.stage_id for s in recipe.stages if s.operation_id == 'morphology.erode')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); workspace = root/'workspace'; workspace.mkdir()
            (workspace/'config').mkdir()
            shutil.copyfile(Path(__file__).parents[1]/'config/traits.json', workspace/'config/traits.json')
            source = workspace/'seeds.png'; image = calibrated_tests.fixture()
            self.assertTrue(cv2.imwrite(str(source), image))
            actual = runner.run_native_recipe_path(recipe, source)
            self.assertEqual(actual.node(stage).status, 'executed', actual.node(stage).reason)
            self.assertEqual(actual.labels_coordinate_frame.shape, (96, 128))
            self.assertEqual(actual.node(calibrated.CROP).outputs['metadata']['local_to_parent_xy'], (600, 260))
            incoming = next(e for e in recipe.dependencies if e.target_stage == stage and e.target_port == 'gray')
            input_mask = actual.node(incoming.source_stage).outputs[incoming.source_port]
            expected = primitive.erode(input_mask, **{
                k: v for k, v in recipe.parameters_for_runner(stage).items() if k != factory.PARAMETER},
                roi=tuple(bounds))
            torch.testing.assert_close(actual.node(stage).outputs['mask'], expected)
            outside = torch.ones(input_mask.shape, dtype=torch.bool, device='cuda')
            outside[20:70, 20:70] = False
            torch.testing.assert_close(actual.node(stage).outputs['mask'][outside], input_mask[outside])
            store = ProjectAnalysisStore(workspace)
            master = store.capture_and_save(
                analysis_settings=analysis_settings_profile_from_graph(build_default_pipeline()),
                images=(ProjectImageSpec(source, image.shape[:2]),), native_recipes=(recipe,),
                destination=workspace/'seeds.seedfiddle-project.json')
            document = store.load(master).document
            self.assertEqual(document.native_recipes, (recipe,))
            ledger = export_project_native_graphs((), source, root/'runs',
                expected_source_sha256=document.images[0].source.sha256,
                project_root=workspace, native_recipes=document.native_recipes)
            row = json.loads(ledger.read_bytes())['graphs'][0]
            self.assertEqual(row['status'], 'executed', row['reason'])
            saved = ledger.parent/Path(row['report']).parent
            report, arrays = archives.load_native_recipe_result(saved)
            self.assertFalse(report['scientifically_validated'])
            targets = ((stage, 'mask'), (calibrated.CROP, 'rgb_image'),
                       (card.AFFINE, 'corrected_rgb'), (diagnostic.STAGE, 'diagnostic'))
            for stage_id, port in targets:
                np.testing.assert_array_equal(arrays[_output(report, stage_id, port)['name']],
                                              actual.node(stage_id).outputs[port].cpu().numpy())
            source.unlink()
            for stage_id, port in targets:
                calls = []
                def profile(frame, event, arg):
                    if event == 'call' and frame.f_code is runner._dispatch.__code__:
                        calls.append(frame.f_locals['operation'])
                sys.setprofile(profile)
                try:
                    result = replay.replay_native_stage(saved, stage_id)
                finally:
                    sys.setprofile(None)
                self.assertEqual(result.status, 'matched', result.reason)
                operation = next(s.operation_id for s in recipe.stages if s.stage_id == stage_id)
                self.assertEqual(calls, [operation])
                torch.testing.assert_close(result.outputs[port], actual.node(stage_id).outputs[port])
                if stage_id in (stage, diagnostic.STAGE):
                    self.assertEqual(result.metadata['frame_scope'], 'calibrated_seed_roi')
                    self.assertEqual(tuple(result.metadata['coordinate_frame']['shape']), (96, 128))
                    self.assertEqual(result.metadata['frame_lineage']['local_to_parent_xy'], (600, 260))


if __name__ == '__main__':
    unittest.main()
