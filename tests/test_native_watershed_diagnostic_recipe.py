from dataclasses import replace
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from seedvision.benchmarking import native_recipe as runner
from seedvision.benchmarking import native_watershed_diagnostic_recipe as factory
from seedvision.benchmarking.recipes import RecipeSpec
from seedvision.benchmarking.native_stage_replay import replay_native_stage
from seedvision.export import native_results as archives
from seedvision.persistence.native_recipes import native_recipes_from_payload, native_recipes_to_payload
from seedvision.segmentation.baseline import PipelineAnalysisCache
from tests import test_native_recipe as seeds
from tests import test_native_calibrated_seed_recipe as calibrated
from tests.test_plantcv_watershed_diagnostic import oracle


def selected(kind='binary', local=False):
    base = calibrated.selected_recipe(kind) if local else runner.build_native_seed_recipe(seeds.selected_recipe(kind))
    return factory.build_native_watershed_diagnostic_recipe(base)


def controls(recipe, **values):
    return replace(recipe,stages=tuple(replace(stage,parameters=tuple(
        replace(p,value=values.get(p.parameter_id,p.value)) for p in stage.parameters))
        if stage.stage_id == factory.STAGE else stage for stage in recipe.stages))


class WatershedDiagnosticRecipeContractTests(unittest.TestCase):
    def test_all_six_additive_graphs_preserve_base_identity_settings_and_round_trip(self):
        for kind in ('binary','mean','otsu'):
            for local in (False,True):
                base = calibrated.selected_recipe(kind) if local else runner.build_native_seed_recipe(seeds.selected_recipe(kind))
                before = base.to_json()
                recipe = factory.build_native_watershed_diagnostic_recipe(base)
                runner._validate_recipe(recipe)
                self.assertEqual(recipe.stages[:-1],base.stages)
                self.assertEqual(recipe.dependencies[:-2],base.dependencies)
                self.assertEqual(recipe.layout[:-1],base.layout)
                self.assertEqual(base.to_json(),before)
                self.assertEqual(RecipeSpec.from_json(recipe.to_json()),recipe)
                self.assertEqual(native_recipes_from_payload(native_recipes_to_payload((recipe,))),(recipe,))
                self.assertEqual(factory.base_recipe_id(recipe.recipe_id),base.recipe_id)
                self.assertTrue(all(p.editable and p.affects_calculation for p in recipe.stage(factory.STAGE).parameters))
        self.assertEqual(len(factory.RECIPE_IDS),7)

    def test_joint_admission_is_explicit_and_full_committed_frame_fits(self):
        for local in (False,True):
            bound = runner._diagnostic_retained_bound(2080*3120,calibrated=local)
            self.assertLessEqual(bound,runner.MAX_RETAINED_CUDA_BYTES)
            with self.assertRaises(runner.NativeRecipeError):
                runner._diagnostic_retained_bound(16_000_001,calibrated=local)


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class WatershedDiagnosticRecipeCalculationTests(unittest.TestCase):
    def test_joint_admission_counts_larger_caller_backing_before_dispatch(self):
        height,width = seeds.FRAME.shape
        pixels = height*width
        gray_backing = torch.zeros(8*pixels,device='cuda',dtype=torch.uint8)
        rgb_backing = torch.zeros(6*pixels,device='cuda',dtype=torch.uint8)
        gray = gray_backing[:pixels].view(height,width)
        rgb = rgb_backing[:3*pixels].view(height,width,3)
        for local in (False,True):
            recipe = selected(local=local)
            bound = runner._diagnostic_retained_bound(pixels,calibrated=local)
            with patch.object(runner,'MAX_RETAINED_CUDA_BYTES',bound+10*pixels-1), \
                    patch.object(runner,'_dispatch') as dispatch:
                with self.assertRaisesRegex(runner.NativeRecipeError,'source backing'):
                    runner.run_native_recipe(recipe,gray,rgb,source_sha256=seeds.SOURCE,
                        coordinate_frame=seeds.FRAME,preparation=seeds.PREPARATION)
                dispatch.assert_not_called()

    def run_fixture(self, recipe, cache=None):
        if factory.base_recipe_id(recipe.recipe_id).startswith('plantcv_calibrated_'):
            bgr = calibrated.fixture()
            rgb = bgr[...,::-1].copy()
            gray = cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
            frame = replace(seeds.FRAME,shape=gray.shape)
        else:
            gray,rgb = seeds.fixture()
            frame = seeds.FRAME
        return runner.run_native_recipe(recipe,torch.as_tensor(gray,device='cuda'),
            torch.as_tensor(rgb,device='cuda'),source_sha256=seeds.SOURCE,
            coordinate_frame=frame,preparation=seeds.PREPARATION,cache=cache)

    def test_all_six_actual_graphs_draw_exact_public_pixels_in_their_source_or_roi_frame(self):
        cases,actuals = [],[]
        for kind in ('binary','mean','otsu'):
            for local in (False,True):
                recipe = selected(kind,local)
                result = self.run_fixture(recipe)
                diagnostic = result.node(factory.STAGE)
                self.assertEqual(diagnostic.status,'executed',diagnostic.reason)
                image_edge = next(e for e in recipe.dependencies if e.target_stage==factory.STAGE and e.target_port=='image')
                image = result.node(image_edge.source_stage).outputs[image_edge.source_port]
                cases.append(dict(rgb=image.cpu().tolist(),labels=result.labels.cpu().tolist(),
                    color_scale='gist_rainbow',random_seed=0,line_thickness=5))
                actuals.append(diagnostic.outputs['diagnostic'].cpu().numpy())
                self.assertEqual(tuple(diagnostic.outputs['diagnostic'].shape[:2]),result.labels_coordinate_frame.shape)
                flood = next(n for n in result.nodes if n.operation_id=='reference.internal.watershed.masked_flood')
                self.assertEqual(set(flood.outputs),{'labels','flood_complete'})
        for actual,expected in zip(actuals,oracle('overlay',cases)):
            np.testing.assert_array_equal(actual,expected['image'])

    def test_palette_and_stroke_edits_recompute_only_diagnostic_and_preserve_analytical_products(self):
        recipe = selected()
        gray,rgb = seeds.fixture()
        gray,rgb = torch.as_tensor(gray,device='cuda'),torch.as_tensor(rgb,device='cuda')
        cache = PipelineAnalysisCache()
        kwargs = dict(source_sha256=seeds.SOURCE,coordinate_frame=seeds.FRAME,
                      preparation=seeds.PREPARATION,cache=cache)
        first = runner.run_native_recipe(recipe,gray,rgb,**kwargs)
        for values in ({'random_seed':91},{'color_scale':'viridis'},{'line_thickness':1}):
            changed = runner.run_native_recipe(controls(recipe,**values),gray,rgb,**kwargs)
            self.assertEqual(cache.last_computed_nodes,(factory.STAGE,))
            self.assertEqual(len(cache.last_reused_nodes),len(recipe.stages)-1)
            self.assertIs(changed.labels,first.labels)
            self.assertFalse(torch.equal(changed.node(factory.STAGE).outputs['diagnostic'],
                                         first.node(factory.STAGE).outputs['diagnostic']))

    def test_archive_round_trip_and_existing_stage_replay_for_source_and_roi_variants(self):
        for local in (False,True):
            recipe = selected(local=local)
            result = self.run_fixture(recipe)
            with tempfile.TemporaryDirectory(dir='artifacts') as temporary:
                root = Path(temporary)/'run'
                archives.export_native_recipe_result(root,recipe,result)
                report,arrays = archives.load_native_recipe_result(root)
                saved = RecipeSpec.from_json(archives._json_bytes(report['recipe']))
                self.assertEqual(saved,recipe)
                node = next(n for n in report['nodes'] if n['stage_id']==factory.STAGE)
                output = dict(node['outputs'])
                np.testing.assert_array_equal(arrays[output['diagnostic']['name']],
                    result.node(factory.STAGE).outputs['diagnostic'].cpu().numpy())
                for operation in ('threshold.binary','reference.internal.watershed.masked_flood',
                                  'reference.internal.watershed.marker_count_observation'):
                    stage = next(s for s in recipe.stages if s.operation_id==operation)
                    replay = replay_native_stage(root,stage.stage_id)
                    self.assertEqual(replay.status,'matched',replay.reason)
                debug_replay = replay_native_stage(root,factory.STAGE)
                self.assertEqual(debug_replay.status,'matched',debug_replay.reason)

    def test_dispatch_resource_unavailability_and_cancellation_raise(self):
        recipe = selected()
        with patch.object(runner._watershed_diagnostic,'MAX_SCAN_WORK',1):
            # Pin the test-only bound before the initial runtime capture. It is
            # a literal source constant, so bypass runtime verification only in
            # this direct dispatch test; actual run identities are never forged.
            labels = torch.zeros(seeds.FRAME.shape,device='cuda',dtype=torch.int32)
            labels[30:50,40:60]=1
            image = torch.zeros((*seeds.FRAME.shape,3),device='cuda',dtype=torch.uint8)
            outputs = runner._dispatch(factory.OPERATION,{'image':image,'labels':labels},
                recipe.parameters_for_runner(factory.STAGE),None)
            self.assertEqual(set(outputs),{'metadata'})
            self.assertEqual(outputs['metadata']['status'],'unavailable')
        with self.assertRaises(runner.NativeRecipeCancelled):
            self.run_fixture_with_cancel(recipe)

    def run_fixture_with_cancel(self,recipe):
        gray,rgb = seeds.fixture()
        return runner.run_native_recipe(recipe,torch.tensor(gray,device='cuda'),torch.tensor(rgb,device='cuda'),
            source_sha256=seeds.SOURCE,coordinate_frame=seeds.FRAME,
            preparation=seeds.PREPARATION,cancel=lambda:True)


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class WatershedDiagnosticDesktopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_native_action_preserves_dragged_layout_and_actual_async_diagnostic_preview(self):
        from seedvision.ui.native_recipe_dialog import NativeRecipeDialog
        base = runner.build_native_seed_recipe(seeds.selected_recipe())
        captures = []
        parent = NativeRecipeDialog(base,keep_in_project=captures.append)
        parent.show();self.app.processEvents()
        moved = base.stages[0].stage_id
        parent.canvas.node_items[moved].setPos(123,456)
        expected = factory.build_native_watershed_diagnostic_recipe(parent._selected_recipe_with_layout())
        parent._add_watershed_diagnostic()
        self.assertEqual(parent.recipe,expected)
        self.assertEqual((parent.canvas.node_items[moved].pos().x(),
                          parent.canvas.node_items[moved].pos().y()),(123,456))
        self.assertIn(factory.STAGE,parent.canvas.node_items)
        self.assertTrue(parent.canvas.viewport().rect().intersects(parent.canvas.mapFromScene(
            parent.canvas.node_items[factory.STAGE].sceneBoundingRect()).boundingRect()))
        parent.close();parent.deleteLater()
        with tempfile.TemporaryDirectory(dir='artifacts') as temporary:
            gray,_ = seeds.fixture()
            path = Path(temporary)/'seed.png'
            self.assertTrue(cv2.imwrite(str(path),np.repeat(gray[...,None],3,axis=2)))
            dialog = NativeRecipeDialog(selected());dialog.show()
            dialog.set_source_path(path);dialog.start_run()
            deadline = time.monotonic()+90
            while dialog._worker is not None and time.monotonic()<deadline:
                self.app.processEvents();time.sleep(.01)
            self.assertIsNone(dialog._worker)
            self.assertIsNotNone(dialog.result,dialog.status_label.text())
            dialog._select_stage(factory.STAGE)
            self.assertIsNotNone(dialog.preview.pixmap())
            self.assertFalse(dialog.preview.pixmap().isNull())
            self.assertIn('palette',dialog.facts.toPlainText().lower())
            self.assertFalse(dialog.diagnostic_button.isVisible())
            dialog.close();dialog.deleteLater();self.app.processEvents()

    def test_actual_worker_locks_graph_addition_and_close_drains_the_same_dialog(self):
        from seedvision.ui import native_recipe_dialog as ui
        base = runner.build_native_seed_recipe(seeds.selected_recipe())
        started,release = threading.Event(),threading.Event()
        original = ui.run_native_recipe_path
        def held(*args,**kwargs):
            started.set()
            if not release.wait(30):
                raise RuntimeError('Test worker was not released')
            return original(*args,**kwargs)
        with tempfile.TemporaryDirectory(dir='artifacts') as temporary:
            gray,_ = seeds.fixture()
            path = Path(temporary)/'seed.png'
            self.assertTrue(cv2.imwrite(str(path),gray))
            dialog = ui.NativeRecipeDialog(base);dialog.show();dialog.set_source_path(path)
            try:
                with patch.object(ui,'run_native_recipe_path',side_effect=held):
                    dialog.start_run()
                    deadline = time.monotonic()+10
                    while not started.is_set() and time.monotonic()<deadline:
                        self.app.processEvents();time.sleep(.01)
                    self.assertTrue(started.is_set())
                    self.assertFalse(dialog.diagnostic_button.isEnabled())
                    self.assertIn(dialog.diagnostic_button,dialog._export_conflicting_buttons())
                    dialog._add_watershed_diagnostic()
                    self.assertEqual(dialog.recipe,base)
                    dialog.close()
                    self.assertTrue(dialog.isVisible())
                    self.assertTrue(dialog._cancel.is_set())
                    release.set()
                    deadline = time.monotonic()+15
                    while dialog._worker is not None and time.monotonic()<deadline:
                        self.app.processEvents();time.sleep(.01)
                    self.assertIsNone(dialog._worker)
                    self.assertFalse(dialog.isVisible())
            finally:
                release.set();dialog.close();dialog.deleteLater();self.app.processEvents()
