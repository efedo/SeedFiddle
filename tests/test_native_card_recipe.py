"""Automatic-card working graph, CUDA cache boundaries and saved outputs."""
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from seedvision.benchmarking import native_card_recipe as factory, native_recipe as runner
from seedvision.benchmarking.checkpoints import CoordinateFrame
from seedvision.benchmarking.recipes import LayoutPosition
from seedvision.segmentation.baseline import PipelineAnalysisCache
from tests.test_native_kmeans_recipe import _parameter
from tests import test_plantcv_card_detection as card_tests

fixture = card_tests.fixture


class CardContractTests(unittest.TestCase):
    def test_additive_recipe_contract_and_retained_bound(self):
        from seedvision.benchmarking.recipe_catalogue import build_recipe_catalogue
        before = build_recipe_catalogue().to_dict()
        recipe = factory.build_native_card_recipe()
        runner._validate_recipe(recipe)
        self.assertEqual(len(recipe.stages), 9)
        self.assertEqual(recipe.parameters_for_runner(factory.STANDARD)['pos'], 3)
        self.assertEqual(build_recipe_catalogue().to_dict(), before)
        maximum = (1024*1024*1024-256*1024*1024)//32
        self.assertEqual(factory.retained_bound(maximum), 1024*1024*1024)
        with self.assertRaises(factory.card.AutomaticCardResourceError):
            factory.retained_bound(maximum+1)
        for module in factory.implementation_modules():
            runner._module_record(module)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class NativeCardRecipeTests(unittest.TestCase):
    oracle = card_tests.AutomaticCardTests.oracle

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.image = fixture(varied=True)
        self.path = self.root/'card.png'
        self.assertTrue(cv2.imwrite(str(self.path), self.image))
        self.recipe = factory.build_native_card_recipe()
        self.cache = PipelineAnalysisCache()

    def tearDown(self):
        self.temp.cleanup()

    def run_recipe(self, recipe=None, **kwargs):
        return runner.run_native_recipe_path(recipe or self.recipe, self.path, cache=self.cache, **kwargs)

    def test_actual_public_auto_and_roi_default_position(self):
        for bounds in (None, [40,40,420,580]):
            recipe = _parameter(self.recipe, factory.PREPROCESS, 'roi_bounds_xyxy', bounds)
            actual = self.run_recipe(recipe)
            self.assertEqual(actual.status, 'executed', [(n.stage_id,n.reason) for n in actual.nodes])
            expected, _ = self.oracle(self.image, operation='auto', roi_bounds_xyxy=bounds)
            mask = actual.node(factory.MASK).outputs['labeled_mask']
            np.testing.assert_array_equal(mask.cpu().numpy(), expected['mask'])
            np.testing.assert_array_equal(actual.node(factory.MASK).outputs['bounding_mask'].cpu(),expected['bbox'])
            np.testing.assert_allclose(actual.node(factory.MEANS).outputs['source_matrix'],expected['source'],atol=2e-15,rtol=0)
            np.testing.assert_array_equal(actual.node(factory.STANDARD).outputs['target_matrix'],expected['target'])
            corrected = actual.node(factory.AFFINE).outputs['corrected_rgb']
            difference = np.abs(corrected.cpu().numpy().astype(int)-expected['corrected'][...,::-1].astype(int))
            self.assertLessEqual(int(difference.max()),1)
            self.assertEqual(corrected.device.type,'cuda')
            self.assertEqual(mask.dtype,torch.float64)
            geometry = actual.node(factory.GEOMETRY).outputs['geometry']
            np.testing.assert_array_equal(geometry['centers_xy'],expected['centers'])
            np.testing.assert_array_equal(geometry['corners_xy'],expected['corners'])
            with self.assertRaises(TypeError): geometry['detected_chips']=0
            self.assertIsInstance(actual.node(factory.MEANS).outputs['source_matrix'],tuple)

    def test_controls_recompute_only_their_dependents(self):
        first = self.run_recipe()
        self.assertTrue(all(node.reused for node in self.run_recipe().nodes))
        cases = (
            (factory.MASK,'radius',22,{factory.MASK,factory.MEANS,factory.AFFINE}),
            (factory.SCALE,'color_chip_size',[7,11],{factory.SCALE}),
            (factory.STANDARD,'pos',2,{factory.STANDARD,factory.AFFINE}),
            (factory.GEOMETRY,'min_size',1200,{factory.GEOMETRY,factory.MASK,factory.SCALE,factory.MEANS,factory.AFFINE}),
            (factory.GEOMETRY,'aspect_ratio',1.4,{factory.GEOMETRY,factory.MASK,factory.SCALE,factory.MEANS,factory.AFFINE}),
            (factory.GEOMETRY,'solidity',.75,{factory.GEOMETRY,factory.MASK,factory.SCALE,factory.MEANS,factory.AFFINE}),
            (factory.PREPROCESS,'block_size',35,{factory.PREPROCESS,factory.GEOMETRY,factory.MASK,factory.SCALE,factory.MEANS,factory.AFFINE}),
            (factory.PREPROCESS,'adaptive_method',0,{factory.PREPROCESS,factory.GEOMETRY,factory.MASK,factory.SCALE,factory.MEANS,factory.AFFINE}),
            (factory.PREPROCESS,'roi_bounds_xyxy',[40,40,420,580],{factory.PREPROCESS,factory.GEOMETRY,factory.MASK,factory.SCALE,factory.MEANS,factory.AFFINE}),
        )
        selected = self.recipe
        for stage,name,value,computed in cases:
            selected = _parameter(selected,stage,name,value)
            result = self.run_recipe(selected)
            self.assertEqual(result.status,'executed', [(n.stage_id,n.reason) for n in result.nodes])
            self.assertEqual(set(self.cache.last_computed_nodes),computed)
            if stage == factory.MASK:
                self.assertFalse(torch.equal(first.node(factory.MASK).outputs['labeled_mask'],result.node(factory.MASK).outputs['labeled_mask']))
            if stage == factory.SCALE:
                self.assertEqual(result.node(factory.SCALE).outputs['metadata']['unit'],'mm')
        layout = replace(selected, title='Working card', layout=tuple(LayoutPosition(s.stage_id,100.+i*200,80.) for i,s in enumerate(selected.stages)))
        self.assertTrue(all(node.reused for node in self.run_recipe(layout).nodes))

    def test_failed_filter_and_precision_have_no_stale_descendants_and_recover(self):
        self.run_recipe()
        failed = self.run_recipe(_parameter(self.recipe,factory.GEOMETRY,'min_size',10000))
        self.assertEqual(failed.node(factory.GEOMETRY).status,'failed')
        for stage in (factory.MASK,factory.SCALE,factory.MEANS,factory.AFFINE):
            self.assertEqual(failed.node(stage).status,'not_run')
            self.assertFalse(failed.node(stage).outputs)
        self.assertTrue(failed.node(factory.STANDARD).reused)
        self.assertEqual(self.run_recipe().status,'executed')
        bad = _parameter(self.recipe,factory.PREPROCESS,'block_size',4)
        failed = self.run_recipe(bad)
        self.assertEqual(failed.node(factory.PREPROCESS).status,'failed')
        self.assertEqual(failed.node(factory.AFFINE).status,'not_run')
        self.assertEqual(self.run_recipe().status,'executed')
        uncertain=fixture();uncertain[uncertain==40]=60
        cv2.imwrite(str(self.path),uncertain)
        failed=self.run_recipe()
        self.assertEqual(failed.node(factory.PREPROCESS).status,'failed')
        self.assertIn('ColorCardPreprocessPrecisionError',failed.node(factory.PREPROCESS).reason)
        self.assertEqual(failed.node(factory.AFFINE).status,'not_run')
        cv2.imwrite(str(self.path),self.image)
        self.assertEqual(self.run_recipe().status,'executed')

    def test_source_and_final_output_callback_mutation_reject_results(self):
        prepared = runner.prepare_native_recipe_inputs(self.path)
        def change_source():
            prepared.rgb[0,0,0] ^= 1
            return False
        with self.assertRaisesRegex(runner.NativeRecipeError,'changed during execution'):
            runner.run_native_recipe(self.recipe,prepared.gray,prepared.rgb,
                source_sha256=prepared.source_sha256,coordinate_frame=prepared.coordinate_frame,
                preparation=prepared.preparation,cache=self.cache,cancel=change_source)
        first = self.run_recipe()
        output = first.node(factory.AFFINE).outputs['corrected_rgb']
        affine_key = 'reference.recipe:'+factory.RECIPE_ID+':'+factory.AFFINE
        calls = 0
        def change_final():
            nonlocal calls
            calls += 1
            # One initial, nine node and one final callback in a fully cached run.
            if calls == 11: output[0,0,0] ^= 1
            return False
        with self.assertRaisesRegex(runner.NativeRecipeError,'changed during execution'):
            prepared = self.cache.values['reference.prepared'][1]
            runner.run_native_recipe(self.recipe,prepared.gray,prepared.rgb,
                source_sha256=prepared.source_sha256,coordinate_frame=prepared.coordinate_frame,
                preparation=prepared.preparation,cache=self.cache,cancel=change_final)
        self.assertNotIn(affine_key,self.cache.values)
        self.assertEqual(self.run_recipe().status,'executed')

    def test_cancel_and_cached_mask_mutation_recovery(self):
        first = self.run_recipe()
        first.node(factory.MASK).outputs['labeled_mask'].zero_()
        repaired = self.run_recipe()
        self.assertEqual(set(self.cache.last_computed_nodes),{factory.MASK,factory.MEANS,factory.AFFINE})
        self.assertGreater(int(repaired.node(factory.MASK).outputs['labeled_mask'].max()),0)
        with self.assertRaises(runner.NativeRecipeCancelled): self.run_recipe(cancel=lambda: True)
        self.assertEqual(self.run_recipe().status,'executed')

    def test_mid_stage_cancel_and_callback_exception_mutation_withhold_outputs(self):
        self.run_recipe()
        key='reference.recipe:'+factory.RECIPE_ID+':'+factory.MASK
        self.cache.values.pop(key)
        calls=0
        def cancel():
            nonlocal calls
            calls+=1
            return calls==18
        with self.assertRaises(runner.NativeRecipeCancelled): self.run_recipe(cancel=cancel)
        self.assertNotIn(key,self.cache.values)
        self.assertEqual(self.run_recipe().status,'executed')
        prepared=self.cache.values['reference.prepared'][1]
        def mutate_and_raise():
            prepared.rgb[0,0,0]^=1
            raise RuntimeError('callback failure')
        with self.assertRaisesRegex(runner.NativeRecipeError,'changed during execution'):
            runner.run_native_recipe(self.recipe,prepared.gray,prepared.rgb,
                source_sha256=prepared.source_sha256,coordinate_frame=prepared.coordinate_frame,
                preparation=prepared.preparation,cache=self.cache,cancel=mutate_and_raise)
        self.assertFalse(any(name.startswith('reference.recipe:'+factory.RECIPE_ID+':') for name in self.cache.values))
        self.assertEqual(self.run_recipe().status,'executed')

    def test_loaded_geometry_and_fixed_metadata_identity_checked(self):
        with patch.object(factory.minrect,'_no_guard',lambda: False):
            with self.assertRaises(ValueError): self.run_recipe()
        changed = factory.card.GRID_BOX_411.copy(); changed.setflags(write=False)
        with patch.object(factory.card,'GRID_BOX_411',changed):
            # A byte-identical immutable replacement is permitted; changed bytes are not.
            factory.metadata_identity()
        changed = factory.card.GRID_BOX_411.copy();changed[0,0]=1;changed.setflags(write=False)
        with patch.object(factory.card,'GRID_BOX_411',changed):
            with self.assertRaisesRegex(ValueError,'Fixed grid'): self.run_recipe()

    def test_workspace_rejection_precedes_cuda_upload(self):
        with patch.object(factory,'retained_bound',side_effect=factory.card.AutomaticCardResourceError('fixture cap')):
            with patch.object(torch,'as_tensor',side_effect=AssertionError('must not allocate')):
                with self.assertRaisesRegex(factory.card.AutomaticCardResourceError,'fixture cap'):
                    self.run_recipe()

    def test_custom_stream_cache_replacement_storage_and_compact_mutation(self):
        import gc
        from types import MappingProxyType
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        with torch.cuda.stream(torch.cuda.Stream()):
            self.run_recipe()
            changed = _parameter(self.recipe,factory.PREPROCESS,'roi_bounds_xyxy',[40,40,420,580])
            result = self.run_recipe(changed)
            self.assertEqual(result.status,'executed')
        torch.cuda.synchronize()
        self.assertLessEqual(torch.cuda.max_memory_allocated()-base,factory.retained_bound(self.image.shape[0]*self.image.shape[1]))
        name = 'reference.recipe:'+factory.RECIPE_ID+':'+factory.GEOMETRY
        entry = self.cache.values[name]
        old = entry.node
        geometry = dict(old.outputs['geometry']);geometry['detected_chips']=0
        entry.node = replace(old,outputs=MappingProxyType({**old.outputs,'geometry':MappingProxyType(geometry)}))
        result = self.run_recipe(changed)
        self.assertEqual(set(self.cache.last_computed_nodes),{factory.GEOMETRY,factory.MASK,factory.SCALE,factory.MEANS,factory.AFFINE})
        self.assertEqual(result.node(factory.GEOMETRY).outputs['geometry']['detected_chips'],24)
        del result,old,entry
        self.cache.values.clear();gc.collect();torch.cuda.synchronize()

    def test_saved_project_batch_and_whole_run_output_roundtrip(self):
        from seedvision.export import native_results as outputs
        from seedvision.export.project_native_graphs import export_project_native_graphs
        from seedvision.persistence.project_analysis import ProjectAnalysisStore, ProjectImageSpec
        from seedvision.persistence.analysis_settings import analysis_settings_profile_from_graph
        from seedvision.pipeline import build_default_pipeline
        from tests.test_native_result_export import _attachment, _output
        from tests.test_native_recipe import selected_recipe
        workspace = self.root/'workspace';workspace.mkdir();(workspace/'config').mkdir()
        source = workspace/'card.png';shutil.copyfile(self.path,source)
        shutil.copyfile(Path(__file__).parents[1]/'config/traits.json',workspace/'config/traits.json')
        history = (_attachment(selected_recipe()),)
        selected = _parameter(self.recipe,factory.SCALE,'color_chip_size','mini')
        store = ProjectAnalysisStore(workspace)
        master = store.capture_and_save(analysis_settings=analysis_settings_profile_from_graph(build_default_pipeline()),
            images=(ProjectImageSpec(source,self.image.shape[:2]),),reference_graphs=history,
            native_recipes=(selected,),destination=workspace/'card.seedfiddle-project.json')
        document = store.load(master).document
        self.assertEqual(document.native_recipes,(selected,));self.assertEqual(document.reference_graphs,history)
        ledger = export_project_native_graphs((),source,self.root/'runs',
            expected_source_sha256=document.images[0].source.sha256,project_root=workspace,native_recipes=document.native_recipes)
        row = json.loads(ledger.read_bytes())['graphs'][0]
        self.assertEqual(row['status'],'executed',row['reason']);self.assertEqual(row['historical_stage_runs'],[])
        report,arrays = outputs.load_native_recipe_result(ledger.parent/Path(row['report']).parent)
        self.assertEqual(report['recipe_identity'],selected.identity.to_dict())
        labels = _output(report,factory.MASK,'labeled_mask')
        self.assertEqual(arrays[labels['name']].dtype,np.float64)
        self.assertEqual(arrays[labels['name']].shape,self.image.shape[:2])
        corrected = _output(report,factory.AFFINE,'corrected_rgb')
        self.assertEqual(arrays[corrected['name']].dtype,np.uint8)
        expected = self.run_recipe(selected)
        np.testing.assert_array_equal(arrays[labels['name']],expected.node(factory.MASK).outputs['labeled_mask'].cpu())
        bbox = _output(report,factory.MASK,'bounding_mask')
        np.testing.assert_array_equal(arrays[bbox['name']],expected.node(factory.MASK).outputs['bounding_mask'].cpu())
        np.testing.assert_array_equal(arrays[corrected['name']],expected.node(factory.AFFINE).outputs['corrected_rgb'].cpu())
        self.assertFalse(report['scientifically_validated'])
        self.assertEqual(store.load(master).document.reference_graphs,history)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class NativeCardDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def wait(self, dialog):
        deadline = time.monotonic()+30
        while dialog._worker is not None and time.monotonic()<deadline:
            self.app.processEvents();time.sleep(.01)
        self.app.processEvents();self.assertIsNone(dialog._worker)

    def test_actual_async_card_preview_effective_edit_keep_and_reopen(self):
        from seedvision.ui.native_recipe_dialog import NativeRecipeDialog
        from seedvision.ui.reference_graph_dialog import ReferenceGraphDialog
        kept = []
        recipe = factory.build_native_card_recipe()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'card.png';cv2.imwrite(str(path),fixture(varied=True))
            dialog = NativeRecipeDialog(recipe,keep_in_project=lambda value: kept.append(value))
            dialog.show()
            try:
                dialog.set_source_path(path);dialog.start_run()
                self.assertFalse(dialog.inspector.isEnabled());self.wait(dialog)
                self.assertEqual(dialog.result.status,'executed',dialog.status_label.text())
                dialog._select_stage(factory.MASK);self.assertFalse(dialog.preview.pixmap().isNull())
                dialog.inspector.parameter_changed.emit(factory.PREPROCESS,'roi_bounds_xyxy','[40,40,420,580]')
                dialog.inspector.parameter_changed.emit(factory.SCALE,'color_chip_size','"mini"')
                dialog.start_run();self.wait(dialog)
                dialog._select_stage(factory.PREPROCESS);self.assertFalse(dialog.preview.pixmap().isNull())
                dialog._select_stage(factory.SCALE);self.assertIn('mm',dialog.facts.toPlainText())
                dialog._keep_recipe_in_project();self.assertEqual(len(kept),1)
                selected = kept[0]
                self.assertEqual(selected.parameters_for_runner(factory.SCALE)['color_chip_size'],'mini')
            finally:
                dialog.reject();self.wait(dialog);dialog.deleteLater()
            reopened = NativeRecipeDialog(selected)
            try: self.assertEqual(reopened.recipe,selected)
            finally: reopened.reject();reopened.deleteLater()
        reference = ReferenceGraphDialog()
        try:
            with patch.object(reference,'_open_working_native_recipe') as opened:
                reference._automatic_card_button.click()
            self.assertEqual(opened.call_args.args[0],recipe)
        finally: reference.reject();reference.deleteLater()
