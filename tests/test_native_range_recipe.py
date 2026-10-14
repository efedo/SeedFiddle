"""Public range outputs through real native, archive, Qt and batch paths."""
from dataclasses import replace
from hashlib import sha256
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from seedvision.benchmarking import native_range_recipe as factory
from seedvision.benchmarking import native_recipe as runner
from seedvision.benchmarking import native_stage_replay as replay
from seedvision.benchmarking.checkpoints import CoordinateFrame
from seedvision.benchmarking.native_extended_threshold_recipes import build_native_extended_threshold_recipes
from seedvision.benchmarking.recipe_catalogue import build_recipe_catalogue
from seedvision.benchmarking.recipes import RecipeSpec
from seedvision.cuda import plantcv_range as primitive
from seedvision.export import native_results as archives
from seedvision.export.project_native_graphs import export_project_native_graphs
from seedvision.persistence import native_recipes as persistence
from seedvision.persistence.reference_regions import file_sha256
from seedvision.segmentation.baseline import PipelineAnalysisCache
from tests.test_native_extended_threshold_recipe import _fixture, _with_parameters
from tests.test_plantcv_range import _reference_custom_range, _reference_channel_bytes


def execute(recipe, gray, rgb, cache=None, cancel=None):
    return runner.run_native_recipe(recipe, gray, rgb, source_sha256='a'*64,
        coordinate_frame=CoordinateFrame('test.native-range', tuple(gray.shape),
            'top-left', 'x-right,y-down', 'pixel'),
        preparation={'gray':'explicit grayscale', 'rgb':'explicit original RGB'},
        cache=cache, cancel=cancel)


class RangeContractTests(unittest.TestCase):
    def test_separate_canonical_recipe_roundtrip_preserves_reference_and_extended_graphs(self):
        history = build_recipe_catalogue().to_json()
        extended = {key:value.to_json() for key,value in build_native_extended_threshold_recipes().items()}
        recipe = factory.build_native_range_recipe()
        runner._validate_recipe(recipe)
        self.assertEqual(RecipeSpec.from_json(recipe.to_json()), recipe)
        self.assertEqual(persistence.native_recipes_from_payload(
            persistence.native_recipes_to_payload((recipe,))), (recipe,))
        stage = recipe.stage(factory.STAGE_ID)
        self.assertEqual([p.name for p in stage.ports], ['rgb','mask','masked_rgb'])
        self.assertTrue(all(p.affects_calculation and p.editable for p in stage.parameters))
        self.assertEqual(stage.source_mappings[0].resolution.value, 'pinned')
        self.assertEqual(build_recipe_catalogue().to_json(), history)
        self.assertEqual({key:value.to_json() for key,value in build_native_extended_threshold_recipes().items()}, extended)
        bad = replace(recipe, stages=(recipe.stages[0],replace(stage, operation_id='threshold.dual_channels')))
        with self.assertRaises(runner.NativeRecipeError):runner._validate_recipe(bad)

    def test_shared_admission_and_exact_replay_resource_boundaries(self):
        controls = dict(channel='CMYK',lower=[0]*4,upper=[255]*4)
        plan = replay._plan((887,887),factory.OPERATION,controls,7*887*887)
        self.assertEqual(plan['cuda_bytes'],factory.retained_bound(887*887))
        self.assertEqual(plan['work_details']['primitive_output_bytes'],4*887*887)
        self.assertGreater(plan['work_details']['tile_scratch_bytes'],0)
        replay._plan((1,7_190_235),factory.OPERATION,controls,7*7_190_235)
        with self.assertRaises(replay.NativeStageReplayUnavailable):
            replay._plan((1,7_190_236),factory.OPERATION,controls,7*7_190_236)
        replay._plan((1,7_340_032),factory.OPERATION,controls,1)
        with self.assertRaises(replay.NativeStageReplayUnavailable):
            replay._plan((1,7_340_033),factory.OPERATION,controls,1)
        with self.assertRaises(primitive.RangeResourceError):primitive._range_admission(64_000_001)
        for changed in (dict(channel='other'),dict(lower=[0]),dict(upper=[256]*4),dict(lower=[False]*4)):
            with self.subTest(changed=changed),self.assertRaises(archives.NativeResultExportError):
                replay._plan((31,37),factory.OPERATION,{**controls,**changed},1)


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class RangeExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(dir='artifacts')
        cls.root = Path(cls.temporary.name)
        gray,rgb,_ = _fixture()
        # Endpoints include unequal R/B slots and the full neutral byte axis.
        flat = rgb.reshape(-1,3)
        flat[:256] = np.repeat(np.arange(256,dtype=np.uint8)[:,None],3,axis=1)
        flat[256:262] = [[250,30,10],[10,30,250],[0,0,0],[255,255,255],[0,255,0],[20,110,210]]
        cls.gray,cls.rgb = torch.tensor(gray,device='cuda'),torch.tensor(rgb,device='cuda')
        cls.recipe = factory.build_native_range_recipe()
        cls.cases = []
        channels = _reference_channel_bytes(np.ascontiguousarray(rgb[...,::-1]))
        for mode in ('RGB','HSV','LAB','CMYK','GRAY'):
            count = 4 if mode=='CMYK' else 1 if mode=='GRAY' else 3
            values = channels[mode.lower()][256].astype(int).tolist()
            cls.cases.extend((dict(name=mode+'-range',channel=mode,
                lower=[0,30,0,20] if mode=='CMYK' else [30]*count,
                upper=[180,220,180,200] if mode=='CMYK' else [180]*count),
                dict(name=mode+'-equal',channel=mode,lower=values,upper=values),
                dict(name=mode+'-reversed',channel=mode,lower=[180]*count,upper=[30]*count)))
        cls.reference = _reference_custom_range(np.ascontiguousarray(rgb[...,::-1]),cls.cases)

    @classmethod
    def tearDownClass(cls):cls.temporary.cleanup()

    def save(self, recipe=None, run=None, *, change=None, runtime=None):
        recipe = recipe or self.recipe
        run = run or execute(recipe,self.gray,self.rgb)
        if change:
            run = replace(run,nodes=tuple(replace(n,outputs={**n.outputs,**change})
                if n.stage_id==factory.STAGE_ID else n for n in run.nodes))
        if runtime is not None:run=replace(run,runtime=runtime)
        directory = self.root/str(len(tuple(self.root.iterdir())))
        archives.export_native_recipe_result(directory,recipe,run)
        return directory,run

    def test_fifteen_mode_endpoint_and_reversed_bound_cases_match_both_public_outputs(self):
        for case in self.cases:
            with self.subTest(case=case['name']):
                recipe = _with_parameters(self.recipe,**{key:value for key,value in case.items() if key!='name'})
                run = execute(recipe,self.gray,self.rgb)
                self.assertEqual(run.status,'executed',[(n.status,n.reason) for n in run.nodes])
                outputs = run.node(factory.STAGE_ID).outputs
                np.testing.assert_array_equal(outputs['mask'].cpu().numpy(),self.reference[case['name']+'_mask'])
                np.testing.assert_array_equal(outputs['masked_rgb'].cpu().numpy(),
                    self.reference[case['name']+'_masked'][...,::-1])
                self.assertTrue(all(t.is_cuda and t.is_contiguous() for t in outputs.values()))
                if case['name'].endswith('equal'):
                    self.assertEqual(outputs['mask'].view(-1)[256].item(),255)
                if case['name'].endswith('range'):
                    count=torch.count_nonzero(outputs['mask']).item()
                    self.assertGreater(count,0);self.assertLess(count,self.gray.numel())
                if case['name'].endswith('reversed'):
                    self.assertEqual(torch.count_nonzero(outputs['mask']).item(),0)

    def test_each_control_changes_calculation_and_only_threshold_is_recomputed(self):
        cache=PipelineAnalysisCache()
        recipe=_with_parameters(self.recipe,lower=[0,0,0],upper=[100,180,255])
        previous=execute(recipe,self.gray,self.rgb,cache)
        self.assertEqual(execute(recipe,self.gray,self.rgb,cache).status,'executed')
        self.assertEqual(cache.last_computed_nodes,())
        for update in (dict(lower=[70,0,0]),dict(upper=[255,180,255]),dict(channel='HSV')):
            changed=_with_parameters(recipe,**update)
            current=execute(changed,self.gray,self.rgb,cache)
            self.assertEqual(current.status,'executed')
            self.assertFalse(torch.equal(current.node(factory.STAGE_ID).outputs['mask'],previous.node(factory.STAGE_ID).outputs['mask']))
            self.assertEqual(cache.last_computed_nodes,(factory.STAGE_ID,))
            self.assertEqual(cache.last_reused_nodes,(recipe.stages[0].stage_id,))
            previous,recipe=current,changed

    def test_actual_archives_replay_exact_arrays_and_leave_original_bytes_unchanged(self):
        for mode in ('RGB','HSV','LAB','CMYK','GRAY'):
            count=4 if mode=='CMYK' else 1 if mode=='GRAY' else 3
            recipe=_with_parameters(self.recipe,channel=mode,
                lower=[0,30,0,20] if mode=='CMYK' else [30]*count,
                upper=[180,220,180,200] if mode=='CMYK' else [180]*count)
            directory,run=self.save(recipe)
            hashes={p.name:sha256(p.read_bytes()).hexdigest() for p in directory.iterdir() if p.is_file()}
            report,_=archives.read_native_recipe_inventory(directory)
            nodes={n['stage_id']:dict(n['outputs']) for n in report['nodes']}
            source=nodes[recipe.stages[0].stage_id]['rgb_image']['name']
            expected={source,*(nodes[factory.STAGE_ID][p]['name'] for p in ('mask','masked_rgb'))}
            loads=[]
            def profile(frame,event,arg):
                if event=='call' and frame.f_code is archives.load_native_recipe_result.__code__:
                    loads.append(set(frame.f_locals['array_names']))
            old=sys.getprofile();sys.setprofile(profile)
            try:result=replay.replay_native_stage(directory,factory.STAGE_ID)
            finally:sys.setprofile(old)
            self.assertEqual(result.status,'matched',result.reason)
            self.assertEqual(loads,[expected])
            self.assertEqual(result.metadata['comparison']['compared_elements'],4*self.gray.numel())
            self.assertFalse(result.scientifically_validated)
            self.assertEqual(result.metadata['preview_port'],'masked_rgb')
            self.assertTrue(torch.equal(result.outputs['masked_rgb'],run.node(factory.STAGE_ID).outputs['masked_rgb']))
            self.assertEqual({p.name:sha256(p.read_bytes()).hexdigest() for p in directory.iterdir() if p.is_file()},hashes)

    def test_well_typed_wrong_image_or_mask_is_mismatch_and_malformed_controls_never_load(self):
        for change in ({'mask':torch.zeros_like(self.gray)}, {'masked_rgb':torch.zeros_like(self.rgb)}):
            directory,_=self.save(change=change)
            result=replay.replay_native_stage(directory,factory.STAGE_ID)
            self.assertEqual(result.status,'mismatch',result.reason)
            self.assertGreater(result.metadata['comparison']['mismatched_elements'],0)
        bad=_with_parameters(self.recipe,lower=[0])
        base=execute(self.recipe,self.gray,self.rgb)
        forged=replace(base,recipe_identity=bad.identity,nodes=tuple(replace(n,
            parameters=bad.parameters_for_runner(n.stage_id)) for n in base.nodes))
        directory,_=self.save(bad,forged)
        with patch.object(archives,'load_native_recipe_result',side_effect=AssertionError('must not load')):
            result=replay.replay_native_stage(directory,factory.STAGE_ID)
        self.assertEqual(result.status,'invalid_archive',result.reason)

    def test_cancellation_mutation_and_runtime_changes_withhold_products_and_cache(self):
        cache=PipelineAnalysisCache()
        execute(self.recipe,self.gray,self.rgb,cache)
        def mutate():
            self.rgb[0,0,0].add_(1)
            return False
        with self.assertRaises(runner.NativeRecipeError):execute(self.recipe,self.gray,self.rgb,cache,mutate)
        self.assertFalse(any(k.startswith('reference.recipe:'+factory.RECIPE_ID+':') for k in cache.values))
        with self.assertRaises(runner.NativeRecipeCancelled):execute(self.recipe,self.gray,self.rgb,cancel=lambda:True)
        directory,_=self.save()
        result=replay.replay_native_stage(directory,factory.STAGE_ID,cancel=lambda:True)
        self.assertEqual(result.status,'cancelled');self.assertFalse(result.outputs)
        changed=inspect.getsource(primitive._range_admission)
        def change_runtime():
            primitive._range_admission=lambda p:{}
            return False
        original=primitive._range_admission
        try:
            result=replay.replay_native_stage(directory,factory.STAGE_ID,cancel=change_runtime)
        finally:primitive._range_admission=original
        self.assertEqual(result.status,'failed',changed)
        self.assertFalse(result.outputs)
        directory,run=self.save(runtime={'legacy':'runtime without a calculation closure'})
        result=replay.replay_native_stage(directory,factory.STAGE_ID)
        self.assertEqual(result.status,'unavailable',result.reason);self.assertFalse(result.outputs)

    def test_saved_recipe_batch_and_empty_mask_archive(self):
        with tempfile.TemporaryDirectory(dir='artifacts') as temporary:
            root=Path(temporary);image=root/'seed.png'
            self.assertTrue(cv2.imwrite(str(image),self.rgb.cpu().numpy()[...,::-1]))
            recipe=_with_parameters(self.recipe,lower=[180]*3,upper=[30]*3)
            ledger=export_project_native_graphs((),image,root/'batch',
                expected_source_sha256=file_sha256(image),native_recipes=(recipe,))
            rows=json.loads(ledger.read_bytes())['graphs']
            self.assertEqual([r['status'] for r in rows],['executed'])
            self.assertEqual(rows[0]['recipe_identity'],recipe.identity.to_dict())
            directory,run=self.save(recipe)
            self.assertEqual(torch.count_nonzero(run.node(factory.STAGE_ID).outputs['mask']).item(),0)
            self.assertEqual(replay.replay_native_stage(directory,factory.STAGE_ID).status,'matched')

    def test_private_output_storage_mutation_at_final_callback_is_withheld(self):
        bgr=torch.empty((1,3,*self.gray.shape),dtype=torch.uint8,device='cuda')
        for c in range(3):bgr[0,c].copy_(self.rgb[...,2-c])
        touched=[]
        def corrupt():
            frame=inspect.currentframe().f_back
            while frame is not None:
                if frame.f_code is primitive.threshold_custom_range.__code__ and 'tile_image' in frame.f_locals:
                    output=frame.f_locals['masked'];storage=output.untyped_storage()
                    storage.resize_(storage.nbytes()+1);touched.append(True)
                    break
                frame=frame.f_back
            return False
        with self.assertRaises(primitive.RangeMutationError):
            primitive.threshold_custom_range(bgr,'RGB',[0]*3,[255]*3,cancel=corrupt)
        self.assertEqual(touched,[True])
        with self.assertRaisesRegex(TypeError,'return bool'):
            primitive.threshold_custom_range(bgr,'RGB',[0]*3,[255]*3,cancel=lambda:1)

    def test_bad_control_clears_old_cached_outputs_and_later_tile_cancel_returns_nothing(self):
        cache=PipelineAnalysisCache()
        execute(self.recipe,self.gray,self.rgb,cache)
        bad=_with_parameters(self.recipe,channel='CMYK')
        failed=execute(bad,self.gray,self.rgb,cache)
        self.assertEqual(failed.node(factory.STAGE_ID).status,'failed')
        self.assertFalse(failed.node(factory.STAGE_ID).outputs)
        self.assertNotIn('reference.recipe:'+factory.RECIPE_ID+':'+factory.STAGE_ID,cache.values)
        bgr=torch.full((1,3,887,887),80,dtype=torch.uint8,device='cuda')
        calls=[]
        def cancel_later():calls.append(True);return len(calls)==4
        with self.assertRaises(primitive.RangeCancelledError):
            primitive.threshold_custom_range(bgr,'CMYK',[0]*4,[255]*4,cancel=cancel_later)
        self.assertEqual(len(calls),4)


class RangeQtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
        from PySide6.QtWidgets import QApplication
        cls.app=QApplication.instance() or QApplication([])

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_actual_async_execution_replay_keep_and_threshold_picker(self):
        from PySide6.QtWidgets import QInputDialog
        from seedvision.ui.native_recipe_dialog import NativeRecipeDialog
        from seedvision.ui.reference_graph_dialog import ReferenceGraphDialog
        selected=[];recipe=factory.build_native_range_recipe()
        dialog=NativeRecipeDialog(recipe,keep_in_project=selected.append)
        with tempfile.TemporaryDirectory(dir='artifacts') as temporary:
            root=Path(temporary);image=root/'seed.png';_,rgb,_=_fixture()
            cv2.imwrite(str(image),rgb[...,::-1]);dialog.show()
            try:
                dialog.set_source_path(image);dialog.start_run()
                deadline=time.monotonic()+40
                while dialog._worker is not None and time.monotonic()<deadline:
                    self.app.processEvents();time.sleep(.01)
                self.assertIsNone(dialog._worker)
                self.assertIsNotNone(dialog.result,dialog.status_label.text())
                self.assertEqual(dialog.result.status,'executed')
                dialog._select_stage(factory.STAGE_ID)
                self.assertFalse(dialog.preview.pixmap().isNull())
                current=dialog.result
                archives.export_native_recipe_result(root/'run',recipe,current)
                dialog.start_stage_replay(root/'run');deadline=time.monotonic()+40
                while dialog._worker is not None and time.monotonic()<deadline:
                    self.app.processEvents();time.sleep(.01)
                self.assertIsNone(dialog._worker)
                self.assertIs(dialog.result,current)
                self.assertIn('matched',dialog.status_label.text().lower())
                self.assertFalse(dialog.preview.pixmap().isNull())
                dialog._keep_recipe_in_project();self.assertEqual(selected,[recipe])
            finally:
                dialog.reject()
                while dialog._worker is not None:self.app.processEvents();time.sleep(.01)
                dialog.deleteLater()
        parent=ReferenceGraphDialog(native_recipes=tuple(selected))
        try:
            with patch.object(QInputDialog,'getItem',return_value=(recipe.title,True)) as picker,\
                    patch.object(parent,'_open_working_native_recipe') as opened:
                parent._open_native_extended_threshold()
            self.assertIn(recipe.title,picker.call_args.args[3]);opened.assert_called_once_with(recipe)
        finally:parent.reject();parent.deleteLater()


if __name__=='__main__':unittest.main()
