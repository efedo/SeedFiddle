from dataclasses import replace
from hashlib import sha256
import inspect
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import cv2
import torch

from seedvision.benchmarking import native_recipe as runner
from seedvision.benchmarking import native_stage_replay as replay
from seedvision.benchmarking import native_watershed_diagnostic_recipe as factory
from seedvision.benchmarking import native_watershed_diagnostic_replay as compact
from seedvision.export import native_results as archives
from tests import test_native_recipe as seeds
from tests import test_native_calibrated_seed_recipe as calibrated
from tests.test_native_watershed_diagnostic_recipe import controls, selected


def binding(value):
    if isinstance(value,dict):
        return {'kind':'mapping','items':[[key,binding(item)] for key,item in value.items()]}
    if isinstance(value,(tuple,list)):
        return {'kind':'tuple','items':[binding(item) for item in value]}
    return value


class DiagnosticReplayPlanTests(unittest.TestCase):
    def test_new_plan_admits_full_fixture_and_preserves_old_contract_envelopes(self):
        operation = factory.OPERATION
        for pixels in (1,1_000_000,1_000_001,2080*3120):
            plan = replay._plan((1,pixels),operation,{},10*pixels)
            self.assertEqual(plan['cuda_bytes'],max(128*pixels+128*1024*1024,
                13*pixels+256*1024*1024+64*min(pixels,1_000_000)+3*1024*1024))
            self.assertIsNone(plan['work_limit'])
            self.assertEqual(plan['selected_limit_bytes'],128*1024*1024)
            self.assertEqual(plan['work_details']['max_metadata_bytes'],256*1024)
        maximum=(1024*1024*1024-128*1024*1024)//128
        self.assertEqual(replay._plan((1,maximum),operation,{},1)['cuda_bytes'],1024*1024*1024)
        for shape,amount in (((1,maximum+1),1),((96,128),128*1024*1024+1)):
            with self.assertRaises(runner.NativeRecipeError):
                replay._plan(shape,operation,{},amount)
        for operation in replay.WATERSHED_TYPED_OPERATIONS:
            p = {'distance':8} if operation.endswith('local_maxima') else {}
            self.assertEqual(replay._plan((96,128),operation,p,1)['cuda_bytes'],128*96*128+128*1024*1024)
        self.assertEqual(replay._plan((96,128),'threshold.binary',{},1)['selected_limit_bytes'],48*1024*1024)

    def test_decoder_counts_keys_depth_bytes_and_rejects_nonprimitive_tags(self):
        bad = [dict(kind='array',name='array_0000'),dict(kind='record',type='anything',fields={}),
               {'kind':'mapping','items':[['a',1],['a',2]]},1.0,{'kind':'tuple','items':[2**65]},
               'x'*100_000,'\u20ac'*342,'\ud800',{'kind':'x'*100_000,'items':[]}]
        deep = 0
        for _ in range(compact.MAX_METADATA_DEPTH+2):
            deep = {'kind':'tuple','items':[deep]}
        bad.append(deep)
        for value in bad:
            with self.assertRaises(ValueError):
                compact.decode(value,ValueError)
        with patch.object(compact,'MAX_METADATA_ITEMS',2):
            with self.assertRaises(ValueError):
                compact.decode(binding({'a':1}),ValueError)
        with patch.object(compact,'MAX_METADATA_BYTES',33):
            with self.assertRaises(ValueError):
                compact.decode(binding((255,255)),ValueError)

    def test_comparison_counts_each_extra_palette_channel_and_finishes_after_path_cap(self):
        actual={'palette':((1,2,3),(4,5,6)),'ids':(1,2)}
        expected={'palette':((1,2,3),),'ids':(1,)}
        result=compact.compare(actual,expected,lambda:False)
        self.assertEqual(result['compared_elements'],8)
        self.assertEqual(result['mismatched_elements'],4)
        result=compact.compare({'palette':tuple((1,2,3) for _ in range(100))},
            {'palette':()},lambda:False)
        self.assertEqual(result['compared_elements'],300)
        self.assertEqual(result['mismatched_elements'],300)
        self.assertEqual(len(result['mismatch_paths']),32)
        self.assertEqual(compact.compare((),None,lambda:False)['mismatched_elements'],1)


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class DiagnosticReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(dir='artifacts')
        cls.root = Path(cls.temporary.name)
        cls.runs = []
        for kind in ('binary','mean','otsu'):
            for local in (False,True):
                recipe = selected(kind,local=local)
                if local:
                    path=cls.root/'card.png'
                    assert cv2.imwrite(str(path),calibrated.fixture())
                    saved=runner.run_native_recipe_path(recipe,path)
                else:
                    gray,rgb=seeds.fixture()
                    saved=runner.run_native_recipe(recipe,torch.tensor(gray,device='cuda'),
                        torch.tensor(rgb,device='cuda'),source_sha256=seeds.SOURCE,
                        coordinate_frame=seeds.FRAME,preparation=seeds.PREPARATION)
                directory=cls.root/(kind+('-local' if local else '-source'))
                archives.export_native_recipe_result(directory,recipe,saved)
                cls.runs.append((recipe,saved,directory))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def changed(self, index=0, stage=None, **outputs):
        recipe,saved,_=self.runs[index]
        stage=stage or factory.STAGE
        node=replace(saved.node(stage),outputs={**dict(saved.node(stage).outputs),**outputs})
        result=replace(saved,nodes=tuple(node if item.stage_id==stage else item for item in saved.nodes))
        directory=self.root/('changed-'+str(len(tuple(self.root.iterdir()))))
        archives.export_native_recipe_result(directory,recipe,result)
        return directory

    def attempt(self,directory,cancel=None):
        calls=[]
        def profile(frame,event,arg):
            if event=='call' and frame.f_code is runner._dispatch.__code__:
                calls.append((frame.f_locals['operation'],set(frame.f_locals['inputs'])))
        previous=sys.getprofile();sys.setprofile(profile)
        try:
            result=replay.replay_native_stage(directory,factory.STAGE,cancel=cancel)
        finally:
            sys.setprofile(previous)
        return result,calls

    def test_actual_source_and_roi_replay_select_only_exact_arrays_compare_all_products_and_preserve_archive(self):
        for recipe,saved,directory in self.runs:
            before={p.relative_to(directory).as_posix():sha256(p.read_bytes()).hexdigest()
                    for p in directory.rglob('*') if p.is_file()}
            inventory,_=archives.read_native_recipe_inventory(directory)
            nodes={node['stage_id']:dict(node['outputs']) for node in inventory['nodes']}
            expected_names={nodes[e.source_stage][e.source_port]['name']
                for e in recipe.dependencies if e.target_stage==factory.STAGE}
            expected_names.add(nodes[factory.STAGE]['diagnostic']['name'])
            selected_names=[]
            def observer(frame,event,arg):
                if event=='call' and frame.f_code is archives.load_native_recipe_result.__code__:
                    selected_names.append(frame.f_locals['array_names'])
            previous=sys.getprofile();sys.setprofile(observer)
            try:
                result=replay.replay_native_stage(directory,factory.STAGE)
            finally:
                sys.setprofile(previous)
            self.assertEqual(result.status,'matched',result.reason)
            self.assertEqual(len(selected_names),1)
            self.assertEqual(len(selected_names[0]),3)
            self.assertEqual(set(selected_names[0]),expected_names)
            self.assertEqual(result.metadata['plan']['selected_bytes'],10*96*128)
            self.assertEqual(result.metadata['preview_port'],'diagnostic')
            self.assertEqual(tuple(result.metadata['coordinate_frame']['shape']),(96,128))
            self.assertTrue(torch.equal(result.outputs['diagnostic'],saved.node(factory.STAGE).outputs['diagnostic']))
            self.assertEqual(result.outputs['metadata'],saved.node(factory.STAGE).outputs['metadata'])
            self.assertEqual(set(result.metadata['comparison']['by_port']),{'diagnostic','metadata'})
            self.assertGreater(result.metadata['comparison']['by_port']['metadata']['compared_elements'],31)
            self.assertFalse(result.scientifically_validated)
            self.assertEqual(before,{p.relative_to(directory).as_posix():sha256(p.read_bytes()).hexdigest()
                for p in directory.rglob('*') if p.is_file()})

    def test_malformed_metadata_and_branch_keys_reject_before_loading(self):
        original=dict(self.runs[0][1].node(factory.STAGE).outputs['metadata'])
        variants=[{**original,'extra':0},{k:v for k,v in original.items() if k!='kind'},
            {**original,'geometry_work':True},{**original,'contour_points':-1},
            {**original,'source_sha256':'0'*64},{**original,'palette_sha256':'A'*64},
            {**original,'line_thickness':6},{**original,'label_ids':(1,1)},
            {**original,'numpy_version':'0.0.0'},{**original,'matplotlib_version':'0.0.0'},
            {**original,'palette_count':4098},{**original,'palette_source_triples':((True,0,0),)},
            {**original,'roi_pixels':1.5}]
        for facts in variants:
            directory=self.changed(metadata=facts)
            with patch.object(archives,'load_native_recipe_result') as load:
                result=replay.replay_native_stage(directory,factory.STAGE)
            self.assertEqual(result.status,'invalid_archive',result.reason)
            self.assertFalse(result.outputs);load.assert_not_called()
        for index,(recipe,saved,_) in enumerate(self.runs):
            for port in ('image','labels'):
                node=replace(saved.node(factory.STAGE),input_keys={**dict(saved.node(factory.STAGE).input_keys),port:'0'*64})
                altered=replace(saved,nodes=tuple(node if n.stage_id==factory.STAGE else n for n in saved.nodes))
                directory=self.root/('keys-'+str(index)+'-'+port)
                archives.export_native_recipe_result(directory,recipe,altered)
                with patch.object(archives,'load_native_recipe_result') as load:
                    result=replay.replay_native_stage(directory,factory.STAGE)
                self.assertEqual(result.status,'invalid_archive',result.reason);load.assert_not_called()

    def test_valid_counter_palette_and_rgb_differences_dispatch_once_and_report_mismatch(self):
        original=self.runs[0][1].node(factory.STAGE).outputs
        facts=dict(original['metadata'])
        palette=[list(c) for c in facts['palette_source_triples']]
        palette[1][0]^=1
        palette=tuple(tuple(c) for c in palette)
        wrong={**facts,'palette_source_triples':palette,
            'palette_sha256':sha256(bytes(c for row in palette for c in row)).hexdigest()}
        image=original['diagnostic'].clone();image[0,0,0]^=1
        for change,port in (({'metadata':{**facts,'topology_work':facts['topology_work']+1}},'metadata'),
                            ({'metadata':wrong},'metadata'),({'diagnostic':image},'diagnostic')):
            result,calls=self.attempt(self.changed(**change))
            self.assertEqual(result.status,'mismatch',result.reason)
            self.assertEqual(calls,[(factory.OPERATION,{'image','labels'})])
            self.assertGreater(result.metadata['comparison']['by_port'][port]['mismatched_elements'],0)
            self.assertEqual(set(result.outputs),{'diagnostic','metadata'})

    def test_actual_label_population_differs_from_recorded_metadata_as_numeric_mismatch(self):
        recipe,saved,_=self.runs[0]
        edge=next(e for e in recipe.dependencies if e.target_stage==factory.STAGE and e.target_port=='labels')
        labels=saved.node(edge.source_stage).outputs[edge.source_port].clone()
        labels[labels==2]=1
        result,calls=self.attempt(self.changed(stage=edge.source_stage,**{edge.source_port:labels}))
        self.assertEqual(result.status,'mismatch',result.reason)
        self.assertEqual(len(calls),1)
        self.assertEqual(result.metadata['diagnostic_input_palette_count'],2)
        self.assertEqual(result.outputs['metadata']['palette_count'],2)

    def test_palette_maximum_and_nested_compact_immutability(self):
        facts=dict(self.runs[0][1].node(factory.STAGE).outputs['metadata'])
        palette,metadata=runner._watershed_diagnostic.watershed_palette(4097,random_seed=0xffffffff)
        facts.update(metadata);facts['label_ids']=tuple(range(1,4097))
        parameters={'random_seed':0xffffffff,'color_scale':'gist_rainbow','line_thickness':5}
        frozen=compact.validate(compact.decode(binding(facts),ValueError),parameters,96*128,ValueError)
        self.assertEqual(len(frozen['palette_source_triples']),4097)
        with self.assertRaises(TypeError):frozen['palette_count']=2
        with self.assertRaises(TypeError):frozen['palette_source_triples'][0][0]=1

    def test_wrong_raster_headers_and_selected_byte_cap_reject_before_loading(self):
        recipe,saved,_=self.runs[0]
        edges={e.target_port:e for e in recipe.dependencies if e.target_stage==factory.STAGE}
        cases=((factory.STAGE,'diagnostic',torch.zeros((95,128,3),device='cuda',dtype=torch.uint8)),
            (edges['image'].source_stage,edges['image'].source_port,torch.zeros((96,128,3),device='cuda',dtype=torch.int32)),
            (edges['labels'].source_stage,edges['labels'].source_port,torch.zeros((96,128),device='cuda',dtype=torch.int64)))
        for stage,port,tensor in cases:
            directory=self.changed(stage=stage,**{port:tensor})
            with patch.object(archives,'load_native_recipe_result') as load:
                result=replay.replay_native_stage(directory,factory.STAGE)
            self.assertEqual(result.status,'invalid_archive',result.reason);load.assert_not_called()
        with patch.object(replay,'MAX_TYPED_SELECTED_BYTES',1),patch.object(archives,'load_native_recipe_result') as load:
            result=replay.replay_native_stage(self.runs[0][2],factory.STAGE)
        self.assertEqual(result.status,'failed',result.reason);self.assertFalse(result.outputs);load.assert_not_called()

    def test_different_recorded_runtime_is_unavailable_before_metadata_current_version_check(self):
        recipe,saved,_=self.runs[0]
        original=saved.node(factory.STAGE)
        facts={**dict(original.outputs['metadata']),'numpy_version':'0.0.0','matplotlib_version':'0.0.0'}
        node=replace(original,outputs={**dict(original.outputs),'metadata':facts})
        altered=replace(saved,runtime={**dict(saved.runtime),'calculation_modules_sha256':'0'*64},
            nodes=tuple(node if item.stage_id==factory.STAGE else item for item in saved.nodes))
        directory=self.root/'other-runtime'
        archives.export_native_recipe_result(directory,recipe,altered)
        with patch.object(archives,'load_native_recipe_result') as load:
            result=replay.replay_native_stage(directory,factory.STAGE)
        self.assertEqual(result.status,'unavailable',result.reason);self.assertFalse(result.outputs);load.assert_not_called()

    def test_empty_actual_segmentation_replays_unchanged_rgb_and_zero_contours(self):
        base=runner.build_native_seed_recipe(seeds.selected_recipe(**{'threshold.binary':{'threshold':255}}))
        recipe=factory.build_native_watershed_diagnostic_recipe(base)
        gray,rgb=seeds.fixture()
        saved=runner.run_native_recipe(recipe,torch.tensor(gray,device='cuda'),torch.tensor(rgb,device='cuda'),
            source_sha256=seeds.SOURCE,coordinate_frame=seeds.FRAME,preparation=seeds.PREPARATION)
        directory=self.root/'empty';archives.export_native_recipe_result(directory,recipe,saved)
        result,calls=self.attempt(directory)
        self.assertEqual(result.status,'matched',result.reason);self.assertEqual(len(calls),1)
        self.assertEqual(result.outputs['metadata']['label_ids'],())
        self.assertEqual(result.outputs['metadata']['contours'],0)
        self.assertTrue(torch.equal(result.outputs['diagnostic'],torch.tensor(rgb,device='cuda')))

    def test_compact_helper_changed_during_final_callback_withholds_products(self):
        original=compact.compare
        changed=[]
        def mutate():
            frame=inspect.currentframe()
            while frame is not None:
                if frame.f_code is replay.replay_native_stage.__code__ and 'comparisons' in frame.f_locals and not changed:
                    compact.compare=lambda actual,expected,poll: original(actual,expected,poll)
                    changed.append(True)
                frame=frame.f_back
            return False
        try:
            result,_=self.attempt(self.runs[0][2],cancel=mutate)
        finally:
            compact.compare=original
        self.assertTrue(changed,result.reason)
        self.assertEqual(result.status,'failed',result.reason);self.assertFalse(result.outputs)

    def test_actual_renderer_resource_limit_withholds_image_during_selected_replay(self):
        original_recipe,saved,_=self.runs[0]
        recipe=controls(original_recipe,line_thickness=32767)
        original=saved.node(factory.STAGE)
        # Deliberately altered expectation: structurally valid, but fresh
        # execution must respect the real geometry limit before comparison.
        facts={**dict(original.outputs['metadata']),'line_thickness':32767}
        node=replace(original,parameters=recipe.parameters_for_runner(factory.STAGE),
            outputs={**dict(original.outputs),'metadata':facts})
        altered=replace(saved,recipe_identity=recipe.identity,
            nodes=tuple(node if item.stage_id==factory.STAGE else item for item in saved.nodes))
        directory=self.root/'resource-limit'
        archives.export_native_recipe_result(directory,recipe,altered)
        result,calls=self.attempt(directory)
        self.assertEqual(result.status,'unavailable',result.reason)
        self.assertEqual(len(calls),1);self.assertFalse(result.outputs)
        self.assertTrue(result.reason)

    def test_cancel_and_every_cuda_input_expected_and_fresh_output_change_withhold_all_products(self):
        directory=self.runs[0][2]
        for group,port in (('inputs','image'),('inputs','labels'),('expected_products','diagnostic'),('outputs','diagnostic')):
            changed=[]
            def mutate():
                frame=inspect.currentframe()
                while frame is not None:
                    if frame.f_code is replay.replay_native_stage.__code__ and group in frame.f_locals and not changed:
                        frame.f_locals[group][port].add_(1);changed.append(True)
                    frame=frame.f_back
                return False
            result,_=self.attempt(directory,cancel=mutate)
            self.assertTrue(changed)
            self.assertEqual(result.status,'failed',result.reason);self.assertFalse(result.outputs)
        def inside():
            frame=inspect.currentframe()
            while frame is not None:
                if frame.f_code is runner._watershed_diagnostic.render_watershed_diagnostic_rgb.__code__:
                    return frame.f_locals.get('canvas') is not None
                frame=frame.f_back
            return False
        result,_=self.attempt(directory,cancel=inside)
        self.assertEqual(result.status,'cancelled',result.reason);self.assertFalse(result.outputs)

    def test_actual_asynchronous_qt_replay_keeps_active_run_and_previews_debug_image(self):
        os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
        from PySide6.QtWidgets import QApplication
        from seedvision.ui.native_recipe_dialog import NativeRecipeDialog
        app=QApplication.instance() or QApplication([])
        recipe,saved,directory=self.runs[1]
        dialog=NativeRecipeDialog(recipe);dialog.show();dialog.result=saved
        dialog._select_stage(factory.STAGE)
        self.assertTrue(dialog.replay_button.isEnabled())
        dialog.start_stage_replay(directory)
        deadline=time.monotonic()+90
        while dialog._worker is not None and time.monotonic()<deadline:
            app.processEvents();time.sleep(.01)
        self.assertIsNone(dialog._worker)
        self.assertEqual(dialog.replay_result.status,'matched',dialog.replay_result.reason)
        self.assertIs(dialog.result,saved)
        self.assertFalse(dialog.preview.pixmap().isNull())
        self.assertIn('metadata',dialog.facts.toPlainText())
        dialog.close();dialog.deleteLater();app.processEvents()
