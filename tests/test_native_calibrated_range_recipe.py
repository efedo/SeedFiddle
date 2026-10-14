"""Complete corrected range-mask workflow with physical observations and replay."""
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from seedvision.benchmarking import native_calibrated_seed_recipe as factory
from seedvision.benchmarking import native_card_recipe as card
from seedvision.benchmarking import native_recipe as runner
from seedvision.benchmarking import native_stage_replay as replay
from seedvision.benchmarking import native_watershed_diagnostic_recipe as diagnostic
from seedvision.benchmarking.checkpoints import CoordinateFrame
from seedvision.benchmarking.recipes import RecipeSpec
from seedvision.export import native_results as archives
from seedvision.export.project_native_graphs import export_project_native_graphs
from seedvision.persistence.native_recipes import native_recipes_from_payload,native_recipes_to_payload
from seedvision.persistence.reference_regions import file_sha256
from seedvision.segmentation.baseline import PipelineAnalysisCache
from tests import test_native_calibrated_seed_recipe as calibrated
from tests import test_native_recipe as seed
from tests.test_native_card_recipe import _parameter
from tests.test_plantcv_range import _reference_custom_range


PREFIX='pcv_seed_range_'
RANGE=PREFIX+'custom_range'
MEDIAN=PREFIX+'median_blur'


def selected_recipe():
    base=calibrated.selected_recipe()
    values={s.operation_id:{p.parameter_id:p.value for p in s.parameters} for s in base.stages}
    recipe=factory.build_native_calibrated_seed_recipe('range')
    recipe=replace(recipe,stages=tuple(replace(s,parameters=tuple(replace(p,
        value=values.get(s.operation_id,{}).get(p.parameter_id,p.value)) for p in s.parameters)) for s in recipe.stages))
    for name,value in (('channel','GRAY'),('lower',[125]),('upper',[255])):
        recipe=_parameter(recipe,RANGE,name,value)
    return recipe


def reference_script():
    script=calibrated.reference_script()
    start=script.index("outputs['median_blur'] = pcv.median_blur")
    end=script.index("outputs['small_object_filter'] = pcv.fill",start)
    return script[:start]+"""outputs['range_mask'], outputs['range_masked_bgr'] = pcv.threshold.custom_range(bgr,[125],[255],channel='GRAY')
outputs['median_blur'] = pcv.median_blur(gray_img=outputs['range_mask'],ksize=3)
outputs['binary_threshold'] = outputs['median_blur']
"""+script[end:]


class RangeCompositionContractTests(unittest.TestCase):
    def test_explicit_range_before_mask_median_local_traits_and_canonical_persistence(self):
        recipe=selected_recipe();runner._validate_recipe(recipe)
        self.assertEqual(len(recipe.stages),24)
        self.assertEqual(RecipeSpec.from_json(recipe.to_json()),recipe)
        self.assertEqual(native_recipes_from_payload(native_recipes_to_payload((recipe,))),(recipe,))
        stages=list(runner._topological_stages(recipe))
        self.assertLess([s.stage_id for s in stages].index(RANGE),[s.stage_id for s in stages].index(MEDIAN))
        median=recipe.stage(MEDIAN)
        self.assertEqual({p.data_type for p in median.ports},{'BinaryMask'})
        connections={(e.source_stage,e.source_port,e.target_stage,e.target_port) for e in recipe.dependencies}
        self.assertIn((factory.CROP,'rgb_image',RANGE,'rgb'),connections)
        self.assertIn((RANGE,'mask',MEDIAN,'gray'),connections)
        self.assertIn((MEDIAN,'image',PREFIX+'small_object_filter','mask'),connections)
        self.assertFalse(any(e.source_stage==RANGE and e.source_port=='masked_rgb' for e in recipe.dependencies))
        for operation in ('analysis.color','analysis.size'):
            stage=next(s for s in recipe.stages if s.operation_id==operation)
            self.assertTrue(any(e.source_stage==factory.CROP and e.source_port=='rgb_image'
                and e.target_stage==stage.stage_id for e in recipe.dependencies))
        for stage in recipe.stages[10:]:
            for port in stage.ports:
                if port.name not in ('scale','metadata'):self.assertEqual(port.semantics['frame'],factory.LOCAL_FRAME)

    def test_old_plans_and_joint_range_admission_boundary(self):
        for pixels in (1,887*887,2080*3120):
            old=factory.retained_bound(pixels)
            self.assertEqual(old,64*pixels+16*min(pixels,factory.peaks.MAX_CANDIDATES)+256*1024*1024)
            self.assertEqual(factory.retained_bound(pixels,range_arm=True),old+16*pixels)
            self.assertEqual(runner._diagnostic_retained_bound(pixels,calibrated=True,range_arm=True),old+19*pixels)
        low,high=1,64_000_000
        while low<high:
            middle=(low+high+1)//2
            if 80*middle+16*min(middle,factory.peaks.MAX_CANDIDATES)+256*1024*1024<=1024*1024*1024:low=middle
            else:high=middle-1
        factory.retained_bound(low,range_arm=True)
        with self.assertRaises(card.card.AutomaticCardResourceError):factory.retained_bound(low+1,range_arm=True)
        debug=diagnostic.build_native_watershed_diagnostic_recipe(selected_recipe())
        runner._validate_recipe(debug)
        self.assertEqual(diagnostic.base_recipe_id(debug.recipe_id),factory.RANGE_RECIPE_ID)

    def test_nine_prior_recipe_identities_are_unchanged(self):
        from seedvision.benchmarking.recipe_catalogue import build_recipe_catalogue
        frozen={
            'binary':'f759988034bc8668364e8f698eef3d211ac4aeb33315a2ebaca6006fd318f193',
            'mean':'071a3be52c91c1ffb821562c435351052c8c89bbb37e9c44612d95f9d359ef67',
            'otsu':'6998ed38c3b9da9413faa34ea57ea21eae722d14ba2f8f91dc7bfcb5ad7e5435',
            'plantcv_seed_binary_working+debug':'7bbe32e959892eb1a05511df2177705729f4ff259ea65c538ca09e6b14882eb8',
            'plantcv_seed_mean_working+debug':'a02c9e2cf1d1e2603cdc04f38f74f301c4de489a8f93bcf1b1ac20d5af0b0081',
            'plantcv_seed_otsu_working+debug':'0d855771ed51c255c709c2308d01f0f3a4f14aba9e0687bebe6c304bc0240e83',
            'plantcv_calibrated_seed_binary+debug':'33fea484ad59f111adb81de8ea6e1392d188a47fd9cafe8a4ffad8b0d08b11cb',
            'plantcv_calibrated_seed_mean+debug':'3045f66c3e0a90cadcc32da1c6451de9de5f7f063ada1a72f77cbde5a5836e39',
            'plantcv_calibrated_seed_otsu+debug':'65189d5e836122bf6af5cc656f585da9c478afb00bf9ca05b71dd82e5ef6a095'}
        actual={kind:factory.build_native_calibrated_seed_recipe(kind) for kind in ('binary','mean','otsu')}
        for kind in ('binary','mean','otsu'):
            for local in (False,True):
                base=actual[kind] if local else runner.build_native_seed_recipe(build_recipe_catalogue().recipe('plantcv_seed_'+kind))
                actual[base.recipe_id+'+debug']=diagnostic.build_native_watershed_diagnostic_recipe(base)
        self.assertEqual({key:sha256(recipe.to_json().encode()).hexdigest() for key,recipe in actual.items()},frozen)


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class CalibratedRangeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary=tempfile.TemporaryDirectory(dir='artifacts');cls.root=Path(cls.temporary.name)
        cls.image=calibrated.fixture();cls.path=cls.root/'card-seeds.png'
        cv2.imwrite(str(cls.path),cls.image);cls.recipe=selected_recipe()
        source=cls.root/'source.npz';out=cls.root/'oracle.npz';obs=cls.root/'oracle.json'
        np.savez(source,bgr=cls.image)
        completed=subprocess.run([str(seed.REFERENCE),'-c',reference_script(),str(source),str(out),str(obs),'range','rgb'],capture_output=True,text=True,timeout=120)
        if completed.returncode:raise AssertionError(completed.stderr)
        with np.load(out,allow_pickle=False) as loaded:cls.expected={name:loaded[name].copy() for name in loaded.files}
        cls.observations=json.loads(obs.read_text(encoding='utf-8'))

    @classmethod
    def tearDownClass(cls):cls.temporary.cleanup()

    def setUp(self):self.cache=PipelineAnalysisCache()

    def run_recipe(self,recipe=None,**kwargs):
        return runner.run_native_recipe_path(recipe or self.recipe,self.path,cache=self.cache,**kwargs)

    def test_full_public_corrected_roi_range_preparation_watershed_and_observations(self):
        run=self.run_recipe()
        self.assertEqual(run.status,'executed',[(n.stage_id,n.status,n.reason) for n in run.nodes])
        expected=self.expected
        for stage,port,key in ((RANGE,'mask','range_mask'),(RANGE,'masked_rgb','range_masked_bgr'),
            (MEDIAN,'image','median_blur'),(PREFIX+'small_object_filter','mask','small_object_filter'),
            (PREFIX+'erode','mask','erode'),(PREFIX+'dilate','mask','dilate'),
            (PREFIX+'ws_distance_map','distance','distance'),(PREFIX+'ws_local_maxima','coordinates','coordinates'),
            (PREFIX+'ws_local_maxima','peaks','peaks'),(PREFIX+'ws_markers','markers','markers'),
            (PREFIX+'ws_invert_distance','negative_distance','negative_distance'),(PREFIX+'ws_masked_flood','labels','labels')):
            array=expected[key][...,::-1] if port=='masked_rgb' else expected[key]
            np.testing.assert_array_equal(run.node(stage).outputs[port].cpu(),array)
        np.testing.assert_array_equal(run.node(factory.CROP).outputs['rgb_image'].cpu(),expected['seed_rgb'])
        size=run.node(PREFIX+'analyze_size').outputs
        np.testing.assert_array_equal(size['diagnostic'].cpu(),expected['size_diagnostic_rgb'])
        calibrated.CalibratedSeedTests.assert_observations(self,run.node(factory.SCALED_SIZE).outputs['observations'],self.observations['__size_observations__'])
        for label in run.node(PREFIX+'analyze_color').outputs['observations'].labels:
            for histogram in label.histograms:np.testing.assert_array_equal(histogram.values,self.observations[label.sample][histogram.variable]['value'])
        self.assertEqual(run.labels_coordinate_frame.shape,(96,128))
        self.assertEqual(run.node(factory.CROP).outputs['metadata']['local_to_parent_xy'],(600,260))
        self.assertTrue(all(t.is_cuda for n in run.nodes for t in runner._iter_nested_tensors(n.outputs)))

    def test_five_modes_use_exact_corrected_roi_pixels_and_local_range_replay(self):
        first=self.run_recipe();rgb=first.node(factory.CROP).outputs['rgb_image'].cpu().numpy()
        cases=[]
        for mode in ('RGB','HSV','LAB','CMYK','GRAY'):
            count=4 if mode=='CMYK' else 1 if mode=='GRAY' else 3
            cases.append(dict(name=mode,channel=mode,lower=[0]*count,upper=[255]*count))
        # Exact single-colour RGB selection and a partial GRAY range also prove
        # the controls affect the corrected crop, rather than a pre-crop image.
        background=rgb[0,0,::-1].tolist()
        cases.extend((dict(name='RGB-partial',channel='RGB',lower=background,upper=background),
            dict(name='GRAY-partial',channel='GRAY',lower=[125],upper=[255])))
        public=_reference_custom_range(np.ascontiguousarray(rgb[...,::-1]),cases)
        for case in cases:
            recipe=self.recipe
            for name,value in case.items():
                if name!='name':recipe=_parameter(recipe,RANGE,name,value)
            run=self.run_recipe(recipe);outputs=run.node(RANGE).outputs
            np.testing.assert_array_equal(outputs['mask'].cpu(),public[case['name']+'_mask'])
            np.testing.assert_array_equal(outputs['masked_rgb'].cpu(),public[case['name']+'_masked'][...,::-1])
            if case['name'].endswith('-partial'):
                self.assertTrue(0<torch.count_nonzero(outputs['mask']).item()<96*128)
            directory=self.root/(case['name']+'-archive')
            archives.export_native_recipe_result(directory,recipe,run)
            before={p.name:sha256(p.read_bytes()).hexdigest() for p in directory.iterdir() if p.is_file()}
            result=replay.replay_native_stage(directory,RANGE)
            self.assertEqual(result.status,'matched',result.reason)
            self.assertEqual(result.metadata['frame_scope'],'calibrated_seed_roi')
            self.assertEqual(result.metadata['plan']['pixels'],96*128)
            self.assertEqual(result.metadata['plan']['selected_bytes'],7*96*128)
            self.assertEqual(result.metadata['frame_lineage']['local_to_parent_xy'],(600,260))
            self.assertEqual(before,{p.name:sha256(p.read_bytes()).hexdigest() for p in directory.iterdir() if p.is_file()})

    def test_range_median_roi_scale_cache_boundaries_and_empty_run_drop_old_traits(self):
        first=self.run_recipe();self.assertEqual(first.status,'executed')
        self.run_recipe();self.assertFalse(self.cache.last_computed_nodes)
        upper=_parameter(self.recipe,RANGE,'upper',[254]);self.run_recipe(upper)
        self.assertIn(factory.CROP,self.cache.last_reused_nodes)
        self.assertIn(RANGE,self.cache.last_computed_nodes);self.assertIn(MEDIAN,self.cache.last_computed_nodes)
        median=_parameter(upper,MEDIAN,'ksize',5);self.run_recipe(median)
        self.assertIn(RANGE,self.cache.last_reused_nodes);self.assertIn(MEDIAN,self.cache.last_computed_nodes)
        scaled=_parameter(self.recipe,card.SCALE,'color_chip_size',[7,13]);self.run_recipe(scaled)
        self.assertEqual(set(self.cache.last_computed_nodes),{RANGE,MEDIAN,*[n.stage_id for n in first.nodes if n.stage_id.startswith(PREFIX) and n.stage_id not in (RANGE,MEDIAN)],card.SCALE,factory.SCALED_SIZE})
        self.run_recipe(scaled);units=_parameter(scaled,card.SCALE,'color_chip_size',[7,14]);self.run_recipe(units)
        self.assertEqual(set(self.cache.last_computed_nodes),{card.SCALE,factory.SCALED_SIZE})
        roi=_parameter(units,factory.CROP,'seed_roi_xyxy',[599,259,729,357]);self.run_recipe(roi)
        self.assertIn(card.AFFINE,self.cache.last_reused_nodes);self.assertIn(factory.CROP,self.cache.last_computed_nodes)
        empty=_parameter(_parameter(self.recipe,RANGE,'lower',[255]),RANGE,'upper',[0]);run=self.run_recipe(empty)
        self.assertEqual(torch.count_nonzero(run.node(RANGE).outputs['mask']).item(),0)
        self.assertFalse(run.node(PREFIX+'analyze_size').outputs)
        self.assertFalse(run.node(PREFIX+'analyze_color').outputs)
        self.assertFalse(run.node(factory.SCALED_SIZE).outputs)
        recovered=self.run_recipe();self.assertEqual(recovered.status,'executed')

    def test_admission_before_upload_cached_preparation_and_extra_input_backing(self):
        pixels=self.image.shape[0]*self.image.shape[1]
        for recipe in (self.recipe,diagnostic.build_native_watershed_diagnostic_recipe(self.recipe)):
            for cached in (False,True):
                self.cache=PipelineAnalysisCache()
                if cached:self.assertEqual(self.run_recipe(recipe).status,'executed')
                with patch.object(factory,'retained_bound',side_effect=card.card.AutomaticCardResourceError('range joint cap')) as admission:
                    with patch.object(torch,'as_tensor',side_effect=AssertionError('no upload')):
                        with self.assertRaisesRegex(card.card.AutomaticCardResourceError,'range joint cap'):self.run_recipe(recipe)
                    admission.assert_called_once_with(pixels,range_arm=True)
        p=96*128;gray_back=torch.zeros(8*p,dtype=torch.uint8,device='cuda');rgb_back=torch.zeros(6*p,dtype=torch.uint8,device='cuda')
        gray=gray_back[:p].view(96,128);rgb=rgb_back[:3*p].view(96,128,3)
        bound=factory.retained_bound(p,range_arm=True)
        with patch.object(runner,'MAX_RETAINED_CUDA_BYTES',bound+10*p-1),patch.object(runner,'_dispatch') as dispatch:
            with self.assertRaisesRegex(runner.NativeRecipeError,'source backing'):
                runner.run_native_recipe(self.recipe,gray,rgb,source_sha256='a'*64,
                    coordinate_frame=CoordinateFrame('test.source',(96,128),'top-left','x-right,y-down','pixel'),preparation=seed.PREPARATION)
            dispatch.assert_not_called()

    def test_debug_branch_and_batch_export_identity(self):
        recipe=diagnostic.build_native_watershed_diagnostic_recipe(self.recipe)
        run=self.run_recipe(recipe)
        self.assertEqual(run.status,'executed',[(n.stage_id,n.status,n.reason) for n in run.nodes])
        self.assertEqual(tuple(run.node(diagnostic.STAGE).outputs['diagnostic'].shape),(96,128,3))
        root=self.root/'batch'
        ledger=export_project_native_graphs((),self.path,root,expected_source_sha256=file_sha256(self.path),native_recipes=(self.recipe,))
        rows=json.loads(ledger.read_bytes())['graphs']
        self.assertEqual([row['status'] for row in rows],['executed'])
        self.assertEqual(rows[0]['recipe_identity'],self.recipe.identity.to_dict())

    def test_actual_saved_project_load_and_companion_arrays(self):
        from seedvision.persistence.project_analysis import ProjectAnalysisStore,ProjectImageSpec
        from seedvision.persistence.analysis_settings import analysis_settings_profile_from_graph
        from seedvision.pipeline import build_default_pipeline
        from tests.test_native_result_export import _output
        workspace=self.root/'range-workspace';workspace.mkdir();(workspace/'config').mkdir()
        source=workspace/'range-seeds.png';shutil.copyfile(self.path,source)
        shutil.copyfile(Path(__file__).parents[1]/'config/traits.json',workspace/'config/traits.json')
        store=ProjectAnalysisStore(workspace)
        master=store.capture_and_save(analysis_settings=analysis_settings_profile_from_graph(build_default_pipeline()),
            images=(ProjectImageSpec(source,self.image.shape[:2]),),native_recipes=(self.recipe,),
            destination=workspace/'range-seeds.seedfiddle-project.json')
        document=store.load(master).document
        self.assertEqual(document.native_recipes,(self.recipe,))
        ledger=export_project_native_graphs((),source,self.root/'range-project-runs',
            expected_source_sha256=document.images[0].source.sha256,project_root=workspace,native_recipes=document.native_recipes)
        row=json.loads(ledger.read_bytes())['graphs'][0];self.assertEqual(row['status'],'executed',row['reason'])
        report,arrays=archives.load_native_recipe_result(ledger.parent/Path(row['report']).parent)
        expected=self.run_recipe()
        for stage,port in ((factory.CROP,'rgb_image'),(RANGE,'mask'),(RANGE,'masked_rgb'),
                (MEDIAN,'image'),(PREFIX+'ws_masked_flood','labels')):
            np.testing.assert_array_equal(arrays[_output(report,stage,port)['name']],expected.node(stage).outputs[port].cpu())
        self.assertEqual(row['recipe_identity'],self.recipe.identity.to_dict())
        self.assertFalse(report['scientifically_validated']);self.assertEqual(row['historical_stage_runs'],[])

    def test_final_callback_mutation_runtime_change_and_cancellation_withhold_run(self):
        current=self.run_recipe();self.assertEqual(current.status,'executed')
        counter={'calls':0}
        def count():counter['calls']+=1;return False
        self.run_recipe(cancel=count)
        prefix='reference.recipe:'+self.recipe.recipe_id+':'
        self.cache.values['unrelated.production']='preserved'
        def on_last(action):
            calls=0
            def callback():
                nonlocal calls
                calls+=1
                if calls==counter['calls']:action()
                return False
            return callback
        def mutate():current.node(RANGE).outputs['masked_rgb'][0,0,0]^=255
        with self.assertRaisesRegex(runner.NativeRecipeError,'changed during execution'):
            self.run_recipe(cancel=on_last(mutate))
        self.assertFalse(any(key.startswith(prefix) for key in self.cache.values))
        self.assertEqual(self.cache.values['unrelated.production'],'preserved')
        self.assertEqual(self.run_recipe().status,'executed')
        original=runner._range._range_controls
        def change_runtime():runner._range._range_controls=lambda *args:None
        try:
            with self.assertRaisesRegex(runner.NativeRecipeError,'Loaded|changed|differs'):
                self.run_recipe(cancel=on_last(change_runtime))
        finally:runner._range._range_controls=original
        self.assertFalse(any(key.startswith(prefix) for key in self.cache.values))
        with self.assertRaises(runner.NativeRecipeCancelled):self.run_recipe(cancel=lambda:True)
        self.assertEqual(self.run_recipe().status,'executed')


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class CalibratedRangeDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
        from PySide6.QtWidgets import QApplication
        cls.app=QApplication.instance() or QApplication([])

    def wait(self,dialog):
        deadline=time.monotonic()+40
        while dialog._worker is not None and time.monotonic()<deadline:
            self.app.processEvents();time.sleep(.01)
        self.app.processEvents();self.assertIsNone(dialog._worker)

    def test_async_local_range_and_mask_median_replay_edit_keep_reopen_and_picker(self):
        from seedvision.ui.native_recipe_dialog import NativeRecipeDialog
        from seedvision.ui.reference_graph_dialog import ReferenceGraphDialog
        kept=[]
        with tempfile.TemporaryDirectory(dir='artifacts') as temporary:
            root=Path(temporary);path=root/'seeds.png';cv2.imwrite(str(path),calibrated.fixture())
            dialog=NativeRecipeDialog(selected_recipe(),keep_in_project=kept.append);dialog.show()
            try:
                dialog.set_source_path(path);dialog.start_run();self.wait(dialog)
                self.assertEqual(dialog.result.status,'executed',dialog.status_label.text())
                current=dialog.result;archives.export_native_recipe_result(root/'run',dialog.recipe,current)
                for stage in (RANGE,MEDIAN):
                    dialog._select_stage(stage)
                    self.assertIn('Parent xy = local xy + (600, 260)',dialog.facts.toPlainText())
                    dialog.start_stage_replay(root/'run');self.wait(dialog)
                    self.assertIs(dialog.result,current)
                    self.assertIn('matched',dialog.status_label.text().lower())
                    self.assertFalse(dialog.preview.pixmap().isNull())
                    self.assertIn('calibrated_seed_roi',dialog.facts.toPlainText())
                    self.assertEqual(dialog.replay_result.metadata['frame_lineage']['local_to_parent_xy'],(600,260))
                dialog.inspector.parameter_changed.emit(RANGE,'upper','[254]')
                dialog.start_run();self.wait(dialog)
                self.assertEqual(dialog.result.status,'executed',dialog.status_label.text())
                self.assertTrue(dialog.result.node(factory.CROP).reused)
                self.assertFalse(dialog.result.node(RANGE).reused)
                dialog._keep_recipe_in_project();self.assertEqual(len(kept),1)
                self.assertEqual(kept[0].parameters_for_runner(RANGE)['upper'],[254])
            finally:dialog.reject();self.wait(dialog);dialog.deleteLater()
            reopened=NativeRecipeDialog(kept[0])
            try:self.assertEqual(reopened.recipe,kept[0])
            finally:reopened.reject();reopened.deleteLater()
        reference=ReferenceGraphDialog(native_recipes=tuple(kept))
        try:
            with patch('seedvision.ui.reference_graph_dialog.QInputDialog.getItem',return_value=('range',True)) as picker, \
                    patch.object(reference,'_open_working_native_recipe') as opened:
                reference._calibrated_seed_button.click()
            self.assertIn('range',picker.call_args.args[3])
            self.assertEqual(opened.call_args.args[0],factory.build_native_calibrated_seed_recipe('range'))
        finally:reference.reject();reference.deleteLater()


if __name__=='__main__':unittest.main()
