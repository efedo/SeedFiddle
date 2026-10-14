"""Pinned codec grayscale and versioned desktop/project reader selection."""
from dataclasses import replace
from hashlib import sha256
import itertools
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

from seedvision.benchmarking import native_image_inputs as inputs
from seedvision.benchmarking import native_recipe as runner
from seedvision.benchmarking import native_reader_recipe as reader
from seedvision.benchmarking import native_median_roi_recipe as median
from seedvision.benchmarking import native_fill_roi_recipe as fill
from seedvision.benchmarking import native_morphology_roi_recipe as morphology
from seedvision.benchmarking import native_stage_replay as replay
from seedvision.benchmarking.recipes import RecipeSpec
from seedvision.export import native_results as archives
from seedvision.persistence.native_recipes import native_recipes_from_payload, native_recipes_to_payload
from seedvision.segmentation.baseline import PipelineAnalysisCache
from tests.test_native_image_inputs import fixtures, REFERENCE, READER_SHA
from tests.test_native_recipe import fixture, selected_recipe
from tests.test_morphology_roi_recipe import _change

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')


def _base():
    return runner.build_native_seed_recipe(selected_recipe())


def _add(recipe, control):
    if control == 'reader':
        return _change(reader.build_native_reader_recipe(recipe),reader.OPERATION,gray_source_mode='codec-gray')
    if control == 'median':
        return _change(median.build_native_median_roi_recipe(recipe),median.OPERATION,
                       ksize=[2,3],roi_bounds_xyxy=[20,20,70,70])
    if control == 'fill':
        recipe = (median.add_fill_roi(recipe) if median.is_median_roi_recipe(recipe)
                  else fill.build_native_fill_roi_recipe(recipe))
        return _change(recipe,fill.OPERATION,size=80,roi_bounds_xyxy=[21,21,71,71])
    recipe = (median.add_morphology_roi(recipe) if median.is_median_roi_recipe(recipe)
              else fill.add_morphology_roi(recipe) if recipe.recipe_id in fill.RECIPE_IDS
              else morphology.build_native_morphology_roi_recipe(recipe))
    recipe = _change(recipe,'morphology.erode',ksize=2,roi_bounds_xyxy=[22,22,72,72])
    return _change(recipe,'morphology.dilate',iterations=2,roi_bounds_xyxy=[23,23,73,73])


ORACLE = r'''
from hashlib import sha256
from importlib.metadata import version
import importlib,json,sys
from pathlib import Path
import cv2,numpy as np
from plantcv import plantcv as pcv
assert version('plantcv')=='4.11.3'
assert version('opencv-python')=='4.11.0.86'
assert version('numpy')=='2.2.6'
reader=importlib.import_module('plantcv.plantcv.readimage')
assert sha256(Path(reader.__file__).read_bytes()).hexdigest()==sys.argv[3]
pcv.params.debug=None;pcv.params.verbose=False
arrays={};rows=[]
for i,name in enumerate(json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))):
    row={'name':Path(name).name}
    try:
        gray,_,_=pcv.readimage(filename=name,mode='gray')
        bgr,_,_=pcv.readimage(filename=name,mode='rgb')
        arrays[f'gray_{i}']=gray;arrays[f'bgr_{i}']=bgr
        row.update(success=True,shape=list(gray.shape),dtype=str(gray.dtype))
    except Exception as error:
        row.update(success=False,error_type=type(error).__name__)
    rows.append(row)
np.savez(sys.argv[2]+'.npz',**arrays)
Path(sys.argv[2]+'.json').write_text(json.dumps(rows),encoding='utf-8')
'''


class ReaderContractTests(unittest.TestCase):
    def test_all_24_ids_have_additive_reader_versions_and_calibrated_controls_reject(self):
        self.assertEqual(len(reader.BASE_IDS),24)
        for identity in reader.BASE_IDS:
            old=median.canonical_base(identity);before=old.to_dict()
            for original in (old,median.build_native_median_roi_recipe(old)):
                new=reader.build_native_reader_recipe(original)
                runner._validate_recipe(new)
                self.assertEqual(new.recipe_id,old.recipe_id)
                self.assertEqual(new.dependencies,old.dependencies)
                self.assertEqual(new.layout,old.layout)
                self.assertEqual(reader.selected_mode(new),'color-derived')
                self.assertEqual(median.is_median_roi_recipe(new),median.is_median_roi_recipe(original))
            self.assertEqual(old.to_dict(),before)
        from seedvision.benchmarking.native_calibrated_seed_recipe import build_native_calibrated_seed_recipe
        calibrated=build_native_calibrated_seed_recipe('binary')
        with self.assertRaises(ValueError):reader.build_native_reader_recipe(calibrated)
        new=reader.build_native_reader_recipe(_base())
        for bad in (replace(new,recipe_version='1.0.0'),replace(new,stages=_base().stages),
                    replace(calibrated,recipe_version=reader.RECIPE_VERSION)):
            with self.assertRaises(runner.NativeRecipeError):runner._validate_recipe(bad)
        with self.assertRaises(ValueError):reader.build_native_reader_recipe(new)

    def test_all_24_action_orders_preserve_selected_parameters_layout_and_project_versions(self):
        old=_base();old=replace(old,layout=tuple(replace(p,x=p.x+123,y=p.y+456) for p in old.layout))
        expected=None
        for order in itertools.permutations(('reader','median','fill','morphology')):
            recipe=old
            for control in order:recipe=_add(recipe,control)
            runner._validate_recipe(recipe)
            self.assertEqual(reader.selected_mode(recipe),'codec-gray')
            self.assertEqual(recipe.layout,old.layout)
            if expected is None:expected=recipe.identity
            self.assertEqual(recipe.identity,expected)
        versions=(old,median.build_native_median_roi_recipe(old),reader.build_native_reader_recipe(old),recipe)
        self.assertEqual(native_recipes_from_payload(native_recipes_to_payload(versions)),versions)
        for selected in versions:self.assertEqual(RecipeSpec.from_json(selected.to_json()).identity,selected.identity)

    def test_invalid_modes_mismatched_decodes_and_owned_array_caps(self):
        raw=cv2.imencode('.png',np.zeros((3,7,3),np.uint8))[1].tobytes()
        for value in (None,False,'gray','codec-gray '):
            with self.assertRaises(runner.NativeRecipeError):runner._gray_preparation_prefix(value)
        with self.assertRaises(inputs.ImageInputError):inputs.decode_gray(raw,(7,3))
        with patch.object(inputs,'MAX_DECODED_BYTES',83):
            with self.assertRaises(inputs.ImageInputError):inputs.decoded_bound(21,color=True,gray=True)
            inputs.decoded_bound(21,color=True,gray=False)
        for shape in ((1,1),(1,7),(9,1)):
            encoded=cv2.imencode('.png',np.zeros((*shape,3),np.uint8))[1].tobytes()
            self.assertEqual(inputs.decode_gray(encoded,shape).shape,shape)


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class ReaderIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.root=Path(self.temporary.name)
        self.path=self.root/'source.png';gray,_=fixture()
        self.bgr=np.repeat(gray[...,None],3,axis=2)
        self.assertTrue(cv2.imwrite(str(self.path),self.bgr))

    def tearDown(self):self.temporary.cleanup()

    def test_pinned_public_gray_decoder_matrix_and_mode_changes_pixels(self):
        self.assertTrue(REFERENCE.is_file(),'Pinned public interpreter is required.')
        directory=self.root/'formats';directory.mkdir();paths=fixtures(directory)
        manifest=self.root/'paths.json';manifest.write_text(json.dumps([str(p) for p in paths]),encoding='utf-8')
        output=self.root/'oracle'
        process=subprocess.run([str(REFERENCE),'-c',ORACLE,str(manifest),str(output),READER_SHA],
                               capture_output=True,text=True,timeout=120)
        self.assertEqual(process.returncode,0,process.stderr)
        rows=json.loads(output.with_suffix('.json').read_text(encoding='utf-8'))
        different=False
        with np.load(output.with_suffix('.npz'),allow_pickle=False) as expected:
            for i,(path,row) in enumerate(zip(paths,rows,strict=True)):
                with self.subTest(case=path.name):
                    if path.name=='seed-色.png':
                        self.assertFalse(row['success'])
                        actual=runner.prepare_native_recipe_inputs(path,gray_source_mode='codec-gray')
                        np.testing.assert_array_equal(actual.gray.cpu().numpy(),expected['gray_0'])
                    elif not row['success']:
                        with self.assertRaises((runner.NativeRecipeError,OSError)):
                            runner.prepare_native_recipe_inputs(path,gray_source_mode='codec-gray')
                    else:
                        actual=runner.prepare_native_recipe_inputs(path,gray_source_mode='codec-gray')
                        np.testing.assert_array_equal(actual.gray.cpu().numpy(),expected[f'gray_{i}'])
                        np.testing.assert_array_equal(actual.rgb.cpu().numpy(),expected[f'bgr_{i}'][...,::-1])
                        self.assertEqual(actual.gray.dtype,torch.uint8)
                        self.assertEqual(tuple(actual.gray.shape),tuple(row['shape']))
                        derived=runner.prepare_native_recipe_inputs(path)
                        different |= not torch.equal(actual.gray,derived.gray)
                        np.testing.assert_array_equal(derived.gray.cpu().numpy(),cv2.cvtColor(expected[f'bgr_{i}'],cv2.COLOR_BGR2GRAY))
        self.assertTrue(different,'A codec fixture must prove the grayscale control affects pixels.')

    def test_mode_switch_reuses_rgb_same_mode_reuses_nodes_and_roi_edits_reuse_source(self):
        old=_base();recipe=reader.build_native_reader_recipe(old);cache=PipelineAnalysisCache()
        first=runner.run_native_recipe_path(recipe,self.path,cache=cache)
        prepared=cache.values['reference.prepared'][1]
        codec=_change(recipe,reader.OPERATION,gray_source_mode='codec-gray')
        original=cv2.imdecode;calls=[]
        def decode(raw,flags):calls.append(flags);return original(raw,flags)
        # Patching changes decoder code identity; establish both modes under that identity.
        with patch.object(cv2,'imdecode',decode):
            runner.run_native_recipe_path(recipe,self.path,cache=cache)
            baseline=cache.values['reference.prepared'][1];calls.clear()
            changed=runner.run_native_recipe_path(codec,self.path,cache=cache)
            new=cache.values['reference.prepared'][1]
            self.assertEqual(calls,[inputs.GRAY_DECODE_FLAGS]);self.assertIs(new.rgb,baseline.rgb)
            self.assertIsNot(new.gray,baseline.gray)
            again=runner.run_native_recipe_path(codec,self.path,cache=cache)
            self.assertIs(cache.values['reference.prepared'][1],new)
            self.assertEqual(len(cache.last_reused_nodes),len(codec.stages))
            source=next(s.stage_id for s in codec.stages if s.operation_id==reader.OPERATION)
            self.assertIs(changed.node(source).outputs['gray_image'],again.node(source).outputs['gray_image'])
            calls.clear();runner.run_native_recipe_path(recipe,self.path,cache=cache)
            self.assertEqual(calls,[]);self.assertIs(cache.values['reference.prepared'][1].rgb,new.rgb)
        med=median.build_native_median_roi_recipe(codec)
        runner.run_native_recipe_path(med,self.path,cache=cache)
        runner.run_native_recipe_path(_change(med,median.OPERATION,roi_bounds_xyxy=[20,20,70,70]),self.path,cache=cache)
        self.assertEqual(cache.last_reused_nodes,(next(s.stage_id for s in med.stages if s.operation_id==reader.OPERATION),))
        self.assertEqual(first.recipe_identity.recipe_id,old.recipe_id)
        self.assertEqual(prepared.source_sha256,sha256(self.path.read_bytes()).hexdigest())

    def test_prepared_mode_facts_are_required_and_calibrated_preparation_rejects_codec(self):
        recipe=reader.build_native_reader_recipe(_base())
        prepared=runner.prepare_native_recipe_inputs(self.path)
        codec=_change(recipe,reader.OPERATION,gray_source_mode='codec-gray')
        for selected,facts in ((codec,prepared.preparation),(recipe,{'gray':'declared gray','rgb':'declared RGB'}),
                               (recipe,dict(prepared.preparation,gray=prepared.preparation['gray']+' codec-gray'))):
            with self.assertRaisesRegex(runner.NativeRecipeError,'grayscale facts'):
                runner.run_native_recipe(selected,prepared.gray,prepared.rgb,source_sha256=prepared.source_sha256,
                    coordinate_frame=prepared.coordinate_frame,preparation=facts)
        with self.assertRaisesRegex(runner.NativeRecipeError,'uncalibrated'):
            runner.prepare_native_recipe_inputs(self.path,gray_source_mode='codec-gray',_calibrated=True)
        from seedvision.benchmarking.native_calibrated_seed_recipe import build_native_calibrated_seed_recipe
        calibrated=build_native_calibrated_seed_recipe('binary')
        self.assertFalse(any(p.parameter_id==reader.PARAMETER for s in calibrated.stages for p in s.parameters))
        self.assertEqual(reader.selected_mode(calibrated),'color-derived')

    def test_codec_shape_and_resources_reject_before_second_decode_or_upload(self):
        calls=[]
        def profile(frame,event,arg):
            if event=='call':
                if frame.f_code is runner._preparation_bound.__code__:calls.append('graph bound')
                if frame.f_code is inputs.decoded_bound.__code__:calls.append('owned bound')
                if frame.f_code is inputs.decode_gray.__code__:calls.append('gray decode')
        sys.setprofile(profile)
        try:
            with patch.object(torch,'as_tensor',side_effect=AssertionError('Reached upload')):
                with self.assertRaisesRegex(AssertionError,'Reached upload'):
                    runner.prepare_native_recipe_inputs(self.path,gray_source_mode='codec-gray')
        finally:sys.setprofile(None)
        self.assertEqual(calls,['graph bound','owned bound','gray decode'])
        original=cv2.imdecode
        def mismatch(raw,flags):
            decoded=original(raw,flags)
            return decoded[:-1] if flags==inputs.GRAY_DECODE_FLAGS else decoded
        with patch.object(cv2,'imdecode',mismatch),patch.object(torch,'as_tensor',side_effect=AssertionError('No upload')):
            with self.assertRaisesRegex(runner.NativeRecipeError,'same contiguous uint8 frame'):
                runner.prepare_native_recipe_inputs(self.path,gray_source_mode='codec-gray')
        calls.clear();sys.setprofile(profile)
        try:
            with patch.object(runner,'MAX_RASTER_PIXELS',self.bgr.shape[0]*self.bgr.shape[1]-1):
                with self.assertRaisesRegex(runner.NativeRecipeError,'raster limit'):
                    runner.prepare_native_recipe_inputs(self.path,gray_source_mode='codec-gray')
        finally:sys.setprofile(None)
        self.assertEqual(calls,['graph bound'])

    def test_private_codec_cpu_and_cuda_gray_mutations_withhold_publication(self):
        original_decode=cv2.imdecode;original_tensor=torch.as_tensor
        for target in ('cpu_pixels','cpu_shape','cuda_gray'):
            for raises in (False,True):
                cache=PipelineAnalysisCache();captured=[];fired=False
                def decode(raw,flags):
                    value=original_decode(raw,flags)
                    if flags==inputs.GRAY_DECODE_FLAGS and target.startswith('cpu'):captured.append(value)
                    return value
                def upload(value,**kwargs):
                    result=original_tensor(value,**kwargs)
                    if target=='cuda_gray' and value.ndim==2:captured.append(result)
                    return result
                def mutate():
                    nonlocal fired
                    if captured and not fired:
                        fired=True
                        if target=='cpu_pixels':captured[0][0,0]^=1
                        elif target=='cpu_shape':captured[0].shape=(128,96)
                        else:captured[0].add_(1)
                        if raises:raise ValueError('Caller after private mutation')
                        return True
                    return False
                with self.subTest(target=target,raises=raises),patch.object(cv2,'imdecode',decode), \
                     patch.object(torch,'as_tensor',upload):
                    with self.assertRaises(runner.NativePreparationMutation):
                        runner.prepare_native_recipe_inputs(self.path,gray_source_mode='codec-gray',cache=cache,cancel=mutate)
                self.assertTrue(fired);self.assertNotIn('reference.prepared',cache.values)

    def test_cancelled_mode_replacement_preserves_valid_prior_and_mutation_evicts_it(self):
        cache=PipelineAnalysisCache();old=runner.prepare_native_recipe_inputs(self.path,cache=cache)
        entry=cache.values['reference.prepared'];decoded=[];original=cv2.imdecode
        def decode(raw,flags):value=original(raw,flags);decoded.append(value);return value
        # Hold the decoder identity constant for both preparations.
        with patch.object(cv2,'imdecode',decode):
            old=runner.prepare_native_recipe_inputs(self.path,cache=cache);entry=cache.values['reference.prepared'];decoded.clear()
            def stop():return bool(decoded)
            with self.assertRaises(runner.NativeRecipeCancelled):
                runner.prepare_native_recipe_inputs(self.path,gray_source_mode='codec-gray',cache=cache,cancel=stop)
            self.assertIs(cache.values['reference.prepared'],entry)
            decoded.clear()
            def mutate():
                if decoded:old.rgb.add_(1);raise ValueError('Caller after cached RGB mutation')
                return False
            with self.assertRaises(runner.NativePreparationMutation):
                runner.prepare_native_recipe_inputs(self.path,gray_source_mode='codec-gray',cache=cache,cancel=mutate)
            self.assertNotIn('reference.prepared',cache.values)

    def test_both_reader_versions_archive_and_replay_without_origin(self):
        recipe=_change(reader.build_native_reader_recipe(_base()),reader.OPERATION,gray_source_mode='codec-gray')
        for selected in (recipe,_add(recipe,'median')):
            result=runner.run_native_recipe_path(selected,self.path)
            stage=next(s.stage_id for s in selected.stages if s.operation_id==median.OPERATION)
            saved=self.root/selected.recipe_version;archives.export_native_recipe_result(saved,selected,result)
            before={str(p.relative_to(saved)):p.read_bytes() for p in saved.rglob('*') if p.is_file()}
            raw=self.path.read_bytes();self.path.unlink()
            calls=[]
            def profile(frame,event,arg):
                if event=='call' and frame.f_code is inputs.source_identity.__code__:calls.append('source read')
                if event=='call' and frame.f_code is runner._dispatch.__code__:calls.append(frame.f_locals['operation'])
            sys.setprofile(profile)
            try:
                actual=replay.replay_native_stage(saved,stage)
            finally:sys.setprofile(None);self.path.write_bytes(raw)
            self.assertEqual(actual.status,'matched',actual.reason)
            self.assertEqual(calls,[median.OPERATION])
            torch.testing.assert_close(actual.outputs['image'],result.node(stage).outputs['image'])
            self.assertEqual(before,{str(p.relative_to(saved)):p.read_bytes() for p in saved.rglob('*') if p.is_file()})

    def test_desktop_all_orders_keep_reopen_and_reader_worker_locks(self):
        from PySide6.QtWidgets import QApplication
        from seedvision.ui.native_recipe_dialog import NativeRecipeDialog
        app=QApplication.instance() or QApplication([]);expected=None;dialog=None;kept=[]
        try:
            for order in itertools.permutations(('reader','median','fill','morphology')):
                candidate=NativeRecipeDialog(_base(),keep_in_project=kept.append)
                moved=candidate.recipe.stages[0].stage_id;candidate.canvas.node_items[moved].setPos(123,456)
                for control in order:
                    getattr(candidate,'reader_button' if control=='reader' else control+'_roi_button').click()
                    if control=='reader':
                        stage=next(s.stage_id for s in candidate.recipe.stages if s.operation_id==reader.OPERATION)
                        candidate._parameter_changed(stage,reader.PARAMETER,'codec-gray')
                    else:
                        operations={'median':(median.OPERATION,),'fill':(fill.OPERATION,),'morphology':morphology.OPERATIONS}[control]
                        for operation in operations:
                            stage=next(s.stage_id for s in candidate.recipe.stages if s.operation_id==operation)
                            candidate._parameter_changed(stage,median.PARAMETER,'[20,20,70,70]')
                if expected is None:expected=candidate.recipe.identity
                self.assertEqual(candidate.recipe.identity,expected)
                self.assertEqual(reader.selected_mode(candidate.recipe),'codec-gray')
                self.assertEqual((candidate.canvas.node_items[moved].pos().x(),candidate.canvas.node_items[moved].pos().y()),(123,456))
                self.assertTrue(candidate.reader_button.isHidden());self.assertTrue(candidate.diagnostic_button.isHidden())
                if dialog is not None:dialog.close();dialog.deleteLater()
                dialog=candidate
            dialog.set_source_path(self.path);dialog.start_run();self.assertFalse(dialog.reader_button.isEnabled())
            deadline=time.monotonic()+60
            while dialog._worker is not None and time.monotonic()<deadline:app.processEvents();time.sleep(.01)
            app.processEvents();self.assertIsNone(dialog._worker);self.assertIsNotNone(dialog.result,dialog.status_label.text())
            dialog.project_button.click();self.assertEqual(reader.selected_mode(kept[0]),'codec-gray')
            reopened=NativeRecipeDialog(kept[0]);self.assertTrue(reopened.reader_button.isHidden());reopened.close()
        finally:
            if dialog is not None:dialog.close()
            app.processEvents()

    def test_project_batch_keeps_old_and_reader_versions_and_selected_mode(self):
        from seedvision.export.project_native_graphs import export_project_native_graphs
        from seedvision.persistence.project_analysis import ProjectAnalysisStore,ProjectImageSpec
        from seedvision.persistence.analysis_settings import analysis_settings_profile_from_graph
        from seedvision.pipeline import build_default_pipeline
        workspace=self.root/'workspace';workspace.mkdir();(workspace/'config').mkdir()
        shutil.copyfile(Path(__file__).parents[1]/'config/traits.json',workspace/'config/traits.json')
        source=workspace/'seeds.png';shutil.copyfile(self.path,source)
        old=_base();selected=_add(_add(reader.build_native_reader_recipe(old),'median'),'fill')
        selected=_change(selected,reader.OPERATION,gray_source_mode='codec-gray')
        store=ProjectAnalysisStore(workspace)
        master=store.capture_and_save(analysis_settings=analysis_settings_profile_from_graph(build_default_pipeline()),
            images=(ProjectImageSpec(source,self.bgr.shape[:2]),),native_recipes=(old,selected),
            destination=workspace/'reader.seedfiddle-project.json')
        loaded=store.load(master);self.assertFalse(loaded.issues);self.assertEqual(loaded.document.native_recipes,(old,selected))
        destination=self.root/'batch';document=loaded.document
        batch=export_project_native_graphs((),source,destination,
            expected_source_sha256=document.images[0].source.sha256,project_root=workspace,
            native_recipes=document.native_recipes)
        rows=json.loads(batch.read_bytes())['graphs'];self.assertEqual(len(rows),2)
        for row in rows:self.assertEqual(row['status'],'executed',row['reason'])
        reports=[batch.parent/row['report'] for row in rows]
        recipes=[RecipeSpec.from_json(json.dumps(json.loads(p.read_bytes())['recipe'])) for p in reports]
        self.assertEqual({r.recipe_version for r in recipes},{old.recipe_version,selected.recipe_version})
        self.assertEqual(reader.selected_mode(next(r for r in recipes if reader.is_reader_recipe(r))),'codec-gray')
        for path,recipe in zip(reports,recipes,strict=True):
            if reader.is_reader_recipe(recipe):
                report,arrays=archives.load_native_recipe_result(path.parent)
                runner._validate_preparation_mode(recipe,report['preparation'])
                node=next(n for n in report['nodes'] if n['operation_id']==reader.OPERATION)
                binding=dict(node['outputs'])['gray_image']
                np.testing.assert_array_equal(arrays[binding['name']],cv2.imdecode(
                    np.frombuffer(source.read_bytes(),np.uint8),inputs.GRAY_DECODE_FLAGS))
