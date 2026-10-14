"""Portable model/class-map replay, lineage, guards and actual classifier Qt."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
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

os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PySide6.QtWidgets import QApplication
from seedvision.benchmarking import native_recipe as runner, native_stage_replay as replay
from seedvision.benchmarking import native_model_replay as contracts
from seedvision.export import native_results as archives
from seedvision.model_io import plantcv_kmeans_model as models, kmeans_model_artifact as artifact
from seedvision.ui.native_naive_bayes_dialog import NativeNaiveBayesDialog
from seedvision.ui.native_kmeans_dialog import NativeKMeansDialog
from tests import test_native_naive_bayes_recipe as nb, test_native_kmeans_recipe as km
from tests.test_kmeans_model_artifact import _conversion

BAYES='analysis.naive_bayes_classifier'
PREDICT='analysis.kmeans_predict'
MASKS='analysis.kmeans_masks'


def _hashes(directory):
    return {str(path.relative_to(directory)):sha256(path.read_bytes()).hexdigest()
            for path in directory.rglob('*') if path.is_file()}


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class ModelStageReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary=tempfile.TemporaryDirectory(dir='artifacts')
        cls.root=Path(cls.temporary.name)
        cls.image=cls.root/'source.png';cv2.imwrite(str(cls.image),km._image())
        cls.cases={}
        cls.model_path=cls.root/'bayes.tsv';cls.model_path.write_bytes(nb._pdf())
        recipe,_=nb._recipe_with_model(cls.model_path)
        saved=runner.run_native_recipe_path(recipe,cls.image)
        assert saved.status=='executed',[(n.stage_id,n.status,n.reason) for n in saved.nodes]
        directory=cls.root/'bayes';archives.export_native_recipe_result(directory,recipe,saved)
        cls.cases['bayes']=(recipe,saved,directory,cls.model_path)
        for channels,patch_size,envelope in ((3,3,False),(1,2,False),(1,2,True)):
            name=f'kmeans-{channels}-{patch_size}-{envelope}'
            f=channels*patch_size*patch_size
            model=models.model_from_centres(tuple((float(v),)*f for v in (25,100,200)),patch_size,channels)
            raw=artifact.encode_kmeans_artifact(model,_conversion(model)) if envelope else models.encode_kmeans_model(model)
            path=cls.root/(name+'.json');path.write_bytes(raw)
            recipe=km._recipe(path);saved=runner.run_native_recipe_path(recipe,cls.image)
            assert saved.status=='executed',[(n.stage_id,n.status,n.reason) for n in saved.nodes]
            directory=cls.root/name;archives.export_native_recipe_result(directory,recipe,saved)
            cls.cases[name]=(recipe,saved,directory,path)

    @classmethod
    def tearDownClass(cls):cls.temporary.cleanup()

    def attempt(self,case='bayes',stage=BAYES,directory=None,cancel=None,inventory_change=None):
        directory=directory or self.cases[case][2]
        calls=[];selected=[];parsed=[]
        def profile(frame,event,arg):
            if event=='call':
                if frame.f_code is runner._dispatch.__code__:calls.append(frame.f_locals['operation'])
                elif frame.f_code is archives.load_native_recipe_result.__code__:selected.append(frame.f_locals['array_names'])
                elif frame.f_code is contracts.parse_asset.__code__:parsed.append(frame.f_locals['binding'].kind)
            elif event=='return' and inventory_change is not None and frame.f_code is archives.read_native_recipe_inventory.__code__ and arg is not None:
                inventory_change(*arg)
        previous=sys.getprofile();sys.setprofile(profile)
        try:result=replay.replay_native_stage(directory,stage,cancel=cancel)
        finally:sys.setprofile(previous)
        return result,calls,selected,parsed

    def changed(self,case='bayes',**replacements):
        recipe,saved,_,_=self.cases[case]
        saved=replace(saved,nodes=tuple(replacements.get(node.stage_id,node) for node in saved.nodes))
        directory=self.root/('changed-'+str(len(tuple(self.root.iterdir()))))
        archives.export_native_recipe_result(directory,recipe,saved)
        return directory

    def test_bayes_every_mask_names_and_model_metadata_match_with_exact_selection(self):
        recipe,saved,directory,_=self.cases['bayes'];before=_hashes(directory)
        result,calls,selected,parsed=self.attempt()
        self.assertEqual(result.status,'matched',result.reason);self.assertEqual(calls,[BAYES]);self.assertEqual(parsed,['bayes'])
        expected=saved.node(BAYES).outputs
        self.assertEqual(tuple(result.outputs['class_masks']),expected['class_names'])
        for name,mask in expected['class_masks'].items():self.assertTrue(torch.equal(result.outputs['class_masks'][name],mask))
        self.assertEqual(len(selected[0]),2+len(expected['class_names']))
        self.assertEqual(result.metadata['comparison']['by_port']['class_masks']['compared_elements'],len(expected['class_names'])*(23*29+1))
        self.assertEqual(result.metadata['comparison']['by_port']['class_names']['compared_elements'],len(expected['class_names']))
        self.assertEqual(result.metadata['plan']['raw_model_cuda_bytes'],0)
        self.assertFalse(result.scientifically_validated);self.assertEqual(before,_hashes(directory))

    def test_gray_rgb_odd_even_and_both_kmeans_formats_select_only_active_image(self):
        for case,(recipe,saved,directory,_) in self.cases.items():
            if case=='bayes':continue
            before=_hashes(directory)
            for stage in (PREDICT,MASKS):
                with self.subTest(case=case,stage=stage):
                    result,calls,selected,parsed=self.attempt(case,stage)
                    self.assertEqual(result.status,'matched',result.reason);self.assertEqual(calls,[stage])
                    if stage==PREDICT:
                        channels=recipe.parameters_for_runner(runner._kmeans_recipe.MODEL_STAGE_ID)['model_source']['channels']
                        self.assertEqual(result.metadata['active_image_port'],'gray_image' if channels==1 else 'rgb_image')
                        self.assertEqual(len(selected[0]),3);self.assertEqual(parsed,['kmeans'])
                        self.assertTrue(torch.equal(result.outputs['labels'],saved.node(PREDICT).outputs['labels']))
                    else:
                        self.assertEqual(len(selected[0]),4);self.assertFalse(parsed)
                        self.assertFalse(result.metadata['model_bytes_selected'])
                        for name,mask in saved.node(MASKS).outputs['class_masks'].items():
                            self.assertTrue(torch.equal(result.outputs['class_masks'][name],mask))
            self.assertEqual(before,_hashes(directory))

    def test_replays_after_original_model_file_is_deleted(self):
        for case,(_,_,directory,path) in self.cases.items():
            raw=path.read_bytes();path.unlink()
            try:
                for stage in (BAYES,) if case=='bayes' else (PREDICT,MASKS):
                    result,_,_,_=self.attempt(case,stage)
                    self.assertEqual(result.status,'matched',result.reason)
            finally:path.write_bytes(raw)

    def test_explicit_duplicate_absent_and_large_integer_union_selection(self):
        recipe,_,_,_=self.cases['kmeans-3-3-False']
        recipe=km._parameter(recipe,MASKS,'cat_list',[0,0,2,-1,500,10**70])
        saved=runner.run_native_recipe_path(recipe,self.image)
        self.assertEqual(saved.status,'executed')
        directory=self.root/'union';archives.export_native_recipe_result(directory,recipe,saved)
        result,calls,selected,parsed=self.attempt('kmeans-3-3-False',MASKS,directory)
        self.assertEqual(result.status,'matched',result.reason);self.assertEqual(calls,[MASKS]);self.assertEqual(len(selected[0]),2)
        self.assertFalse(parsed);self.assertEqual(result.outputs['class_names'],('selected',))
        self.assertEqual(result.metadata['plan']['work_details']['category_pixel_visits'],23*29*6)

    def test_valid_renamed_reordered_and_byte_changed_masks_are_product_mismatches(self):
        saved=self.cases['bayes'][1];node=saved.node(BAYES);outputs=dict(node.outputs)
        names=outputs['class_names'];masks=outputs['class_masks']
        renamed=('changed-class',*names[1:])
        changed={**outputs,'class_names':renamed,'class_masks':dict(zip(renamed,masks.values()))}
        result,_,_,_=self.attempt(directory=self.changed(**{BAYES:replace(node,outputs=changed)}))
        self.assertEqual(result.status,'mismatch',result.reason)
        reversed_names=tuple(reversed(names));changed={**outputs,'class_names':reversed_names,
            'class_masks':{name:masks[name] for name in reversed_names}}
        result,_,_,_=self.attempt(directory=self.changed(**{BAYES:replace(node,outputs=changed)}))
        self.assertEqual(result.status,'mismatch',result.reason)
        pixels={name:mask.clone() for name,mask in masks.items()};pixels[names[0]][0,0]=17
        result,_,_,_=self.attempt(directory=self.changed(**{BAYES:replace(node,outputs={**outputs,'class_masks':pixels})}))
        self.assertEqual(result.status,'mismatch',result.reason)
        self.assertEqual(result.metadata['comparison']['mismatched_elements'],1)

    def test_all_absent_categories_produce_zero_union_and_changed_valid_label_mismatches(self):
        case='kmeans-3-3-False';recipe,saved,_,_=self.cases[case]
        recipe=km._parameter(recipe,MASKS,'cat_list',[-1,500,10**70])
        selected=runner.run_native_recipe_path(recipe,self.image)
        self.assertEqual(selected.status,'executed')
        directory=self.root/'all-absent';archives.export_native_recipe_result(directory,recipe,selected)
        result,_,_,parsed=self.attempt(case,MASKS,directory)
        self.assertEqual(result.status,'matched',result.reason);self.assertFalse(parsed)
        self.assertEqual(result.outputs['class_names'],('selected',))
        self.assertEqual(torch.count_nonzero(result.outputs['class_masks']['selected']).item(),0)
        node=saved.node(PREDICT);labels=node.outputs['labels'].clone()
        labels[0,0]=(int(labels[0,0].item())+1)%3
        result,_,_,_=self.attempt(case,PREDICT,self.changed(case,**{
            PREDICT:replace(node,outputs={**node.outputs,'labels':labels})}))
        self.assertEqual(result.status,'mismatch',result.reason)
        self.assertEqual(result.metadata['comparison']['mismatched_elements'],1)

    def test_inconsistent_names_and_missing_maps_are_invalid_before_selected_load(self):
        saved=self.cases['bayes'][1];node=saved.node(BAYES);outputs=dict(node.outputs)
        changes=({**outputs,'class_names':tuple(reversed(outputs['class_names']))},
                 {**outputs,'class_names':(outputs['class_names'][0],)*len(outputs['class_names'])},
                 {**outputs,'class_masks':dict(list(outputs['class_masks'].items())[:-1])})
        for changed in changes:
            result,calls,selected,_=self.attempt(directory=self.changed(**{BAYES:replace(node,outputs=changed)}))
            self.assertEqual(result.status,'invalid_archive',result.reason);self.assertFalse(calls);self.assertFalse(selected)

    def test_producer_key_identity_and_inactive_header_tampering_rejects_before_load(self):
        def mutate_input_key(report,specs):
            next(n for n in report['nodes'] if n['stage_id']==PREDICT)['input_keys']['model_identity']='0'*64
        def mutate_identity(report,specs):
            node=next(n for n in report['nodes'] if n['stage_id']==runner._kmeans_recipe.MODEL_STAGE_ID)
            binding=dict(node['outputs'])['model_identity']
            next(item for item in binding['items'] if item[0]=='source_bytes')[1]+=1
        def mutate_inactive_header(report,specs):
            node=next(n for n in report['nodes'] if n['stage_id']==runner._kmeans_recipe.IMAGE_STAGE_ID)
            name=dict(node['outputs'])['rgb_image']['name'];specs[name]['shape']=[23,28,3]
        for mutate in (mutate_input_key,mutate_identity,mutate_inactive_header):
            result,calls,selected,_=self.attempt('kmeans-1-2-False',PREDICT,inventory_change=mutate)
            self.assertEqual(result.status,'invalid_archive',result.reason);self.assertFalse(calls);self.assertFalse(selected)

    def test_selected_and_cuda_limits_precede_runtime_and_gpu_allocation(self):
        for case,stage in (('bayes',BAYES),('kmeans-1-2-False',PREDICT),('kmeans-3-3-False',MASKS)):
            with patch.object(replay,'MAX_TYPED_SELECTED_BYTES',1),patch.object(replay,'_runtime') as runtime:
                result,calls,selected,_=self.attempt(case,stage)
            self.assertEqual(result.status,'unavailable',result.reason);self.assertFalse(calls);self.assertFalse(selected);runtime.assert_not_called()
            with patch.object(replay,'MAX_REPLAY_CUDA_BYTES',64*1024-1),patch.object(replay,'_runtime') as runtime, \
                    patch.object(torch,'tensor',side_effect=AssertionError('late GPU allocation')):
                result,calls,selected,_=self.attempt(case,stage)
            self.assertEqual(result.status,'unavailable',result.reason);self.assertFalse(calls);runtime.assert_not_called()

    def test_small_many_class_masks_fit_rounded_cuda_allocation_plan(self):
        import gc
        model=models.model_from_centres(tuple((float(i*4),) for i in range(56)),1,1)
        path=self.root/'many-small-model.json';path.write_bytes(models.encode_kmeans_model(model))
        recipe=km._recipe(path);source=self.root/'many-small-source.png'
        assert cv2.imwrite(str(source),np.full((5,7,3),100,dtype=np.uint8))
        saved=runner.run_native_recipe_path(recipe,source)
        self.assertEqual(saved.status,'executed')
        directory=self.root/'many-small';archives.export_native_recipe_result(directory,recipe,saved)
        gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache()
        baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
        result,_,_,_=self.attempt('kmeans-1-2-False',MASKS,directory)
        torch.cuda.synchronize();delta=torch.cuda.max_memory_allocated()-baseline
        self.assertEqual(result.status,'matched',result.reason)
        self.assertEqual(len(result.outputs['class_masks']),56)
        self.assertLessEqual(delta,result.metadata['plan']['cuda_bytes'])

    def mutate_callback(self,mode):
        changed=[]
        def callback():
            frame=sys._getframe(1)
            while frame is not None:
                local=frame.f_locals
                if not changed and frame.f_code is runner._model_dispatch.__code__:
                    if mode=='input':
                        tensor=next(v for v in local['inputs'].values() if torch.is_tensor(v));tensor.zero_();changed.append(mode)
                    elif mode=='adapter' and 'bgr' in local:
                        local['bgr'].zero_();changed.append(mode)
                    elif mode=='model' and local.get('model') is not None:
                        model=local['model'];field='probabilities' if hasattr(model,'probabilities') else 'centres'
                        object.__setattr__(model,field,tuple(reversed(getattr(model,field))));changed.append(mode)
                elif not changed and mode=='output' and frame.f_code is replay.replay_native_stage.__code__:
                    outputs=local.get('outputs',{})
                    if 'class_masks' in outputs:
                        next(iter(outputs['class_masks'].values())).zero_();changed.append(mode)
                frame=frame.f_back
            return False
        return callback,changed

    def test_callbacks_cannot_mutate_source_private_bgr_model_tables_or_nested_outputs(self):
        for mode in ('input','adapter','model','output'):
            callback,changed=self.mutate_callback(mode)
            result,_,_,_=self.attempt(cancel=callback)
            self.assertTrue(changed,mode);self.assertEqual(result.status,'failed',result.reason);self.assertFalse(result.outputs)
        callback,changed=self.mutate_callback('model')
        result,_,_,_=self.attempt('kmeans-1-2-False',PREDICT,cancel=callback)
        self.assertTrue(changed);self.assertEqual(result.status,'failed',result.reason);self.assertFalse(result.outputs)

    def test_private_cuda_buffers_are_guarded_including_callback_exception_paths(self):
        cases=(('bayes',BAYES,runner._naive_bayes.naive_bayes_masks,'table'),
               ('bayes',BAYES,runner._naive_bayes.naive_bayes_masks,'winner'),
               ('kmeans-1-2-False',PREDICT,runner._kmeans_preprocess.gaussian_patch_raster,'vertical'),
               ('kmeans-3-3-False',PREDICT,runner._kmeans_features.iter_kmeans_patch_tiles,'features'),
               ('kmeans-1-2-False',PREDICT,runner._kmeans_classifier._classify_prepared_features,'centres_gpu'),
               ('kmeans-3-3-False',PREDICT,runner._kmeans_classifier._classify_prepared_features,'x'))
        for case,stage,function,field in cases:
            with self.subTest(case=case,field=field):
                changed=[]
                def callback():
                    frame=sys._getframe(1)
                    while frame:
                        tensor=frame.f_locals.get(field)
                        if not changed and frame.f_code is function.__code__ and torch.is_tensor(tensor):
                            tensor.zero_();changed.append(field)
                            raise RuntimeError('Caller exception after buffer mutation')
                        frame=frame.f_back
                    return False
                result,_,_,_=self.attempt(case,stage,cancel=callback)
                self.assertTrue(changed);self.assertEqual(result.status,'failed',result.reason);self.assertFalse(result.outputs)
                self.assertIn('callback changed',result.reason.lower())

    def test_parent_prediction_buffers_and_prepared_metadata_survive_nested_callbacks(self):
        for mode in ('blurred','labels','prepared'):
            changed=[]
            def callback():
                frame=sys._getframe(1);inside_scoring=False
                while frame:
                    if frame.f_code is runner._kmeans_classifier._classify_prepared_features.__code__:
                        inside_scoring=True
                    if not changed and inside_scoring and frame.f_code is runner._kmeans_predict.predict_kmeans.__code__:
                        if mode=='prepared':
                            prepared=frame.f_locals['prepared']
                            object.__setattr__(prepared,'centres_gpu',prepared.centres_gpu.clone())
                        else:frame.f_locals[mode].zero_()
                        changed.append(mode)
                    frame=frame.f_back
                return False
            result,_,_,_=self.attempt('kmeans-1-2-False',PREDICT,cancel=callback)
            self.assertTrue(changed,mode);self.assertEqual(result.status,'failed',result.reason);self.assertFalse(result.outputs)

    def test_final_map_order_and_each_partial_mask_are_guarded(self):
        for case,stage in (('bayes',BAYES),('kmeans-3-3-False',MASKS)):
            for mode in ('partial','order'):
                changed=[]
                def callback():
                    frame=sys._getframe(1)
                    while frame:
                        local=frame.f_locals
                        if not changed and mode=='partial' and stage==MASKS and frame.f_code is runner._kmeans_masks.mask_kmeans.__code__:
                            masks=local.get('masks',{})
                            if len(masks)>=2:next(reversed(masks.values())).zero_();changed.append(mode)
                        elif not changed and mode=='partial' and stage==BAYES and frame.f_code is runner._naive_bayes.naive_bayes_masks.__code__:
                            masks=local.get('masks')
                            if torch.is_tensor(masks):masks[-1].zero_();changed.append(mode)
                        elif not changed and mode=='order' and frame.f_code is runner._model_dispatch.__code__:
                            outputs=local.get('outputs',{})
                            if 'class_masks' in outputs:
                                outputs['class_masks']=dict(reversed(tuple(outputs['class_masks'].items())));changed.append(mode)
                        frame=frame.f_back
                    return False
                result,_,_,_=self.attempt(case,stage,cancel=callback)
                self.assertTrue(changed,(case,mode));self.assertEqual(result.status,'failed',result.reason);self.assertFalse(result.outputs)

    def test_strict_and_final_cancellation_publish_no_products(self):
        for case,stage in (('bayes',BAYES),('kmeans-1-2-False',PREDICT),('kmeans-3-3-False',MASKS)):
            for invalid in (1,np.bool_(False)):
                result,_,_,_=self.attempt(case,stage,cancel=lambda:invalid)
                self.assertEqual(result.status,'failed');self.assertFalse(result.outputs)
            cancelled=[]
            def stop():
                frame=sys._getframe(1)
                while frame:
                    if frame.f_code is runner._model_dispatch.__code__ and 'outputs' in frame.f_locals:
                        cancelled.append(True);return True
                    frame=frame.f_back
                return False
            result,_,_,_=self.attempt(case,stage,cancel=stop)
            self.assertTrue(cancelled);self.assertEqual(result.status,'cancelled',result.reason);self.assertFalse(result.outputs)

    def test_complete_archive_payload_verification_rejects_corruption_outside_selection(self):
        import shutil
        original=self.cases['kmeans-1-2-False'][2]
        directory=self.root/'corrupt';shutil.copytree(original,directory)
        npz=next(directory.rglob('*.npz'));raw=bytearray(npz.read_bytes());raw[len(raw)//2]^=1;npz.write_bytes(raw)
        result,calls,_,_=self.attempt('kmeans-1-2-False',PREDICT,directory)
        self.assertEqual(result.status,'invalid_archive',result.reason);self.assertFalse(calls)

    def test_actual_async_dialog_replay_class_selector_and_label_preview_preserve_active_run(self):
        app=QApplication.instance() or QApplication([])
        for case,dialog_class,stage in (('bayes',NativeNaiveBayesDialog,BAYES),
                ('kmeans-3-3-False',NativeKMeansDialog,MASKS),('kmeans-1-2-False',NativeKMeansDialog,PREDICT)):
            recipe,saved,directory,_=self.cases[case]
            dialog=dialog_class(recipe);dialog.resize(1050,650);dialog.show()
            dialog.source_path=self.image;dialog._completed(saved);dialog._select_stage(stage)
            dialog._replay_button_states=[]
            with patch('seedvision.ui.native_recipe_dialog.QFileDialog.getExistingDirectory',return_value=str(directory)):
                dialog._choose_stage_replay()
            deadline=time.monotonic()+30
            while dialog._worker is not None and time.monotonic()<deadline:app.processEvents();time.sleep(.01)
            self.assertIsNone(dialog._worker);self.assertIs(dialog.result,saved)
            self.assertEqual(dialog.replay_result.status,'matched',dialog.replay_result.reason)
            self.assertFalse(dialog.preview.pixmap().isNull())
            self.assertNotIn('tensor(',dialog.facts.toPlainText())
            if stage!=PREDICT:
                self.assertEqual(dialog.class_selector.count(),len(saved.node(stage).outputs['class_names']))
                dialog.class_selector.setCurrentIndex(dialog.class_selector.count()-1);app.processEvents()
                name=dialog.class_selector.currentText()
                np.testing.assert_array_equal(dialog._last_class_preview,dialog.replay_result.outputs['class_masks'][name].cpu().numpy())
                dialog._select_stage(stage);self.assertIsNone(dialog._replay_class_masks)
            dialog.close();app.processEvents()
