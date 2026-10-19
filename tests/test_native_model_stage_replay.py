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
from tests.catalogue_fixtures import CatalogueTestCase
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
class ModelStageReplayTests(CatalogueTestCase):
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
        from tests.observation import observe_functions
        with observe_functions(profile, runner._dispatch, archives.load_native_recipe_result,
                               contracts.parse_asset, archives.read_native_recipe_inventory):
            result=replay.replay_native_stage(directory,stage,cancel=cancel)
        return result,calls,selected,parsed

    def changed(self,case='bayes',**replacements):
        recipe,saved,_,_=self.cases[case]
        saved=replace(saved,nodes=tuple(replacements.get(node.stage_id,node) for node in saved.nodes))
        directory=self.root/('changed-'+str(len(tuple(self.root.iterdir()))))
        archives.export_native_recipe_result(directory,recipe,saved)
        return directory

    def test_bayes_every_mask_names_and_model_metadata_match_with_exact_selection(self):
        from tests.replay_fixtures import hidden_origins
        with hidden_origins(*(item[3] for item in self.cases.values())):
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
        from tests.replay_fixtures import hidden_origins
        with hidden_origins(*(item[3] for item in self.cases.values())):
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
