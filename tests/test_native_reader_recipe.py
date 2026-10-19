"""Pinned codec grayscale and versioned desktop/project reader selection."""

from tests.functional_units import functional_units
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest
from tests.catalogue_fixtures import CatalogueTestCase
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
from tests.test_native_recipe import fixture, selected_recipe
from tests.test_morphology_roi_recipe import _change, _composition_orders

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


@functional_units
class ReaderContractTests(CatalogueTestCase):
    def check_test_all_24_ids_have_additive_reader_versions_and_calibrated_controls_reject(self):
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

    def check_test_all_24_action_orders_preserve_selected_parameters_layout_and_project_versions(self):
        old=_base();old=replace(old,layout=tuple(replace(p,x=p.x+123,y=p.y+456) for p in old.layout))
        expected=None
        for order,recipe in _composition_orders(old,('reader','median','fill','morphology'),_add):
            with self.subTest(order=order):
                runner._validate_recipe(recipe)
                self.assertEqual(reader.selected_mode(recipe),'codec-gray')
                self.assertEqual(recipe.layout,old.layout)
                for operation,values in (
                        (median.OPERATION,{'ksize':[2,3],median.PARAMETER:[20,20,70,70]}),
                        (fill.OPERATION,{'size':80,median.PARAMETER:[21,21,71,71]}),
                        ('morphology.erode',{'ksize':2,median.PARAMETER:[22,22,72,72]}),
                        ('morphology.dilate',{'iterations':2,median.PARAMETER:[23,23,73,73]})):
                    stage=next(s.stage_id for s in recipe.stages if s.operation_id==operation)
                    actual=recipe.parameters_for_runner(stage)
                    for parameter,value in values.items():self.assertEqual(actual[parameter],value)
                if expected is None:expected=recipe.identity
                self.assertEqual(recipe.identity,expected)
        versions=(old,median.build_native_median_roi_recipe(old),reader.build_native_reader_recipe(old),recipe)
        self.assertEqual(native_recipes_from_payload(native_recipes_to_payload(versions)),versions)
        for selected in versions:self.assertEqual(RecipeSpec.from_json(selected.to_json()).identity,selected.identity)

    def check_test_invalid_modes_mismatched_decodes_and_owned_array_caps(self):
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

    def test_validation_and_recovery(self):
        """2 related ReaderContractTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_all_24_ids_have_additive_reader_versions_and_calibrated_controls_reject)
        self.scenario(self.check_test_invalid_modes_mismatched_decodes_and_owned_array_caps)

    def test_workflow_lifecycle(self):
        """1 related ReaderContractTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_all_24_action_orders_preserve_selected_parameters_layout_and_project_versions)


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
@functional_units
class ReaderIntegrationTests(CatalogueTestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.root=Path(self.temporary.name)
        self.path=self.root/'source.png';gray,_=fixture()
        self.bgr=np.repeat(gray[...,None],3,axis=2)
        self.assertTrue(cv2.imwrite(str(self.path),self.bgr))

    def tearDown(self):self.temporary.cleanup()

    def check_test_mode_switch_reuses_rgb_same_mode_reuses_nodes_and_roi_edits_reuse_source(self):
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





    def check_test_both_reader_versions_archive_and_replay_without_origin(self):
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
            from tests.observation import observe_functions
            try:
                with observe_functions(profile, inputs.source_identity, runner._dispatch):
                    actual=replay.replay_native_stage(saved,stage)
            finally:self.path.write_bytes(raw)
            self.assertEqual(actual.status,'matched',actual.reason)
            self.assertEqual(calls,[median.OPERATION])
            torch.testing.assert_close(actual.outputs['image'],result.node(stage).outputs['image'])
            self.assertEqual(before,{str(p.relative_to(saved)):p.read_bytes() for p in saved.rglob('*') if p.is_file()})


    def check_test_project_batch_keeps_old_and_reader_versions_and_selected_mode(self):
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

    def test_workflow_lifecycle(self):
        """3 related ReaderIntegrationTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_mode_switch_reuses_rgb_same_mode_reuses_nodes_and_roi_edits_reuse_source)
        self.scenario(self.check_test_both_reader_versions_archive_and_replay_without_origin)
        self.scenario(self.check_test_project_batch_keeps_old_and_reader_versions_and_selected_mode)
