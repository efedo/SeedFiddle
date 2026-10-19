"""Complete corrected range-mask workflow with physical observations and replay."""

from tests.catalogue_fixtures import CatalogueTestCase

from tests.functional_units import functional_units
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


class RangeCompositionContractTests(CatalogueTestCase):
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



@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
@functional_units
class CalibratedRangeTests(CatalogueTestCase):
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

    def check_test_five_modes_use_exact_corrected_roi_pixels_and_local_range_replay(self):
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

    def check_test_range_median_roi_scale_cache_boundaries_and_empty_run_drop_old_traits(self):
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

    def test_additional_numerical_regressions(self):
        self.scenario(self.check_test_five_modes_use_exact_corrected_roi_pixels_and_local_range_replay)
        self.scenario(self.check_test_range_median_roi_scale_cache_boundaries_and_empty_run_drop_old_traits)









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



if __name__=='__main__':unittest.main()
