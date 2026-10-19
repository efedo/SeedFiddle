"""Real composed card/seed execution, coordinate binding and desktop persistence."""

from tests.catalogue_fixtures import CatalogueTestCase

from tests.functional_units import functional_units
from dataclasses import replace
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
from seedvision.benchmarking.checkpoints import CoordinateFrame
from seedvision.benchmarking.recipes import LayoutPosition
from seedvision.segmentation.baseline import PipelineAnalysisCache
from tests import test_native_recipe as seed_tests
from tests import test_plantcv_card_detection as card_tests
from tests.test_native_card_recipe import _parameter


def fixture():
    image=np.full((620,900,3),220,np.uint8)
    image[:,:460]=card_tests.fixture(varied=True)
    gray,_=seed_tests.fixture()
    # This deliberately varied synthetic card maps the dark seed bytes above
    # the corrected background. Keep the desired light-object polarity explicit.
    seed=np.where(gray==35,210,np.where(gray==210,35,255)).astype(np.uint8)
    image[260:356,600:728]=seed[...,None]
    return image


def selected_recipe(kind='binary'):
    recipe=factory.build_native_calibrated_seed_recipe(kind)
    selected=seed_tests.selected_recipe(kind, **{'threshold.binary':{'threshold':124}})
    by_operation={s.operation_id:s for s in selected.stages if s.operation_id!='external.image_inputs'}
    stages=[]
    for stage in recipe.stages:
        if stage.operation_id in by_operation:
            values={p.parameter_id:p.value for p in by_operation[stage.operation_id].parameters}
            stage=replace(stage,parameters=tuple(replace(p,value=values.get(p.parameter_id,p.value)) for p in stage.parameters))
        stages.append(stage)
    recipe=replace(recipe,stages=tuple(stages))
    for stage,name,value in ((card.PREPROCESS,'adaptive_method',0),
            (card.PREPROCESS,'roi_bounds_xyxy',[0,0,460,620]),
            (factory.CROP,'seed_roi_xyxy',[600,260,728,356]),
            (card.SCALE,'color_chip_size',[7,11])):
        recipe=_parameter(recipe,stage,name,value)
    return recipe


def reference_script():
    script=seed_tests.REFERENCE_SCRIPT.replace('threshold=100,','threshold=124,')
    original="rgb = values['rgb']\nbgr = np.ascontiguousarray(rgb[..., ::-1])\noutputs = {}"
    combined="""image = values['bgr']
auto_module = importlib.import_module('plantcv.plantcv.transform.auto_correct_color')
assert hashlib.sha256(Path(auto_module.__file__).read_bytes()).hexdigest() == 'bed2eb9e50e42e39f7ad4bbef157150190585268178f06083092edb19ffaf57a'
helper_module = importlib.import_module('plantcv.plantcv._helpers')
assert hashlib.sha256(Path(helper_module.__file__).read_bytes()).hexdigest() == 'c05b0c944210657332ce307f2066caab55f2ebc8d8fea11988db3e476e7cdfba'
pcv.params.px_width = pcv.params.px_height = 1.0
pcv.params.unit = 'pixels'
roi = pcv.Objects(contours=[[np.array([[[0,0]],[[0,620]],[[460,620]],[[460,0]]],dtype=np.int32)]],
    hierarchy=[np.array([[[-1,-1,-1,-1]]],dtype=np.int32)])
corrected = pcv.transform.auto_correct_color(image, roi=roi, adaptive_method=0, color_chip_size=[7,11])
scale = [pcv.params.px_width,pcv.params.px_height,pcv.params.unit]
bgr = corrected[260:356,600:728].copy()
rgb = bgr[...,::-1].copy()
values = {'gray':cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)}
outputs = {'corrected':corrected,'seed_rgb':rgb,'seed_gray':values['gray']}"""
    assert script.count(original)==1
    script=script.replace(original,combined).replace("pcv.params.unit = 'pixels'\noutputs['labels']", "pcv.params.unit = scale[2]\noutputs['labels']")
    return script.replace("observations['__size_observations__'] = dict(pcv.outputs.observations)",
        "observations['__size_observations__'] = dict(pcv.outputs.observations)\nobservations['__scale__'] = scale")


class CalibratedContractTests(unittest.TestCase):
    pass


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
@functional_units
class CalibratedSeedTests(CatalogueTestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.image=fixture();self.path=self.root/'card-seeds.png'
        self.assertTrue(cv2.imwrite(str(self.path),self.image))
        self.recipe=selected_recipe();self.cache=PipelineAnalysisCache()

    def tearDown(self):self.temp.cleanup()

    def run_recipe(self,recipe=None,**kwargs):
        return runner.run_native_recipe_path(recipe or self.recipe,self.path,cache=self.cache,**kwargs)

    def reference(self,kind):
        source=self.root/'inputs.npz';output=self.root/'output.npz';observations=self.root/'observations.json'
        np.savez(source,bgr=self.image)
        completed=subprocess.run([str(seed_tests.REFERENCE),'-c',reference_script(),str(source),str(output),
            str(observations),kind,'rgb'],capture_output=True,text=True,timeout=120)
        self.assertEqual(completed.returncode,0,completed.stderr)
        with np.load(output,allow_pickle=False) as loaded:arrays={name:loaded[name].copy() for name in loaded.files}
        return arrays,json.loads(observations.read_text(encoding='utf-8'))

    def assert_observations(self,actual,expected):
        self.assertEqual(set(actual),set(expected))
        for sample,values in expected.items():
            self.assertEqual(set(actual[sample]),set(values))
            for variable,item in values.items():
                row=actual[sample][variable]
                for field in ('trait','method','scale','datatype'):self.assertEqual(row[field],item[field],(variable,field))
                self.assertEqual(row['label'],tuple(item['label']) if isinstance(item['label'],list) else item['label'])
                np.testing.assert_allclose(row['value'],item['value'],atol=1e-6,rtol=1e-6,err_msg=variable)

    def check_test_actual_public_auto_crop_seed_chains_and_physical_observations(self):
        for kind in ('binary','mean','otsu'):
            with self.subTest(kind=kind):
                expected,observations=self.reference(kind)
                result=self.run_recipe(selected_recipe(kind))
                prefix='pcv_seed_'+kind+'_'
                corrected=result.node(card.AFFINE).outputs['corrected_rgb'].cpu().numpy()
                self.assertLessEqual(np.abs(corrected.astype(int)-expected['corrected'][...,::-1].astype(int)).max(),1)
                crop=result.node(factory.CROP).outputs
                np.testing.assert_array_equal(crop['rgb_image'].cpu(),expected['seed_rgb'])
                np.testing.assert_array_equal(crop['gray_image'].cpu(),expected['seed_gray'])
                for suffix,port,reference in (('median_blur','image','median_blur'),
                        (kind+'_threshold','mask','binary_threshold'),('small_object_filter','mask','small_object_filter'),
                        ('erode','mask','erode'),('dilate','mask','dilate'),('ws_distance_map','distance','distance'),
                        ('ws_local_maxima','coordinates','coordinates'),('ws_local_maxima','peaks','peaks'),
                        ('ws_markers','markers','markers'),('ws_invert_distance','negative_distance','negative_distance'),
                        ('ws_masked_flood','labels','labels')):
                    node=result.node(prefix+suffix);self.assertEqual(node.status,'executed',node.reason)
                    np.testing.assert_array_equal(node.outputs[port].cpu(),expected[reference])
                color=result.node(prefix+'analyze_color')
                self.assertEqual(color.status,'executed',color.reason)
                for label in color.outputs['observations'].labels:
                    for histogram in label.histograms:
                        np.testing.assert_array_equal(histogram.values,observations[label.sample][histogram.variable]['value'])
                size=result.node(prefix+'analyze_size');scaled=result.node(factory.SCALED_SIZE)
                if size.status=='unavailable':
                    self.assertEqual(kind,'mean');self.assertIn('orientation numerically unstable',size.reason)
                    self.assertEqual(scaled.status,'not_run')
                else:
                    self.assertEqual(result.status,'executed',[(n.stage_id,n.reason) for n in result.nodes])
                    self.assert_observations(scaled.outputs['observations'],observations['__size_observations__'])
                    np.testing.assert_array_equal(size.outputs['diagnostic'].cpu(),expected['size_diagnostic_rgb'])
                    scale=result.node(card.SCALE).outputs['metadata']
                    self.assertEqual([scale['px_width'],scale['px_height'],scale['unit']],observations['__scale__'])
                self.assertTrue(all(t.device.type=='cuda' for n in result.nodes for t in runner._iter_nested_tensors(n.outputs)))
                self.assertEqual(result.labels.shape,(96,128))
                metadata=crop['metadata'];local=CoordinateFrame.from_dict(dict(metadata['coordinate_frame']))
                self.assertEqual(local.shape,(96,128));self.assertEqual(metadata['local_to_parent_xy'],(600,260))
                self.assertEqual(result.labels_coordinate_frame,local)
                self.assertEqual(CoordinateFrame.from_dict(dict(metadata['parent_frame'])),result.coordinate_frame)
                self.assertEqual(metadata['card_bounding_overlap_pixels'],0)
                self.assertRegex(local.transform_sha256,r'^[0-9a-f]{64}$')

    def check_test_roi_scale_and_seed_edits_reuse_independent_branches(self):
        first=self.run_recipe();self.assertEqual(first.status,'executed')
        self.assertTrue(all(n.reused for n in self.run_recipe().nodes))
        changed=_parameter(self.recipe,card.SCALE,'color_chip_size',[7,13])
        physical=self.run_recipe(changed)
        self.assertEqual(set(self.cache.last_computed_nodes),{card.SCALE,factory.SCALED_SIZE})
        a=first.node(factory.SCALED_SIZE).outputs['observations']['graph_1']
        b=physical.node(factory.SCALED_SIZE).outputs['observations']['graph_1']
        self.assertNotEqual(a['area']['value'],b['area']['value'])
        self.assertEqual(a['width']['value'],b['width']['value'])
        self.assertEqual(a['center_of_mass'],b['center_of_mass'])
        self.assertIs(first.labels,physical.labels)
        roi=_parameter(changed,factory.CROP,'seed_roi_xyxy',[601,259,730,356])
        cropped=self.run_recipe(roi)
        self.assertEqual(cropped.node(factory.CROP).outputs['gray_image'].shape,(97,129))
        self.assertEqual(set(self.cache.last_reused_nodes),{s.stage_id for s in roi.stages[:9]})
        self.assertIs(first.node(card.AFFINE).outputs['corrected_rgb'],cropped.node(card.AFFINE).outputs['corrected_rgb'])
        threshold=_parameter(roi,'pcv_seed_binary_binary_threshold','threshold',125)
        self.run_recipe(threshold)
        self.assertIn(factory.CROP,self.cache.last_reused_nodes)
        self.assertIn('pcv_seed_binary_median_blur',self.cache.last_reused_nodes)
        layout=replace(threshold,title='Saved calibrated seed graph',layout=tuple(LayoutPosition(s.stage_id,200*i,80) for i,s in enumerate(threshold.stages)))
        reused=self.run_recipe(layout)
        self.assertFalse(self.cache.last_computed_nodes)
        self.assertTrue(all(n.reused for n in reused.nodes if n.product_key is not None))







    def check_test_saved_project_batch_and_exact_mixed_frame_archive(self):
        from seedvision.export import native_results as outputs
        from seedvision.export.project_native_graphs import export_project_native_graphs
        from seedvision.persistence.project_analysis import ProjectAnalysisStore,ProjectImageSpec
        from seedvision.persistence.analysis_settings import analysis_settings_profile_from_graph
        from seedvision.pipeline import build_default_pipeline
        from tests.test_native_result_export import _attachment,_output
        workspace=self.root/'workspace';workspace.mkdir();(workspace/'config').mkdir()
        source=workspace/'seeds.png';shutil.copyfile(self.path,source)
        shutil.copyfile(Path(__file__).parents[1]/'config/traits.json',workspace/'config/traits.json')
        history=(_attachment(seed_tests.selected_recipe()),)
        store=ProjectAnalysisStore(workspace)
        master=store.capture_and_save(analysis_settings=analysis_settings_profile_from_graph(build_default_pipeline()),
            images=(ProjectImageSpec(source,self.image.shape[:2]),),reference_graphs=history,
            native_recipes=(self.recipe,),destination=workspace/'seeds.seedfiddle-project.json')
        document=store.load(master).document
        self.assertEqual(document.native_recipes,(self.recipe,));self.assertEqual(document.reference_graphs,history)
        ledger=export_project_native_graphs((),source,self.root/'runs',expected_source_sha256=document.images[0].source.sha256,
            project_root=workspace,native_recipes=document.native_recipes)
        row=json.loads(ledger.read_bytes())['graphs'][0];self.assertEqual(row['status'],'executed',row['reason'])
        report,arrays=outputs.load_native_recipe_result(ledger.parent/Path(row['report']).parent)
        expected=self.run_recipe()
        for stage,port in ((card.AFFINE,'corrected_rgb'),(factory.CROP,'rgb_image'),(factory.CROP,'gray_image'),
                ('pcv_seed_binary_ws_masked_flood','labels')):
            array=arrays[_output(report,stage,port)['name']]
            np.testing.assert_array_equal(array,expected.node(stage).outputs[port].cpu())
        self.assertFalse(report['scientifically_validated']);self.assertEqual(row['historical_stage_runs'],[])
        self.assertIn('not per-stage replay',report['checkpoint_scope'])

    def test_workflow_lifecycle(self):
        """3 related CalibratedSeedTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_actual_public_auto_crop_seed_chains_and_physical_observations)
        self.scenario(self.check_test_roi_scale_and_seed_edits_reuse_independent_branches)
        self.scenario(self.check_test_saved_project_batch_and_exact_mixed_frame_archive)


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class CalibratedSeedDialogTests(unittest.TestCase):
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
