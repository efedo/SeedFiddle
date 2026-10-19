
from tests.catalogue_fixtures import CatalogueTestCase

from tests.functional_units import functional_units
from dataclasses import fields,is_dataclass,replace
from collections.abc import Mapping
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import tracemalloc
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from seedvision.benchmarking import native_core_recovery_recipe as factory
from seedvision.benchmarking import native_recipe as runner
from seedvision.benchmarking.checkpoints import CoordinateFrame
from seedvision.benchmarking.recipe_catalogue import build_recipe_catalogue
from seedvision.benchmarking.recipes import RecipeSpec
from seedvision.segmentation.baseline import PipelineAnalysisCache
from tests.test_plantcv_watershed import _REFERENCE, _PINS_SCRIPT


def parameter(recipe, stage_id, name, value):
    return replace(recipe, stages=tuple(replace(stage,parameters=tuple(
        replace(p,value=value) if p.parameter_id == name else p for p in stage.parameters))
        if stage.stage_id == stage_id else stage for stage in recipe.stages))


def selected_recipe():
    recipe=factory.build_native_core_recovery_recipe()
    for stage,name,value in ((factory.FOREGROUND,'threshold',100),
                             (factory.ERODE,'ksize',5),(factory.ERODE,'iterations',1)):
        recipe=parameter(recipe,stage,name,value)
    return recipe


def fixture(touching=True):
    yy,xx=np.indices((64,96))
    right=50 if touching else 70
    selected=((yy-32)**2+(xx-26)**2<=12**2)|((yy-32)**2+(xx-right)**2<=12**2)
    gray=np.where(selected,200,0).astype(np.uint8)
    return gray,np.repeat(gray[...,None],3,axis=2)


def metadata_size(value,seen=None):
    seen=set() if seen is None else seen
    if id(value) in seen:return 0
    seen.add(id(value))
    size=sys.getsizeof(value)
    if torch.is_tensor(value):return size  # CUDA backing has a separate budget.
    if isinstance(value,Mapping):
        return size+sum(metadata_size(k,seen)+metadata_size(v,seen) for k,v in value.items())
    if isinstance(value,(tuple,list)):
        return size+sum(metadata_size(child,seen) for child in value)
    if is_dataclass(value):
        return size+sum(metadata_size(getattr(value,field.name),seen) for field in fields(value))
    return size


_REFERENCE_SCRIPT=_PINS_SCRIPT+r'''
import sys
import cv2
import numpy as np
from scipy import ndimage
from skimage.segmentation import watershed
from plantcv import plantcv as pcv
assert version('plantcv') == '4.11.3'
assert version('opencv-python') == '4.11.0.86'
with np.load(sys.argv[1],allow_pickle=False) as values:
    gray=values['gray']
foreground=pcv.threshold.binary(gray,threshold=100,object_type='light')
eroded=pcv.erode(foreground,int(sys.argv[3]),int(sys.argv[4]))
components,count=ndimage.label(eroded)
ids=np.zeros(count+1,np.int32)
for label_id in range(1,count+1):
    ids[label_id]=np.flatnonzero(components==label_id)[0]+1
markers=ids[components]
distance,_=cv2.distanceTransformWithLabels(foreground,cv2.DIST_L2,5)
labels=watershed(-distance,markers,mask=foreground)
np.savez(sys.argv[2],foreground=foreground,eroded=eroded,markers=markers,distance=distance,labels=labels)
'''


class CoreRecoveryGraphContractTests(unittest.TestCase):

    pass


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
@functional_units
class CoreRecoveryGraphTests(CatalogueTestCase):
    def setUp(self):
        self.gray,self.rgb=fixture()
        self.gray_gpu=torch.as_tensor(self.gray,device='cuda')
        self.rgb_gpu=torch.as_tensor(self.rgb,device='cuda')
        self.recipe=selected_recipe()
        self.cache=PipelineAnalysisCache()
        self.frame=CoordinateFrame('source.decoded',self.gray.shape,'top-left','x-right,y-down','pixel')

    def run_recipe(self, recipe=None, **kwargs):
        return runner.run_native_recipe(recipe or self.recipe,self.gray_gpu,self.rgb_gpu,
            source_sha256='a'*64,coordinate_frame=self.frame,preparation={'gray':'explicit gray','rgb':'explicit RGB'},
            cache=self.cache,**kwargs)

    def check_test_actual_pinned_full_chain_and_pre_erosion_union(self):
        for size,iterations in ((3,1),(5,1),(5,2)):
            recipe=parameter(parameter(self.recipe,factory.ERODE,'ksize',size),factory.ERODE,'iterations',iterations)
            result=self.run_recipe(recipe)
            self.assertEqual(result.status,'executed')
            with tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary)
                np.savez(root/'input.npz',gray=self.gray)
                completed=subprocess.run([str(_REFERENCE),'-c',_REFERENCE_SCRIPT,
                    str(root/'input.npz'),str(root/'output.npz'),str(size),str(iterations)],
                    capture_output=True,text=True,timeout=120)
                self.assertEqual(completed.returncode,0,completed.stderr)
                with np.load(root/'output.npz',allow_pickle=False) as expected:
                    for stage,port,name in ((factory.FOREGROUND,'mask','foreground'),
                            (factory.ERODE,'mask','eroded'),(factory.CORES,'markers','markers'),
                            (factory.DISTANCE,'distance','distance'),(factory.PARTITION,'labels','labels')):
                        np.testing.assert_array_equal(result.node(stage).outputs[port].cpu().numpy(),expected[name])
            value=result.node(factory.PARTITION).outputs['partition']
            self.assertIs(value['tensors']['original_foreground'],result.node(factory.FOREGROUND).outputs['mask'])
            self.assertIs(result.labels,result.node(factory.PARTITION).outputs['labels'])
            self.assertEqual(result.labels_coordinate_frame,self.frame)
            self.assertEqual(value['metadata']['binding']['source_sha256'],'a'*64)
            self.assertTrue(all(o['assigned_pixels']>o['seed_pixels'] for o in value['metadata']['objects']))

    def check_test_radius_and_policy_reuse_partition_and_control_measurement_calculation(self):
        first=self.run_recipe()
        self.assertEqual(len(first.node(factory.CORES).outputs['inventory']['objects']),2)
        initial=first.node(factory.ANNOTATION).outputs['metadata']['objects']
        self.assertTrue(all(not o['measurement_permitted'] for o in initial))
        enlarged=parameter(self.recipe,factory.ANNOTATION,'uncertainty_radius',3)
        changed=self.run_recipe(enlarged)
        self.assertEqual(self.cache.last_computed_nodes,(factory.ANNOTATION,factory.MEASUREMENTS))
        self.assertIs(changed.labels,first.labels)
        current=changed.node(factory.ANNOTATION).outputs['metadata']['objects']
        self.assertGreater(sum(o['uncertainty_pixels'] for o in current),sum(o['uncertainty_pixels'] for o in initial))
        flagged=parameter(enlarged,factory.ANNOTATION,'measurement_policy','flag_uncertain')
        measured=self.run_recipe(flagged)
        self.assertEqual(self.cache.last_computed_nodes,(factory.ANNOTATION,factory.MEASUREMENTS))
        self.assertIs(measured.labels,first.labels)
        for sample,variables in measured.node(factory.MEASUREMENTS).outputs['observations'].items():
            self.assertGreater(variables['assigned_area']['value'],0)
            self.assertTrue(variables['assigned_area']['available'])
            self.assertFalse(variables['assigned_area']['scientific_validation'])
            self.assertIsNone(first.node(factory.MEASUREMENTS).outputs['observations'][sample]['assigned_area']['value'])

    def check_test_roi_preserves_ids_and_reports_wholly_and_partly_removed_cores(self):
        self.gray,self.rgb=fixture(False)
        self.gray_gpu,self.rgb_gpu=torch.as_tensor(self.gray,device='cuda'),torch.as_tensor(self.rgb,device='cuda')
        first=self.run_recipe()
        inventory=first.node(factory.CORES).outputs['inventory']
        roi={'shape':list(self.gray.shape),'bounds':[0,0,50,64]}
        recipe=parameter(self.recipe,factory.VALIDITY,'validity_roi',roi)
        recipe=parameter(recipe,factory.ANNOTATION,'measurement_policy','flag_uncertain')
        changed=self.run_recipe(recipe)
        self.assertEqual(set(self.cache.last_computed_nodes),
            {factory.VALIDITY,factory.CLIP,factory.PARTITION,factory.ANNOTATION,factory.MEASUREMENTS})
        self.assertIs(changed.node(factory.CORES).outputs['markers'],first.node(factory.CORES).outputs['markers'])
        self.assertEqual(changed.node(factory.CORES).outputs['inventory'],inventory)
        objects=changed.node(factory.ANNOTATION).outputs['metadata']['objects']
        self.assertEqual([o['label_id'] for o in objects],[o['label_id'] for o in inventory['objects']])
        self.assertTrue(objects[0]['measurement_permitted'])
        self.assertEqual(objects[1]['reasons'],('disappeared_by_validity',))
        self.assertFalse(objects[1]['measurement_permitted'])
        self.assertEqual(objects[1]['clipped_core_pixels'],objects[1]['original_core_pixels'])
        self.assertIsNone(changed.node(factory.MEASUREMENTS).outputs['observations']['core_'+str(objects[1]['label_id'])]['assigned_area']['value'])
        partial=parameter(recipe,factory.VALIDITY,'validity_roi',{'shape':list(self.gray.shape),'bounds':[0,0,26,64]})
        clipped=self.run_recipe(partial)
        obj=clipped.node(factory.ANNOTATION).outputs['metadata']['objects'][0]
        self.assertGreater(obj['seed_pixels'],0)
        self.assertGreater(obj['clipped_core_pixels'],0)
        self.assertIn('invalid_support',obj['reasons'])
        self.assertFalse(obj['uncontested_visible_partition'])
        self.assertEqual(clipped.coordinate_frame,self.frame)


    def check_test_nested_products_immutable_cache_mutation_regenerates_and_callback_mutation_clears(self):
        first=self.run_recipe()
        product=first.node(factory.PARTITION).outputs['partition']
        with self.assertRaises(TypeError):
            product['tensors']['labels']=first.labels.clone()
        with self.assertRaises(TypeError):
            product['metadata']['objects'][0]['assigned_pixels']=1
        first.labels.zero_()
        fresh=self.run_recipe()
        self.assertEqual(set(self.cache.last_computed_nodes),{factory.PARTITION,factory.ANNOTATION,factory.MEASUREMENTS})
        self.assertGreater(int(torch.count_nonzero(fresh.labels)),0)
        calls=0
        def count():
            nonlocal calls
            calls+=1
            return False
        self.run_recipe(cancel=count)
        total,calls=calls,0
        def mutate():
            nonlocal calls
            calls+=1
            if calls == total-1:
                fresh.labels.add_(1)
            return False
        with self.assertRaises(runner.NativeRecipeError):
            self.run_recipe(cancel=mutate)
        self.assertFalse(any(name.startswith('reference.recipe:'+factory.RECIPE_ID+':') for name in self.cache.values))





    def check_test_core_producer_real_caps_batch_boundaries_and_bounded_failure(self):
        for pixels in (4095,4096,4097,65535,65536,65537):
            with self.subTest(empty_pixels=pixels):
                output=factory._core_labels(torch.zeros((1,pixels),dtype=torch.uint8,device='cuda'),None)
                self.assertEqual(output['inventory']['objects'],())
                self.assertFalse(bool(torch.count_nonzero(output['markers'])))
        for pixels in (2048,2049):
            output=factory._core_labels(torch.full((1,pixels),255,dtype=torch.uint8,device='cuda'),None)
            self.assertTrue(bool(torch.all(output['markers']==1)))
            self.assertEqual(output['inventory']['objects'][0]['original_core_pixels'],pixels)
        mask=torch.zeros((1,65535),dtype=torch.uint8,device='cuda');mask[:,::2]=255
        output=factory._core_labels(mask,None)
        objects=output['inventory']['objects']
        self.assertEqual(len(objects),32768)
        self.assertEqual([o['label_id'] for o in objects],list(range(1,65536,2)))
        self.assertTrue(all(o['original_core_pixels']==1 for o in objects))
        larger=torch.zeros((1,65537),dtype=torch.uint8,device='cuda');larger[:,::2]=255
        with self.assertRaises(factory.core.CoreRecoveryResourceError):factory._core_labels(larger,None)
        actual_program=factory.fixed_kernel_program(factory._CUDA_SOURCE,mask.device)
        class Program:
            def launch_1d(self,*args,**kwargs):return actual_program.launch_1d(*args,**kwargs)
            def launch(self,name,tensors,scalars):
                if name=='recovery_graph_cores':scalars=(*scalars[:2],1,*scalars[3:])
                return actual_program.launch(name,tensors,scalars)
        with patch.object(factory,'fixed_kernel_program',return_value=Program()):
            with self.assertRaises(factory.core.CoreRecoveryResourceError):
                factory._core_labels(torch.full((1,2),255,dtype=torch.uint8,device='cuda'),None)
        pixel_limit=(factory.core.MAX_LIVE_BYTES-32*1024*1024-factory.MAX_CORE_LABELS*256)//64
        self.assertEqual(pixel_limit,16121856)
        factory.retained_bound(pixel_limit)
        with self.assertRaises(factory.core.CoreRecoveryResourceError):factory.retained_bound(pixel_limit+1)
        calls=0
        def cancel():
            nonlocal calls
            calls+=1
            return calls==5
        with patch.object(factory,'CORE_BATCH',3),self.assertRaises(factory.core.CoreRecoveryCancelled):
            factory._core_labels(torch.full((1,9),255,dtype=torch.uint8,device='cuda'),cancel)

    def check_test_graph_metadata_maximum_actual_peak_compact_signatures_and_early_rejection(self):
        maximum=(factory.core.MAX_CPU_METADATA_BYTES-factory.GRAPH_METADATA_FIXED_BYTES)//factory.GRAPH_METADATA_BYTES_PER_LABEL
        self.assertEqual(maximum,8064)
        self.assertEqual(factory.graph_metadata_bound(maximum),64*1024*1024)
        for count in (maximum+1,True,-1,32769):
            with self.assertRaises(factory.core.CoreRecoveryResourceError):factory.graph_metadata_bound(count)
        with patch.object(factory.core,'recover_core_labels') as recover:
            with self.assertRaises(factory.core.CoreRecoveryResourceError):
                factory.execute_stage('recovery.partition',{'inventory':{'objects':(None,)*(maximum+1)}},{},None)
            recover.assert_not_called()
        mask=torch.zeros((1,2*maximum-1),dtype=torch.uint8,device='cuda');mask[:,::2]=255
        validity=torch.full_like(mask,255);elevation=torch.zeros(mask.shape,dtype=torch.float32,device='cuda')
        binding={'coordinate_frame':CoordinateFrame('source.decoded',tuple(mask.shape),
            'top-left','x-right,y-down','pixel').to_dict(),'source_sha256':'c'*64}
        factory._core_labels(torch.zeros((1,1),dtype=torch.uint8,device='cuda'),None)
        tracemalloc.start()
        try:
            produced=factory._core_labels(mask,None)
            partition=factory.execute_stage('recovery.partition',{'foreground':mask,'markers':produced['markers'],
                'validity':validity,'elevation':elevation,'inventory':produced['inventory'],'source_frame':binding},{},None)
            annotation=factory.execute_stage('recovery.annotation',{'partition':partition['partition']},
                {'uncertainty_radius':0,'measurement_policy':'flag_uncertain'},None)
            measured=factory.execute_stage('recovery.measurements',{'annotation':annotation['annotation']},{},None)
            products=(produced,partition,annotation,measured)
            with (patch.object(torch.Tensor,'cpu',side_effect=AssertionError('No raster transfer in cache signature')),
                  patch.object(torch.Tensor,'tolist',side_effect=AssertionError('No raster transfer in cache signature'))):
                signatures=tuple(runner._compact_recovery_signature(value) for value in products)
            _,peak=tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        retained=metadata_size((products,signatures))
        self.assertLessEqual(peak,factory.graph_metadata_bound(maximum))
        self.assertLessEqual(retained,factory.graph_metadata_bound(maximum))
        self.assertTrue(all(len(repr(value))<160 for value in signatures))
        self.assertEqual(len(measured['observations']),maximum)
        expected=[obj['label_id'] for obj in produced['inventory']['objects']]
        self.assertEqual(list(measured['observations']),['core_'+str(i) for i in expected])
        self.assertTrue(all(row['assigned_area']['value']==1 for row in measured['observations'].values()))
        initial=runner._compact_recovery_signature({'x':mask})
        mask.zero_()
        self.assertNotEqual(initial,runner._compact_recovery_signature({'x':mask}))
        over=torch.zeros((4,3*(maximum+1)+1),dtype=torch.uint8,device='cuda')
        over[1:3,1::3]=200;over[1:3,2::3]=200
        recipe=parameter(self.recipe,factory.ERODE,'ksize',2)
        result=runner.run_native_recipe(recipe,over,over[...,None].repeat(1,1,3),source_sha256='d'*64,
            coordinate_frame=CoordinateFrame('source.decoded',tuple(over.shape),'top-left','x-right,y-down','pixel'),
            preparation={'gray':'explicit sparse gray','rgb':'gray repeat'})
        self.assertEqual(result.status,'partial')
        self.assertEqual(len(result.node(factory.CORES).outputs['inventory']['objects']),maximum+1)
        self.assertEqual(result.node(factory.PARTITION).status,'failed')
        self.assertIn('compact CPU metadata label limit',result.node(factory.PARTITION).reason)
        self.assertIsNone(result.labels)
        self.assertEqual(result.node(factory.ANNOTATION).status,'not_run')
        self.assertEqual(result.node(factory.MEASUREMENTS).status,'not_run')
        print(f'\nMaximum graph metadata K={maximum}: retained {retained:,} bytes; traced peak {peak:,}; '
              f'graph-owned limit {factory.graph_metadata_bound(maximum):,} (explicit export has separate caps).',flush=True)



    def check_test_committed_full_graph_actual_source_and_dense_uhd_producer_peaks(self):
        from seedvision.cuda.core_recovery import _backing_bytes
        root=Path(__file__).parents[1]
        gray=cv2.imread(str(root/'images'/'IMG_9632c.JPG'),cv2.IMREAD_GRAYSCALE)
        self.assertEqual(gray.shape,(2080,3120))
        gray_gpu=torch.as_tensor(gray,device='cuda');rgb_gpu=gray_gpu[...,None].repeat(1,1,3)
        frame=CoordinateFrame('source.decoded',gray.shape,'top-left','x-right,y-down','pixel')
        cache=PipelineAnalysisCache()
        torch.cuda.synchronize();before=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter()
        result=runner.run_native_recipe(self.recipe,gray_gpu,rgb_gpu,source_sha256='b'*64,
            coordinate_frame=frame,preparation={'gray':'committed fixture','rgb':'explicit gray repeat'},cache=cache)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        self.assertEqual(result.status,'executed')
        peak=torch.cuda.max_memory_allocated()-before+_backing_bytes((gray_gpu,rgb_gpu))
        self.assertLessEqual(peak,factory.retained_bound(gray.size))
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary);np.savez(directory/'input.npz',gray=gray)
            completed=subprocess.run([str(_REFERENCE),'-c',_REFERENCE_SCRIPT,str(directory/'input.npz'),
                str(directory/'output.npz'),'5','1'],capture_output=True,text=True,timeout=120)
            self.assertEqual(completed.returncode,0,completed.stderr)
            with np.load(directory/'output.npz',allow_pickle=False) as expected:
                for stage,port,name in ((factory.CORES,'markers','markers'),(factory.PARTITION,'labels','labels')):
                    np.testing.assert_array_equal(result.node(stage).outputs[port].cpu().numpy(),expected[name])
        print(f'\nFull graph committed fixture: {elapsed:.3f}s; CUDA input plus additional peak {peak:,}; '
              f'bound {factory.retained_bound(gray.size):,}; cores '
              f"{len(result.node(factory.CORES).outputs['inventory']['objects'])}.",flush=True)
        del result,cache,gray_gpu,rgb_gpu
        mask=torch.full((2160,3840),255,dtype=torch.uint8,device='cuda')
        torch.cuda.synchronize();before=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter();output=factory._core_labels(mask,None);torch.cuda.synchronize()
        elapsed=time.perf_counter()-start
        self.assertTrue(bool(torch.all(output['markers']==1)))
        self.assertEqual(output['inventory']['objects'][0]['original_core_pixels'],mask.numel())
        peak=torch.cuda.max_memory_allocated()-before+_backing_bytes((mask,))
        producer_bound=_backing_bytes((mask,))+8*mask.numel()+8*factory.MAX_CORE_LABELS+32+factory.core.BACKEND_HEADROOM_BYTES
        self.assertLessEqual(peak,producer_bound)
        print(f'Dense UHD core producer: {elapsed:.3f}s; CUDA input plus additional peak {peak:,}; '
              f'producer bound {producer_bound:,}; at most {(2*mask.numel()+factory.CORE_BATCH-1)//factory.CORE_BATCH} batches.',flush=True)


    def test_workflow_lifecycle(self):
        """4 related CoreRecoveryGraphTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_actual_pinned_full_chain_and_pre_erosion_union)
        self.scenario(self.check_test_radius_and_policy_reuse_partition_and_control_measurement_calculation)
        self.scenario(self.check_test_roi_preserves_ids_and_reports_wholly_and_partly_removed_cores)
        self.scenario(self.check_test_committed_full_graph_actual_source_and_dense_uhd_producer_peaks)

    def test_validation_and_recovery(self):
        """3 related CoreRecoveryGraphTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_nested_products_immutable_cache_mutation_regenerates_and_callback_mutation_clears)
        self.scenario(self.check_test_core_producer_real_caps_batch_boundaries_and_bounded_failure)
        self.scenario(self.check_test_graph_metadata_maximum_actual_peak_compact_signatures_and_early_rejection)


class CoreRecoveryDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
        from PySide6.QtWidgets import QApplication
        cls.application=QApplication.instance() or QApplication([])

    def wait_until(self,predicate,timeout=30):
        deadline=time.monotonic()+timeout
        while not predicate() and time.monotonic()<deadline:
            self.application.processEvents();time.sleep(.01)
        self.application.processEvents()
        self.assertTrue(predicate(),'Recovery Qt worker did not complete.')





if __name__ == '__main__':
    unittest.main()
