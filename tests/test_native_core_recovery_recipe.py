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
    def test_additive_graph_roundtrip_controls_and_loaded_identity(self):
        before=build_recipe_catalogue().to_json()
        recipe=selected_recipe()
        runner._validate_recipe(recipe)
        self.assertEqual(RecipeSpec.from_json(recipe.to_json()),recipe)
        self.assertEqual(len(recipe.stages),11)
        self.assertEqual(recipe.provenance.reference_id,'seedfiddle_enhancement')
        self.assertIsNone(recipe.parameters_for_runner(factory.VALIDITY)['validity_roi'])
        self.assertEqual({p.parameter_id for p in recipe.stage(factory.ANNOTATION).parameters},
                         {'uncertainty_radius','measurement_policy'})
        for stage in recipe.stages:
            for p in stage.parameters:
                self.assertTrue(p.affects_calculation)
        runner._module_record(factory)
        runner._module_record(factory.core)
        self.assertEqual(build_recipe_catalogue().to_json(),before)
        with self.assertRaises(runner.NativeRecipeError):
            runner._validate_recipe(replace(recipe,stages=recipe.stages[:-1],layout=recipe.layout[:-1],
                dependencies=tuple(e for e in recipe.dependencies if e.target_stage != factory.MEASUREMENTS)))

    def test_compact_cache_signature_preserves_types_structure_and_rejects_unknowns(self):
        signature=runner._compact_recovery_signature
        cases=({'x':0},{'x':False},{'x':0.0},{'x':-0.0},{'x':'0'},
               {'x':None},{'x':[]},{'x':()},{'x':[1]},{'x':[2]})
        self.assertEqual(len({signature(value) for value in cases}),len(cases))
        self.assertEqual(signature({'b':2,'a':1}),signature({'a':1,'b':2}))
        for value in ({'x':object()},{1:'x'},{'x':'a'*65537},{'x':2**257}):
            with self.assertRaises(runner.NativeRecipeError):signature(value)
        cycle=[];cycle.append(cycle)
        with self.assertRaises(runner.NativeRecipeError):signature(cycle)
        deep=0
        for _ in range(34):deep=[deep]
        with self.assertRaises(runner.NativeRecipeError):signature(deep)


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class CoreRecoveryGraphTests(unittest.TestCase):
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

    def test_actual_pinned_full_chain_and_pre_erosion_union(self):
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

    def test_radius_and_policy_reuse_partition_and_control_measurement_calculation(self):
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

    def test_roi_preserves_ids_and_reports_wholly_and_partly_removed_cores(self):
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

    def test_erosion_threshold_and_invalid_roi_dependencies_never_reuse_stale_descendants(self):
        first=self.run_recipe()
        changed=self.run_recipe(parameter(self.recipe,factory.ERODE,'ksize',7))
        self.assertEqual(set(self.cache.last_computed_nodes),
            {factory.ERODE,factory.CORES,factory.CLIP,factory.PARTITION,factory.ANNOTATION,factory.MEASUREMENTS})
        self.assertIs(changed.node(factory.DISTANCE).outputs['distance'],first.node(factory.DISTANCE).outputs['distance'])
        self.assertFalse(torch.equal(changed.node(factory.CORES).outputs['markers'],first.node(factory.CORES).outputs['markers']))
        wrong=parameter(self.recipe,factory.VALIDITY,'validity_roi',{'shape':[63,96],'bounds':[0,0,50,63]})
        invalid=self.run_recipe(wrong)
        self.assertEqual(invalid.node(factory.VALIDITY).status,'failed')
        self.assertIsNone(invalid.labels)
        self.assertEqual(invalid.node(factory.ANNOTATION).status,'not_run')
        empty=self.run_recipe(parameter(self.recipe,factory.FOREGROUND,'threshold',220))
        self.assertEqual(empty.status,'executed')
        self.assertEqual(empty.node(factory.PARTITION).outputs['metadata']['status'],'empty_foreground')
        self.assertFalse(bool(torch.count_nonzero(empty.labels)))
        self.assertEqual(empty.node(factory.MEASUREMENTS).outputs['observations'],{})

    def test_nested_products_immutable_cache_mutation_regenerates_and_callback_mutation_clears(self):
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

    def test_final_loaded_closure_audit_and_callback_exception_clear_recovery_cache(self):
        for target,raises in (('_measure',False),('_CUDA_SOURCE',False),('_measure',True)):
            with self.subTest(target=target,raises=raises):
                self.run_recipe()
                original=getattr(factory,target)
                calls=0
                def replacement(annotation):return original(annotation)
                def callback():
                    nonlocal calls
                    calls+=1
                    if calls==3:  # Initial runtime audit has finished; source node is already watched.
                        setattr(factory,target,original+'\n// changed callback source\n' if target=='_CUDA_SOURCE' else replacement)
                        if raises:raise RuntimeError('Caller raised after changing its helper.')
                    return False
                with patch.object(factory,target,original):
                    with self.assertRaises(RuntimeError if raises else runner.NativeRecipeError):
                        self.run_recipe(cancel=callback)
                self.assertGreaterEqual(calls,3)
                self.assertFalse(any(name.startswith('reference.recipe:'+factory.RECIPE_ID+':') for name in self.cache.values))
        self.assertEqual(self.run_recipe().status,'executed')

    def test_prepared_cache_handoff_and_fresh_owned_source_callback_mutations_reject(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'source.png';self.assertTrue(cv2.imwrite(str(path),self.rgb))
            cache=PipelineAnalysisCache()
            baseline=runner.run_native_recipe_path(self.recipe,path,cache=cache)
            labels=baseline.labels.clone()
            for raises in (False,True):
                runner.run_native_recipe_path(self.recipe,path,cache=cache)
                prepared=cache.values['reference.prepared'][1]
                calls=0
                def mutate_cached():
                    nonlocal calls
                    calls+=1
                    if calls==3:  # path entry, preparation entry, then cached-return callback.
                        prepared.gray.zero_()
                        if raises:raise RuntimeError('Caller raised after changing prepared pixels.')
                    return False
                with self.assertRaises(runner.NativeRecipeError):
                    runner.run_native_recipe_path(self.recipe,path,cache=cache,cancel=mutate_cached)
                self.assertNotIn('reference.prepared',cache.values)
            restored=runner.run_native_recipe_path(self.recipe,path,cache=cache)
            torch.testing.assert_close(restored.labels,labels)
            original_inputs=runner.NativeRecipeInputs
            original_as_tensor=torch.as_tensor
            original_empty=torch.empty
            for target in ('bgr','gray_during_fill','gray_final','rgb_final'):
                captured=[]
                def as_tensor(*args,**kwargs):
                    tensor=original_as_tensor(*args,**kwargs)
                    if target=='bgr' and not captured:captured.append(tensor)
                    return tensor
                def empty(*args,**kwargs):
                    tensor=original_empty(*args,**kwargs)
                    if target=='gray_during_fill' and not captured:captured.append(tensor)
                    return tensor
                def inputs(*args,**kwargs):
                    prepared=original_inputs(*args,**kwargs)
                    if target in ('gray_final','rgb_final'):
                        captured.append(prepared.gray if target=='gray_final' else prepared.rgb)
                    return prepared
                fired=False
                def mutate_fresh():
                    nonlocal fired
                    if captured and not fired:
                        fired=True;captured[0].zero_()
                    return False
                with self.subTest(target=target),patch.object(torch,'as_tensor',as_tensor),patch.object(torch,'empty',empty),patch.object(runner,'NativeRecipeInputs',inputs):
                    with self.assertRaises(runner.NativeRecipeError):
                        runner.prepare_native_recipe_inputs(path,cache=PipelineAnalysisCache(),
                            cancel=mutate_fresh,_core_recovery=True)
                self.assertTrue(fired)

    def test_linear_core_labeling_exhaustive_four_connectivity_sparse_ids_and_caps(self):
        masks=[np.array([(value>>i)&1 for i in range(9)],np.uint8).reshape(3,3)*255 for value in range(512)]
        script=_PINS_SCRIPT+r'''
import sys
import numpy as np
from scipy import ndimage
with np.load(sys.argv[1],allow_pickle=False) as values:
    masks=values['masks']
results=[]
for mask in masks:
    labels,count=ndimage.label(mask)
    ids=np.zeros(count+1,np.int32)
    for i in range(1,count+1): ids[i]=np.flatnonzero(labels==i)[0]+1
    results.append(ids[labels])
np.save(sys.argv[2],np.asarray(results),allow_pickle=False)
'''
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            np.savez(root/'input.npz',masks=masks)
            completed=subprocess.run([str(_REFERENCE),'-c',script,str(root/'input.npz'),str(root/'output.npy')],
                                     capture_output=True,text=True,timeout=120)
            self.assertEqual(completed.returncode,0,completed.stderr)
            expected=np.load(root/'output.npy',allow_pickle=False)
        for mask,oracle in zip(masks,expected):
            result=factory._core_labels(torch.as_tensor(mask,device='cuda'),None)
            np.testing.assert_array_equal(result['markers'].cpu().numpy(),oracle)
            self.assertEqual(sum(o['original_core_pixels'] for o in result['inventory']['objects']),int(np.count_nonzero(mask)))
        diagonal=torch.tensor([[255,0],[0,255]],dtype=torch.uint8,device='cuda')
        with patch.object(factory,'MAX_CORE_LABELS',1):
            with self.assertRaises(factory.core.CoreRecoveryResourceError):
                factory._core_labels(diagonal,None)
        bad=diagonal.clone();bad[0,1]=1
        with self.assertRaises(ValueError):factory._core_labels(bad,None)

    def test_core_producer_workspaces_cancellation_and_joint_static_admission(self):
        original_watch=factory.core._Guard.watch
        for target in range(6):
            watched=[]
            mask=torch.full((13,17),255,dtype=torch.uint8,device='cuda')
            def watch(guard,*tensors):
                original_watch(guard,*tensors);watched.extend(tensors)
            def cancel():
                if len(watched)>target:watched[target].add_(1)
                return False
            with patch.object(factory.core._Guard,'watch',watch), self.assertRaises(factory.core.CoreRecoveryMutationError):
                factory._core_labels(mask,cancel)
        with self.assertRaises(runner.NativeRecipeCancelled):self.run_recipe(cancel=lambda:True)
        with patch.object(factory,'MAX_RASTER_PIXELS',1), patch.object(factory,'fixed_kernel_program') as compile:
            with self.assertRaises(factory.core.CoreRecoveryResourceError):self.run_recipe()
            compile.assert_not_called()
        stream=torch.cuda.Stream()
        with torch.cuda.stream(stream),patch.object(factory,'CORE_BATCH',3),patch.object(factory,'VALIDATION_TILE',7):
            mask=torch.full((19,23),255,dtype=torch.uint8,device='cuda')
            result=factory._core_labels(mask,None)
        stream.synchronize()
        self.assertTrue(bool(torch.all(result['markers']==1)))
        self.assertEqual(result['inventory']['objects'][0]['original_core_pixels'],19*23)

    def test_core_producer_real_caps_batch_boundaries_and_bounded_failure(self):
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

    def test_graph_metadata_maximum_actual_peak_compact_signatures_and_early_rejection(self):
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

    def test_engineering_isolated_and_touching_controls_preserve_visible_shapes(self):
        from seedvision.cuda.plantcv_watershed import masked_watershed
        metrics=[]
        yy,xx=np.indices(self.gray.shape)
        for touching in (False,True):
            self.gray,self.rgb=fixture(touching)
            self.gray_gpu,self.rgb_gpu=torch.as_tensor(self.gray,device='cuda'),torch.as_tensor(self.rgb,device='cuda')
            result=self.run_recipe()
            partition=result.node(factory.PARTITION).outputs['partition']
            tensors=partition['tensors']
            control=masked_watershed(tensors['elevation'],tensors['markers'],tensors['domain'])
            torch.testing.assert_close(result.labels,control)
            labels=result.labels.cpu().numpy()
            cores=tensors['markers'].cpu().numpy()
            instances=[(yy-32)**2+(xx-26)**2<=12**2,(yy-32)**2+(xx-(50 if touching else 70))**2<=12**2]
            instances[1]&=~instances[0]  # The constructed shared pixel belongs to the first shape.
            objects=result.node(factory.ANNOTATION).outputs['metadata']['objects']
            self.assertEqual(len(objects),2)
            for truth,obj in zip(instances,objects):
                recovered=labels==obj['label_id'];eroded=cores==obj['label_id']
                recovered_iou=np.count_nonzero(recovered&truth)/np.count_nonzero(recovered|truth)
                core_iou=np.count_nonzero(eroded&truth)/np.count_nonzero(eroded|truth)
                self.assertGreater(recovered_iou,core_iou)
                if not touching:
                    np.testing.assert_array_equal(recovered,truth)
                    self.assertTrue(obj['uncontested_visible_partition'])
                    self.assertTrue(obj['measurement_permitted'])
                else:
                    self.assertIn('inferred_shared_boundary',obj['reasons'])
                    self.assertFalse(obj['measurement_permitted'])
                metrics.append({'fixture':'touching' if touching else 'isolated','label_id':obj['label_id'],
                    'recovered_iou':recovered_iou,'erosion_only_iou':core_iou,
                    'false_positive_pixels':int(np.count_nonzero(recovered&~truth)),
                    'false_negative_pixels':int(np.count_nonzero(truth&~recovered)),
                    'uncertainty_pixels':obj['uncertainty_pixels']})
            self.assertEqual(partition['metadata']['unresolved_pixels'],0)
            self.assertEqual(int(np.count_nonzero(labels)),int(np.count_nonzero(self.gray)))
        print('\nSynthetic visible-shape engineering controls (not laboratory validation): '+json.dumps(metrics),flush=True)

    def test_erosion_lost_unknowns_all_excluded_and_canvas_contact_stay_explicit(self):
        self.gray=np.zeros((64,96),np.uint8)
        self.gray[4:6,4:6]=200;self.gray[20:44,40:64]=200
        self.rgb=np.repeat(self.gray[...,None],3,axis=2)
        self.gray_gpu,self.rgb_gpu=torch.as_tensor(self.gray,device='cuda'),torch.as_tensor(self.rgb,device='cuda')
        result=self.run_recipe()
        metadata=result.node(factory.ANNOTATION).outputs['metadata']
        self.assertEqual(len(metadata['objects']),1)
        self.assertEqual(metadata['unresolved_pixels'],4)
        self.assertNotIn('disappeared_by_validity',metadata['objects'][0]['reasons'])
        self.assertTrue(bool(torch.all(result.labels[4:6,4:6]==0)))
        selected=parameter(self.recipe,factory.VALIDITY,'validity_roi',{'shape':[64,96],'bounds':[0,0,2,2]})
        selected=parameter(selected,factory.ANNOTATION,'measurement_policy','flag_uncertain')
        excluded=self.run_recipe(selected)
        metadata=excluded.node(factory.ANNOTATION).outputs['metadata']
        self.assertEqual(metadata['partition_status'],'all_foreground_excluded_by_validity')
        self.assertEqual(metadata['validity_clipped_foreground_pixels'],580)
        self.assertEqual(metadata['unresolved_pixels'],0)
        self.assertEqual(metadata['objects'][0]['reasons'],('disappeared_by_validity',))
        self.assertFalse(metadata['objects'][0]['measurement_permitted'])
        self.gray[20:44,40:64]=0
        self.gray_gpu=torch.as_tensor(self.gray,device='cuda');self.rgb_gpu=self.gray_gpu[...,None].repeat(1,1,3)
        unseeded=self.run_recipe()
        metadata=unseeded.node(factory.ANNOTATION).outputs['metadata']
        self.assertEqual(metadata['partition_status'],'unresolved_foreground')
        self.assertEqual(metadata['unresolved_pixels'],4)
        self.assertEqual(metadata['objects'],())
        self.assertEqual(unseeded.node(factory.MEASUREMENTS).outputs['observations'],{})
        self.gray[:]=0;self.gray[:20,:20]=200
        self.gray_gpu=torch.as_tensor(self.gray,device='cuda');self.rgb_gpu=self.gray_gpu[...,None].repeat(1,1,3)
        bordered=self.run_recipe()
        obj=bordered.node(factory.ANNOTATION).outputs['metadata']['objects'][0]
        self.assertIn('canvas_boundary',obj['reasons'])
        self.assertFalse(obj['measurement_permitted'])
        np.testing.assert_array_equal(bordered.labels.cpu().numpy()>0,self.gray>0)

    def test_committed_full_graph_actual_source_and_dense_uhd_producer_peaks(self):
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

    def test_actual_archive_and_moved_project_companion_batch_preserve_experimental_status(self):
        from seedvision.export.native_results import export_native_recipe_result,load_native_recipe_result
        from seedvision.export.project_native_graphs import export_project_native_graphs
        from seedvision.persistence.project_analysis import ProjectAnalysisStore,ProjectImageSpec
        from seedvision.persistence.analysis_settings import analysis_settings_profile_from_graph
        from seedvision.pipeline import build_default_pipeline
        result=self.run_recipe()
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            archive=root/'archive'
            export_native_recipe_result(archive,self.recipe,result)
            loaded,arrays=load_native_recipe_result(archive)
            self.assertEqual(loaded['status'],'executed')
            self.assertEqual(loaded['recipe']['recipe_id'],factory.RECIPE_ID)
            raw=(archive/'native-results.json').read_text()
            self.assertNotIn('_snapshot',raw)
            self.assertTrue('inferred_shared_boundary' in raw)
            workspace=root/'workspace';workspace.mkdir()
            (workspace/'config').mkdir()
            shutil.copyfile(Path(__file__).parents[1]/'config/traits.json',workspace/'config/traits.json')
            source=workspace/'seeds.png';self.assertTrue(cv2.imwrite(str(source),self.rgb))
            master=ProjectAnalysisStore(workspace).capture_and_save(
                analysis_settings=analysis_settings_profile_from_graph(build_default_pipeline()),
                images=(ProjectImageSpec(source,self.gray.shape),),native_recipes=(self.recipe,),
                destination=workspace/'test.seedfiddle-project.json')
            moved=root/'moved';shutil.move(str(workspace),moved)
            restored=ProjectAnalysisStore(moved).load(moved/master.name)
            self.assertFalse(restored.issues)
            self.assertEqual(restored.document.native_recipes,(self.recipe,))
            reopened=runner.run_native_recipe_path(restored.document.native_recipes[0],moved/'seeds.png')
            self.assertEqual(reopened.status,'executed')
            torch.testing.assert_close(reopened.labels,result.labels)
            report=export_project_native_graphs((),moved/'seeds.png',root/'batch',
                expected_source_sha256=reopened.source_sha256,project_root=moved,
                native_recipes=restored.document.native_recipes)
            payload=json.loads(report.read_bytes())
            self.assertEqual(payload['graphs'][0]['status'],'executed')
            self.assertFalse(payload['scientifically_validated'])
            batch,_=load_native_recipe_result(root/'batch'/Path(payload['graphs'][0]['report']).parent)
            self.assertEqual(batch['recipe']['recipe_id'],factory.RECIPE_ID)


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

    def test_launcher_is_additive_and_opens_experimental_working_recipe(self):
        from seedvision.ui.reference_graph_dialog import ReferenceGraphDialog
        from seedvision.ui.native_recipe_dialog import NativeRecipeDialog
        dialog=ReferenceGraphDialog()
        captured=[]
        try:
            before=dialog.reference_graphs
            with patch.object(NativeRecipeDialog,'exec',lambda child:captured.append(child.recipe) or 0):
                dialog._core_recovery_button.click()
            self.assertEqual(captured[0].recipe_id,factory.RECIPE_ID)
            self.assertEqual(dialog.reference_graphs,before)
        finally:
            dialog.reject();dialog.deleteLater()

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_sparse_ids_separated_by_255_have_distinct_actual_preview_colours(self):
        from seedvision.ui.native_recipe_dialog import NativeRecipeDialog
        with tempfile.TemporaryDirectory() as temporary:
            gray=np.zeros((64,96),np.uint8)
            gray[8:13,8:13]=200;gray[10:15,71:76]=200
            path=Path(temporary)/'sparse.png';self.assertTrue(cv2.imwrite(str(path),gray))
            dialog=NativeRecipeDialog(selected_recipe());dialog.show()
            try:
                dialog.set_source_path(path);dialog.start_run();self.wait_until(lambda:dialog._worker is None)
                self.assertIsNotNone(dialog.result,dialog.status_label.text())
                labels=dialog.result.labels
                self.assertEqual(int(labels[12,73])-int(labels[10,10]),255)
                dialog._select_stage(factory.PARTITION)
                image=dialog.preview.pixmap().toImage()
                def colour(y,x):return image.pixelColor(int((x+.5)*image.width()/96),int((y+.5)*image.height()/64)).getRgb()
                self.assertNotEqual(colour(10,10),colour(12,73))
                self.assertEqual(colour(0,0)[:3],(0,0,0))
            finally:
                if dialog._worker is not None:
                    dialog.stop();self.wait_until(lambda:dialog._worker is None)
                dialog.reject();dialog.deleteLater()

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_actual_async_radius_policy_preview_save_and_keep(self):
        from seedvision.ui.native_recipe_dialog import NativeRecipeDialog
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            _,rgb=fixture()
            path=root/'seeds.png';self.assertTrue(cv2.imwrite(str(path),rgb))
            kept=[]
            dialog=NativeRecipeDialog(selected_recipe(),keep_in_project=lambda recipe:kept.append(recipe))
            dialog.show()
            try:
                dialog.set_source_path(path);dialog.start_run()
                self.wait_until(lambda:dialog._worker is None)
                self.assertIsNotNone(dialog.result,dialog.status_label.text())
                first=dialog.result.labels
                dialog._select_stage(factory.ANNOTATION)
                self.assertFalse(dialog.preview.pixmap().isNull())
                dialog.resize(1050,650);self.application.processEvents()
                editors=dialog.inspector._parameter_widgets
                dialog.inspector_scroll.ensureWidgetVisible(editors[0])
                self.application.processEvents()
                self.assertTrue(dialog.inspector_scroll.viewport().rect().intersects(
                    editors[0].rect().translated(editors[0].mapTo(dialog.inspector_scroll.viewport(),editors[0].rect().topLeft()))))
                editors[0].setText('2');editors[0].editingFinished.emit()
                # Updating a parameter rebuilds the inspector and its editors.
                dialog.inspector._parameter_widgets[1].setCurrentText('flag_uncertain')
                self.assertIsNone(dialog.result)
                dialog.start_run();self.wait_until(lambda:dialog._worker is None)
                self.assertIsNotNone(dialog.result,dialog.status_label.text())
                self.assertIs(first,dialog.result.labels)
                self.assertEqual(dialog.cache.last_computed_nodes,(factory.ANNOTATION,factory.MEASUREMENTS))
                dialog._select_stage(factory.MEASUREMENTS)
                self.assertIn('assigned_area',dialog.facts.toPlainText())
                self.assertIn('inferred_shared_boundary',dialog.facts.toPlainText())
                output=root/'recovery.seedfiddle-recipe.json'
                with patch('seedvision.ui.native_recipe_dialog.QFileDialog.getSaveFileName',return_value=(str(output),'')):
                    dialog._save_recipe()
                self.assertEqual(RecipeSpec.from_json(output.read_bytes()),dialog.recipe)
                dialog.project_button.click()
                self.assertEqual(kept,[dialog.recipe])
                dialog.inspector.parameters_reset.emit(factory.ANNOTATION)
                self.assertEqual(dialog.recipe.parameters_for_runner(factory.ANNOTATION)['uncertainty_radius'],0)
                self.assertEqual(dialog.recipe.parameters_for_runner(factory.ANNOTATION)['measurement_policy'],'withhold_uncertain')
            finally:
                if dialog._worker is not None:
                    dialog.stop();self.wait_until(lambda:dialog._worker is None)
                dialog.reject();dialog.deleteLater()


if __name__ == '__main__':
    unittest.main()
