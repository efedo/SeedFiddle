"""Actual saved recovery inputs, compact contracts, guards and desktop replay."""
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
from seedvision.benchmarking import native_core_recovery_recipe as recovery, native_recovery_replay as contracts
from seedvision.benchmarking import native_recipe as runner, native_stage_replay as replay
from seedvision.export import native_results as archives
from seedvision.ui.native_recipe_dialog import NativeRecipeDialog
from tests.test_native_core_recovery_recipe import selected_recipe, parameter, fixture

STAGES=(recovery.CORES,recovery.VALIDITY,recovery.CLIP,recovery.PARTITION,recovery.ANNOTATION,recovery.MEASUREMENTS)
ARRAY_COUNTS={recovery.CORES:2,recovery.VALIDITY:2,recovery.CLIP:3,
    recovery.PARTITION:9,recovery.ANNOTATION:10,recovery.MEASUREMENTS:0}


def hashes(directory):
    return {str(path.relative_to(directory)):sha256(path.read_bytes()).hexdigest()
        for path in directory.rglob('*') if path.is_file()}


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class RecoveryStageReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary=tempfile.TemporaryDirectory(dir='artifacts');cls.root=Path(cls.temporary.name)
        cls.image=cls.root/'source.png';gray,rgb=fixture(touching=False)
        assert cv2.imwrite(str(cls.image),rgb)
        cls.cases={}
        for name,roi in (('full',None),('roi',{'shape':[64,96],'bounds':[0,1,60,64]})):
            recipe=parameter(selected_recipe(),recovery.VALIDITY,'validity_roi',roi)
            saved=runner.run_native_recipe_path(recipe,cls.image)
            assert saved.node(recovery.CORES).status=='executed'
            assert saved.node(recovery.VALIDITY).status=='executed'
            assert saved.node(recovery.CLIP).status=='executed'
            for stage in STAGES:assert saved.node(stage).status=='executed',saved.node(stage).reason
            directory=cls.root/name;archives.export_native_recipe_result(directory,recipe,saved)
            cls.cases[name]=(recipe,saved,directory)

    @classmethod
    def tearDownClass(cls):cls.temporary.cleanup()

    def attempt(self,stage,case='roi',directory=None,cancel=None,change=None,observer=None):
        calls=[];selected=[]
        def profile(frame,event,arg):
            if event=='call' and frame.f_code is runner._dispatch.__code__:calls.append(frame.f_locals['operation'])
            elif event=='call' and frame.f_code is archives.load_native_recipe_result.__code__:selected.append(frame.f_locals['array_names'])
            elif event=='return' and change is not None and frame.f_code is archives.read_native_recipe_inventory.__code__ and arg is not None:change(*arg)
            if observer is not None:observer(frame,event,arg)
        previous=sys.getprofile();sys.setprofile(profile)
        try:result=replay.replay_native_stage(directory or self.cases[case][2],stage,cancel=cancel)
        finally:sys.setprofile(previous)
        return result,calls,selected

    def changed(self,case='roi',**nodes):
        recipe,saved,_=self.cases[case]
        changed=replace(saved,nodes=tuple(nodes.get(node.stage_id,node) for node in saved.nodes))
        directory=self.root/('changed-'+str(len(tuple(self.root.iterdir()))))
        archives.export_native_recipe_result(directory,recipe,changed)
        return directory

    def test_operations_match_every_product_select_direct_arrays_and_preserve_archives(self):
        for case,(_,saved,directory) in self.cases.items():
            before=hashes(directory)
            for stage in STAGES:
                with self.subTest(case=case,stage=stage):
                    result,calls,selected=self.attempt(stage,case)
                    self.assertEqual(result.status,'matched',result.reason)
                    self.assertEqual(calls,[saved.node(stage).operation_id]);self.assertEqual(len(selected),1)
                    self.assertEqual(len(selected[0]),ARRAY_COUNTS[stage])
                    self.assertEqual(set(result.outputs),set(saved.node(stage).outputs))
                    for name,value in result.outputs.items():
                        if torch.is_tensor(value):self.assertTrue(torch.equal(value,saved.node(stage).outputs[name]))
                    self.assertEqual(set(result.metadata['comparison']['by_port']),set(result.outputs))
                    self.assertEqual(result.metadata['frame_scope'],'source');self.assertFalse(result.scientifically_validated)
                    if stage==recovery.PARTITION:
                        self.assertIs(result.outputs['partition']['tensors']['labels'],result.outputs['labels'])
                        self.assertEqual(set(result.metadata['comparison']['by_port']['partition']['by_field']),
                            {*contracts.PARTITION_TENSORS,'metadata'})
                    elif stage==recovery.ANNOTATION:
                        self.assertIs(result.outputs['annotation']['uncertainty_band'],result.outputs['uncertainty_band'])
                    elif stage==recovery.MEASUREMENTS:
                        self.assertEqual(result.metadata['plan']['selected_bytes'],0)
                        self.assertIn('not recomputed',result.metadata['upstream_statistics'])
            self.assertEqual(hashes(directory),before)

    def test_deleted_origin_is_not_opened(self):
        raw=self.image.read_bytes();self.image.unlink()
        try:
            for stage in STAGES:
                result,_,_=self.attempt(stage);self.assertEqual(result.status,'matched',result.reason)
        finally:self.image.write_bytes(raw)

    def test_valid_wrong_pixels_and_inventory_leaves_report_mismatch(self):
        saved=self.cases['roi'][1];node=saved.node(recovery.CORES)
        objects=[dict(item) for item in node.outputs['inventory']['objects']]
        objects[0]['original_core_pixels']+=1;objects[1]['original_core_pixels']-=1
        inventory={**node.outputs['inventory'],'objects':tuple(objects)}
        result,_,_=self.attempt(recovery.CORES,directory=self.changed(**{recovery.CORES:replace(node,outputs={**node.outputs,'inventory':inventory})}))
        self.assertEqual(result.status,'mismatch',result.reason)
        self.assertEqual(result.metadata['comparison']['mismatched_elements'],2)
        for stage,port,value in ((recovery.CORES,'markers',1),(recovery.VALIDITY,'validity',255),
                (recovery.CLIP,'markers',1)):
            node=saved.node(stage);raster=node.outputs[port].clone();raster[0,0]=value
            result,_,_=self.attempt(stage,directory=self.changed(**{stage:replace(node,outputs={**node.outputs,port:raster})}))
            self.assertEqual(result.status,'mismatch',result.reason)
            self.assertEqual(result.metadata['comparison']['mismatched_elements'],1)

    def test_bad_headers_source_binding_producer_keys_and_controls_fail_before_load(self):
        def bad_key(report,specs):
            next(node for node in report['nodes'] if node['stage_id']==recovery.CORES)['input_keys']['mask']='0'*64
        def bad_header(report,specs):
            node=next(node for node in report['nodes'] if node['stage_id']==recovery.ERODE)
            specs[dict(node['outputs'])['mask']['name']]['dtype']='|b1'
        def bad_binding(report,specs):
            node=next(node for node in report['nodes'] if node['stage_id']==recovery.SOURCE)
            value=dict(node['outputs'])['source_frame']
            next(item for item in value['items'] if item[0]=='source_sha256')[1]='2'*64
        def bad_control(report,specs):
            node=next(node for node in report['nodes'] if node['stage_id']==recovery.ERODE)
            next(item for item in node['parameters']['items'] if item[0]=='ksize')[1]=7
        for mutate in (bad_key,bad_header,bad_binding,bad_control):
            result,calls,selected=self.attempt(recovery.CORES,change=mutate)
            self.assertEqual(result.status,'invalid_archive',result.reason);self.assertFalse(calls);self.assertFalse(selected)

    def test_malformed_binary_labels_inventory_and_metadata_are_invalid(self):
        saved=self.cases['roi'][1]
        for producer,port,stage,bad in ((recovery.ERODE,'mask',recovery.CORES,1),
                (recovery.CORES,'markers',recovery.CORES,6145),(recovery.FOREGROUND,'mask',recovery.VALIDITY,1),
                (recovery.CORES,'markers',recovery.CLIP,6145),(recovery.VALIDITY,'validity',recovery.CLIP,1)):
            node=saved.node(producer);value=node.outputs[port].clone();value[0,0]=bad
            result,calls,_=self.attempt(stage,directory=self.changed(**{producer:replace(node,outputs={**node.outputs,port:value})}))
            self.assertEqual(result.status,'invalid_archive',result.reason);self.assertFalse(calls)
        node=saved.node(recovery.CORES);value={**node.outputs['inventory'],'core_pixels':True}
        result,calls,selected=self.attempt(recovery.CORES,directory=self.changed(**{recovery.CORES:replace(node,outputs={**node.outputs,'inventory':value})}))
        self.assertEqual(result.status,'invalid_archive',result.reason);self.assertFalse(calls);self.assertFalse(selected)
        node=saved.node(recovery.VALIDITY);value={**node.outputs['metadata'],'bounds_xyxy':(0,0,96,64)}
        result,calls,selected=self.attempt(recovery.VALIDITY,directory=self.changed(**{recovery.VALIDITY:replace(node,outputs={**node.outputs,'metadata':value})}))
        self.assertEqual(result.status,'invalid_archive',result.reason);self.assertFalse(calls);self.assertFalse(selected)

    def test_lower_replay_limit_rejects_before_runtime_loading_or_dispatch(self):
        for stage in STAGES:
            with patch.object(replay,'MAX_REPLAY_CUDA_BYTES',1),patch.object(replay,'_runtime') as runtime:
                result,calls,selected=self.attempt(stage)
                self.assertEqual(result.status,'unavailable',result.reason)
                runtime.assert_not_called();self.assertFalse(calls);self.assertFalse(selected)

    def test_cancel_bool_first_and_final_callbacks_withhold_all_products(self):
        for stage in STAGES:
            result,_,_=self.attempt(stage,cancel=lambda:True)
            self.assertEqual(result.status,'cancelled');self.assertFalse(result.outputs)
            result,_,_=self.attempt(stage,cancel=lambda:1)
            self.assertEqual(result.status,'failed');self.assertFalse(result.outputs)
            def final():
                frame=sys._getframe()
                while frame is not None:
                    if frame.f_code is replay.replay_native_stage.__code__:
                        return bool(frame.f_locals.get('comparisons'))
                    frame=frame.f_back
                return False
            result,_,_=self.attempt(stage,cancel=final)
            self.assertEqual(result.status,'cancelled',result.reason);self.assertFalse(result.outputs)

    def test_private_workspaces_mutation_followed_by_callback_exception_is_failure(self):
        for target in range(6):
            watched=[];mutated=[]
            def observe(frame,event,arg):
                if event=='return' and frame.f_code is recovery.core._Guard.watch.__code__:
                    watched.extend(frame.f_locals['tensors'])
            def cancel():
                if len(watched)>target and not mutated:
                    mutated.append(True);watched[target].add_(1);raise ValueError('caller exception after mutation')
                return False
            result,_,_=self.attempt(recovery.CORES,cancel=cancel,observer=observe)
            self.assertTrue(mutated);self.assertEqual(result.status,'failed',result.reason);self.assertFalse(result.outputs)
            self.assertIn('changed',result.reason)

    def test_final_compact_parameter_and_tensor_mutation_are_failures(self):
        for stage,target in ((recovery.VALIDITY,'parameters'),(recovery.VALIDITY,'output'),
                (recovery.CORES,'output'),(recovery.CLIP,'output')):
            mutated=[]
            def cancel():
                frame=sys._getframe()
                while frame is not None:
                    if frame.f_code is replay.replay_native_stage.__code__:
                        values=frame.f_locals
                        if target=='parameters' and values.get('recovery_watched') and 'parameters' in values and not mutated:
                            values['parameters']['validity_roi']=None;mutated.append(True)
                        elif target=='output' and values.get('outputs') and not mutated:
                            next(value for value in values['outputs'].values() if torch.is_tensor(value)).add_(1);mutated.append(True)
                        break
                    frame=frame.f_back
                return False
            result,_,_=self.attempt(stage,cancel=cancel)
            self.assertTrue(mutated);self.assertEqual(result.status,'failed',result.reason);self.assertFalse(result.outputs)

    def test_empty_diagonal_sparse_ids_and_maximum_inventory_replay_without_partition_cap(self):
        for index,mask in enumerate((np.zeros((2,3),np.uint8),
                np.array([[255,255,0],[255,255,255],[0,255,255]],np.uint8),
                np.where(np.arange(98304)[None,:]%3!=2,255,0).astype(np.uint8))):
            image=self.root/f'cores-{index}.png';cv2.imwrite(str(image),np.repeat(mask[...,None],3,axis=2))
            recipe=parameter(selected_recipe(),recovery.ERODE,'ksize',2)
            saved=runner.run_native_recipe_path(recipe,image)
            self.assertEqual(saved.node(recovery.CORES).status,'executed',saved.node(recovery.CORES).reason)
            directory=self.root/f'cores-{index}';archives.export_native_recipe_result(directory,recipe,saved)
            result,calls,_=self.attempt(recovery.CORES,directory=directory)
            self.assertEqual(result.status,'matched',result.reason);self.assertEqual(calls,['recovery.core_labels'])
            objects=result.outputs['inventory']['objects']
            self.assertEqual(len(objects),(0,2,32768)[index])
            if index==1:self.assertEqual(tuple(item['label_id'] for item in objects),(1,9))
            if index==2:
                self.assertEqual(objects[-1]['label_id'],98303)
                self.assertEqual(saved.node(recovery.PARTITION).status,'failed')
                self.assertIn('compact CPU metadata',saved.node(recovery.PARTITION).reason)
            result,calls,selected=self.attempt(recovery.CLIP,directory=directory)
            self.assertEqual(result.status,'matched',result.reason)
            self.assertEqual(calls,['recovery.clip_cores']);self.assertEqual(len(selected[0]),3)
            self.assertTrue(torch.equal(result.outputs['markers'],saved.node(recovery.CORES).outputs['markers']))

    def test_clipping_uses_half_open_support_and_preserves_surviving_sparse_ids(self):
        for case,(_,saved,_) in self.cases.items():
            markers=saved.node(recovery.CORES).outputs['markers']
            validity=saved.node(recovery.VALIDITY).outputs['validity']
            before=markers.clone(),validity.clone()
            inputs=[]
            def observe(frame,event,arg):
                if event=='call' and frame.f_code is runner._dispatch.__code__:
                    inputs.append(frame.f_locals['inputs'])
            result,_,_=self.attempt(recovery.CLIP,case,observer=observe)
            self.assertEqual(result.status,'matched',result.reason)
            expected=torch.where(validity==255,markers,torch.zeros_like(markers))
            self.assertTrue(torch.equal(result.outputs['markers'],expected))
            self.assertTrue(torch.equal(markers,before[0]));self.assertTrue(torch.equal(validity,before[1]))
            self.assertNotEqual(result.outputs['markers'].untyped_storage().data_ptr(),
                inputs[0]['markers'].untyped_storage().data_ptr())
            if case=='roi':
                self.assertTrue(bool((markers[validity==0]>0).any().item()))
                self.assertTrue(bool((result.outputs['markers'][validity==255]>0).any().item()))
                self.assertFalse(bool((result.outputs['markers'][validity==0]>0).any().item()))

    def test_clipping_rejects_malformed_ancestor_inventory_and_support_metadata_before_upload(self):
        def bad_inventory(report,specs):
            node=next(node for node in report['nodes'] if node['stage_id']==recovery.CORES)
            value=dict(node['outputs'])['inventory']
            next(item for item in value['items'] if item[0]=='core_pixels')[1]=True
        def bad_support(report,specs):
            node=next(node for node in report['nodes'] if node['stage_id']==recovery.VALIDITY)
            value=dict(node['outputs'])['metadata']
            next(item for item in value['items'] if item[0]=='bounds_xyxy')[1]['items'][0]=1
        for change in (bad_inventory,bad_support):
            result,calls,selected=self.attempt(recovery.CLIP,change=change)
            self.assertEqual(result.status,'invalid_archive',result.reason)
            self.assertFalse(calls);self.assertFalse(selected)

    def test_clipping_all_invalid_saved_support_returns_zero_without_pruning_inventory(self):
        saved=self.cases['full'][1]
        validity=saved.node(recovery.VALIDITY);clipped=saved.node(recovery.CLIP)
        # Replay consumes the archived direct input; it does not numerically
        # recompute or assert the upstream validity producer's ROI pixels.
        empty_support=torch.zeros_like(validity.outputs['validity'])
        empty_markers=torch.zeros_like(clipped.outputs['markers'])
        directory=self.changed(case='full',**{
            recovery.VALIDITY:replace(validity,outputs={**validity.outputs,'validity':empty_support}),
            recovery.CLIP:replace(clipped,outputs={'markers':empty_markers})})
        before=hashes(directory)
        result,calls,_=self.attempt(recovery.CLIP,'full',directory=directory)
        self.assertEqual(result.status,'matched',result.reason);self.assertEqual(calls,['recovery.clip_cores'])
        self.assertFalse(bool(result.outputs['markers'].any().item()))
        self.assertEqual(hashes(directory),before)
        self.assertEqual(len(saved.node(recovery.CORES).outputs['inventory']['objects']),2)

    def test_clipping_input_and_private_output_changes_fail_even_with_callback_exception(self):
        for target in range(3):
            watched=[];mutated=[]
            def observe(frame,event,arg):
                if event=='return' and frame.f_code is recovery.core._Guard.watch.__code__:
                    watched.extend(frame.f_locals['tensors'])
            def cancel():
                if len(watched)>target and not mutated:
                    mutated.append(True);watched[target].add_(1)
                    raise ValueError('caller exception after clipping mutation')
                return False
            result,_,_=self.attempt(recovery.CLIP,cancel=cancel,observer=observe)
            self.assertTrue(mutated);self.assertEqual(result.status,'failed',result.reason)
            self.assertFalse(result.outputs);self.assertIn('changed',result.reason)

    def test_caller_errors_and_changed_contract_constants_withhold_results(self):
        def caller_error():raise ValueError('ordinary caller error')
        result,_,_=self.attempt(recovery.VALIDITY,cancel=caller_error)
        self.assertEqual(result.status,'failed');self.assertIn('ordinary caller error',result.reason)
        changed=[];original=contracts.COMPARE_ELEMENTS
        def cancel():
            frame=sys._getframe()
            while frame is not None:
                if frame.f_code is replay.replay_native_stage.__code__:
                    if frame.f_locals.get('outputs') and not changed:
                        contracts.COMPARE_ELEMENTS=original+1;changed.append(True)
                    break
                frame=frame.f_back
            return False
        try:result,_,_=self.attempt(recovery.VALIDITY,cancel=cancel)
        finally:contracts.COMPARE_ELEMENTS=original
        self.assertTrue(changed);self.assertEqual(result.status,'failed',result.reason);self.assertFalse(result.outputs)

    def test_actual_async_qt_replay_retains_active_run_and_shows_compact_summaries(self):
        app=QApplication.instance() or QApplication([])
        recipe,saved,directory=self.cases['roi'];dialog=NativeRecipeDialog(recipe)
        dialog.result=saved;dialog.resize(1050,650);dialog.show();app.processEvents()
        before=hashes(directory)
        try:
            for stage,text in ((recovery.CORES,'Four-connected cores:'),(recovery.VALIDITY,'Valid source support'),
                    (recovery.CLIP,'Cores clipped to valid source support'),(recovery.PARTITION,'Recovered partition:'),
                    (recovery.ANNOTATION,'Boundary uncertainty:'),(recovery.MEASUREMENTS,'Recovery measurements:')):
                dialog._selected_stage=stage;dialog.start_stage_replay(directory)
                end=time.monotonic()+60
                while dialog._worker is not None and time.monotonic()<end:app.processEvents();time.sleep(.01)
                self.assertIsNone(dialog._worker);self.assertEqual(dialog.replay_result.status,'matched',dialog.replay_result.reason)
                self.assertIs(dialog.result,saved);self.assertIn(text,dialog.facts.toPlainText())
                if stage==recovery.MEASUREMENTS:self.assertIn('No image output',dialog.preview.text())
                else:self.assertFalse(dialog.preview.pixmap().isNull())
        finally:dialog.close();app.processEvents()
        self.assertEqual(hashes(directory),before)

    def test_remaining_stages_compare_nested_pixels_plan_and_measurement_values(self):
        saved=self.cases['full'][1]
        node=saved.node(recovery.PARTITION);transport=node.outputs['partition']
        labels=node.outputs['labels'].clone();labels[0,0]=int(transport['tensors']['label_ids'][0])
        changed={**node.outputs,'labels':labels,'partition':{**transport,'tensors':{**transport['tensors'],'labels':labels}}}
        result,_,_=self.attempt(recovery.PARTITION,'full',directory=self.changed('full',**{recovery.PARTITION:replace(node,outputs=changed)}))
        self.assertEqual(result.status,'mismatch',result.reason)
        self.assertEqual(result.metadata['comparison']['mismatched_elements'],2)
        metadata={**node.outputs['metadata'],'plan':{**node.outputs['metadata']['plan'],
            'peak_live_bytes':node.outputs['metadata']['plan']['peak_live_bytes']+1}}
        changed={**node.outputs,'metadata':metadata,'partition':{**transport,'metadata':metadata}}
        result,_,_=self.attempt(recovery.PARTITION,'full',directory=self.changed('full',**{recovery.PARTITION:replace(node,outputs=changed)}))
        self.assertEqual(result.status,'mismatch',result.reason)
        self.assertEqual(result.metadata['comparison']['mismatched_elements'],2)
        node=saved.node(recovery.ANNOTATION);band=node.outputs['uncertainty_band'].clone();band[0,0]=1
        changed={**node.outputs,'uncertainty_band':band,'annotation':{**node.outputs['annotation'],'uncertainty_band':band}}
        result,_,_=self.attempt(recovery.ANNOTATION,'full',directory=self.changed('full',**{recovery.ANNOTATION:replace(node,outputs=changed)}))
        self.assertEqual(result.status,'mismatch',result.reason)
        self.assertEqual(result.metadata['comparison']['mismatched_elements'],2)
        node=saved.node(recovery.MEASUREMENTS);observations={sample:{name:dict(row) for name,row in rows.items()}
            for sample,rows in node.outputs['observations'].items()}
        first=next(iter(observations.values()));self.assertTrue(first['assigned_area']['available'])
        first['assigned_area']['value']+=1
        result,calls,selected=self.attempt(recovery.MEASUREMENTS,'full',directory=self.changed('full',
            **{recovery.MEASUREMENTS:replace(node,outputs={**node.outputs,'observations':observations})}))
        self.assertEqual(result.status,'mismatch',result.reason);self.assertEqual(selected,[()])
        self.assertEqual(calls,['recovery.measurements']);self.assertEqual(result.metadata['comparison']['mismatched_elements'],1)

    def test_nested_aliases_and_headers_rejected_before_upload(self):
        saved=self.cases['full'][1]
        for stage,port in ((recovery.PARTITION,'labels'),(recovery.ANNOTATION,'uncertainty_band')):
            node=saved.node(stage)
            directory=self.changed('full',**{stage:replace(node,outputs={**node.outputs,port:node.outputs[port].clone()})})
            result,calls,selected=self.attempt(stage,'full',directory=directory)
            self.assertEqual(result.status,'invalid_archive',result.reason);self.assertFalse(calls);self.assertFalse(selected)
        def bad_header(report,specs):
            node=next(node for node in report['nodes'] if node['stage_id']==recovery.ANNOTATION)
            array=dict(node['outputs'])['uncertainty_band']['name'];specs[array]['dtype']='<i4'
        result,calls,selected=self.attempt(recovery.MEASUREMENTS,change=bad_header)
        self.assertEqual(result.status,'invalid_archive',result.reason);self.assertFalse(calls);self.assertFalse(selected)

    def test_partition_direct_input_encodings_and_id_vector_fail_before_dispatch(self):
        saved=self.cases['full'][1];node=saved.node(recovery.PARTITION);transport=node.outputs['partition']
        for field,bad in (('domain',255),('unresolved',255),('interface',255),('label_ids',0),
                ('elevation',float('nan')),('elevation',float('inf'))):
            raster=transport['tensors'][field].clone();raster.view(-1)[0]=bad
            changed={**node.outputs,'partition':{**transport,'tensors':{**transport['tensors'],field:raster}}}
            nodes={recovery.PARTITION:replace(node,outputs=changed)}
            if field=='elevation':
                producer=saved.node(recovery.ELEVATION)
                nodes[recovery.ELEVATION]=replace(producer,outputs={'negative_distance':raster})
            result,calls,_=self.attempt(recovery.ANNOTATION,'full',directory=self.changed('full',**nodes))
            self.assertEqual(result.status,'invalid_archive',result.reason);self.assertFalse(calls)
        node=saved.node(recovery.ANNOTATION);band=node.outputs['uncertainty_band'].clone();band[0,0]=255
        changed={**node.outputs,'uncertainty_band':band,'annotation':{**node.outputs['annotation'],'uncertainty_band':band}}
        result,calls,_=self.attempt(recovery.ANNOTATION,'full',directory=self.changed('full',
            **{recovery.ANNOTATION:replace(node,outputs=changed)}))
        self.assertEqual(result.status,'invalid_archive',result.reason);self.assertFalse(calls)

    def test_remaining_nested_mutation_and_callback_exception_withhold_outputs(self):
        for stage,target in ((recovery.PARTITION,'input'),(recovery.PARTITION,'output'),
                (recovery.ANNOTATION,'input'),(recovery.ANNOTATION,'output')):
            mutated=[]
            def cancel():
                frame=sys._getframe()
                while frame is not None:
                    if frame.f_code is replay.replay_native_stage.__code__:
                        values=frame.f_locals;products=values.get('inputs' if target=='input' else 'outputs')
                        if products and not mutated:
                            if stage==recovery.PARTITION and target=='input':products['markers'].add_(1)
                            elif stage==recovery.PARTITION and target=='output':products['partition']['tensors']['labels'].add_(1)
                            elif stage==recovery.ANNOTATION and target=='output':products['annotation']['uncertainty_band'].add_(1)
                            else:products['partition']['tensors']['labels'].add_(1)
                            mutated.append(True);raise ValueError('caller exception after nested mutation')
                        break
                    frame=frame.f_back
                return False
            result,_,_=self.attempt(stage,cancel=cancel)
            self.assertTrue(mutated);self.assertEqual(result.status,'failed',result.reason);self.assertFalse(result.outputs)

    def test_measurement_compact_metadata_is_immutable_and_caller_errors_withhold_outputs(self):
        for target in ('inputs','outputs'):
            attempted=[]
            def cancel():
                frame=sys._getframe()
                while frame is not None:
                    if frame.f_code is replay.replay_native_stage.__code__:
                        products=frame.f_locals.get(target)
                        if products and not attempted:
                            attempted.append(True)
                            metadata=products['annotation']['metadata'] if target=='inputs' else products['metadata']
                            metadata['binding']['source_sha256']='2'*64
                        break
                    frame=frame.f_back
                return False
            result,_,_=self.attempt(recovery.MEASUREMENTS,cancel=cancel)
            self.assertTrue(attempted);self.assertEqual(result.status,'failed',result.reason);self.assertFalse(result.outputs)
            self.assertIn('TypeError',result.reason)

    def test_each_nested_partition_raster_mutation_and_exception_withholds_publication(self):
        for name in contracts.PARTITION_TENSORS:
            with self.subTest(field=name):
                mutated=[]
                def cancel():
                    frame=sys._getframe()
                    while frame is not None:
                        if frame.f_code is replay.replay_native_stage.__code__:
                            products=frame.f_locals.get('outputs')
                            if products and not mutated:
                                products['partition']['tensors'][name].add_(1);mutated.append(True)
                                raise ValueError('caller exception after nested tensor mutation')
                            break
                        frame=frame.f_back
                    return False
                result,_,_=self.attempt(recovery.PARTITION,cancel=cancel)
                self.assertTrue(mutated);self.assertEqual(result.status,'failed',result.reason);self.assertFalse(result.outputs)
                self.assertIn('changed',result.reason)

    def test_generic_upstream_recovery_stages_complete_the_calculated_graph(self):
        for stage in (recovery.FOREGROUND,recovery.ERODE,recovery.DISTANCE,recovery.ELEVATION):
            result,calls,selected=self.attempt(stage)
            self.assertEqual(result.status,'matched',result.reason)
            self.assertEqual(calls,[self.cases['roi'][1].node(stage).operation_id]);self.assertEqual(len(selected[0]),2)

    def test_empty_unseeded_and_singleton_recovery_replays(self):
        for index,mask in enumerate((np.zeros((2,3),np.uint8),np.full((1,1),255,np.uint8),
                np.array([[0,255,0],[0,0,0]],np.uint8))):
            image=self.root/f'recovery-small-{index}.png';cv2.imwrite(str(image),np.repeat(mask[...,None],3,axis=2))
            recipe=parameter(selected_recipe(),recovery.ERODE,'ksize',2)
            saved=runner.run_native_recipe_path(recipe,image)
            directory=self.root/f'recovery-small-{index}';archives.export_native_recipe_result(directory,recipe,saved)
            for stage in (recovery.PARTITION,recovery.ANNOTATION,recovery.MEASUREMENTS):
                self.assertEqual(saved.node(stage).status,'executed',saved.node(stage).reason)
                result,calls,selected=self.attempt(stage,directory=directory)
                self.assertEqual(result.status,'matched',result.reason);self.assertEqual(calls,[saved.node(stage).operation_id])
                self.assertEqual(len(selected[0]),ARRAY_COUNTS[stage])

    def test_annotation_all_radii_policies_and_disappeared_core_inventory(self):
        for radius in range(9):
            recipe=parameter(self.cases['roi'][0],recovery.ANNOTATION,'uncertainty_radius',radius)
            policy='flag_uncertain' if radius%2 else 'withhold_uncertain'
            recipe=parameter(recipe,recovery.ANNOTATION,'measurement_policy',policy)
            saved=runner.run_native_recipe_path(recipe,self.image)
            directory=self.root/f'recovery-controls-{radius}';archives.export_native_recipe_result(directory,recipe,saved)
            for stage in (recovery.ANNOTATION,recovery.MEASUREMENTS):
                result,calls,_=self.attempt(stage,directory=directory)
                self.assertEqual(result.status,'matched',result.reason);self.assertEqual(calls,[saved.node(stage).operation_id])
            rows=saved.node(recovery.ANNOTATION).outputs['metadata']['objects']
            self.assertEqual(len(rows),2);self.assertEqual(rows[-1]['reasons'],('disappeared_by_validity',))


if __name__=='__main__':unittest.main()
