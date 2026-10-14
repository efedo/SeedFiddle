"""Actual paper saved-input replay, primitive schema/admission and Qt workflow."""
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
from seedvision.benchmarking import native_smpts_recipe as paper, native_paper_replay as contracts
from seedvision.benchmarking import native_recipe as runner, native_stage_replay as replay
from seedvision.benchmarking.checkpoints import CoordinateFrame
from seedvision.export import native_results as archives
from seedvision.ui.native_recipe_dialog import NativeRecipeDialog
from tests.test_native_smpts_recipe import parameter, selected_recipe
from tests.test_smpts_paper import contact_fixture, canonical_components

STAGES=(paper.MEAN,paper.MEDIAN,paper.GLOBAL,paper.INTERSECTION,paper.COMPONENTS,
    paper.CONTACTS,paper.EROSION,paper.RELOCALIZE,paper.CROPS)
COUNTS=(2,2,2,3,2,1,5,2,5)


class PaperReplayContractTests(unittest.TestCase):
    def test_schema_preserves_tuple_types_ids_bounds_and_strict_integer_fields(self):
        rows=({'label_id':1,'area_pixels':2,'bounds_xyxy':(0,0,2,1)},)
        value=contracts.canonical(rows)
        self.assertEqual(contracts.inventory(value,(2,3)),rows)
        for changed in (dict(rows[0],label_id=True),dict(rows[0],label_id=2),
                dict(rows[0],area_pixels=0),dict(rows[0],bounds_xyxy=(0,0,4,1)),
                dict(rows[0],extra=1)):
            with self.assertRaises(contracts.NativePaperReplayError):
                contracts.inventory(contracts.canonical((changed,)),(2,3))
        with self.assertRaises(contracts.NativePaperReplayError):contracts.inventory(list(value),(2,3))
        for invalid in ((True,),(2,),(1,1),(1,0)):
            with self.assertRaises(contracts.NativePaperReplayError):contracts.ids(invalid,1,'Selected')
        difference=contracts.compare(value,contracts.canonical((dict(rows[0],area_pixels=1),)))
        self.assertEqual(difference['mismatched_elements'],1)

    def test_metadata_decoder_has_byte_item_depth_and_primitive_bounds(self):
        binding={'kind':'tuple','items':[{'kind':'mapping','items':[['label_id',1],
            ['area_pixels',2],['bounds_xyxy',{'kind':'tuple','items':[0,0,2,1]}]]}]}
        self.assertEqual(contracts.decode(binding)[0]['area_pixels'],2)
        for bad in ({'kind':'object','items':[]}, {'kind':'mapping','items':[['a',1],['a',2]]},
                {'kind':'array','name':'x'},float('nan')):
            with self.assertRaises(contracts.NativePaperReplayError):contracts.decode(bad)
        with patch.object(contracts,'MAX_METADATA_BYTES',32):
            with self.assertRaises(contracts.NativePaperReplayError):contracts.canonical('\\n'*32)
        with patch.object(contracts,'MAX_METADATA_DEPTH',1):
            with self.assertRaises(contracts.NativePaperReplayError):contracts.decode(binding)
        with patch.object(contracts,'MAX_METADATA_ITEMS',1):
            with self.assertRaises(contracts.NativePaperReplayError):contracts.decode(binding)

    def test_crop_offsets_parents_shape_and_overlapping_boxes(self):
        # Two disjoint components can have overlapping boxes; T may exceed P.
        rows=contracts.canonical(tuple({'label_id':i,'area_pixels':2,'bounds_xyxy':(0,0,3,3)} for i in (1,2)))
        context={'source_components':rows,'final_components':rows}
        crops=contracts.canonical(tuple(dict(row,parent_id=1,offset_pixels=(i-1)*9,crop_shape=(3,3))
            for i,row in enumerate(rows,1)))
        contracts.validate(crops,'crops',(3,3),context)
        self.assertEqual(contracts.crop_total(rows),18)
        self.assertEqual(contracts.raster_spec('packed_rgb',(3,3),context),((54,),'|u1'))
        for key,bad in (('offset_pixels',1),('parent_id',3),('crop_shape',(3,2)),
                ('area_pixels',True),('bounds_xyxy',(False,0,3,3)),('bounds_xyxy',(0.,0,3,3))):
            changed=contracts.canonical((dict(crops[0],**{key:bad}),crops[1]))
            with self.assertRaises(contracts.NativePaperReplayError):
                contracts.validate(changed,'crops',(3,3),context)
        value=contracts.assumptions(2);contracts.validate(value,'metadata',(3,3),context)
        for key,bad in (('original_execution',True),('crop_count',True),('median_border','reflect')):
            with self.assertRaises(contracts.NativePaperReplayError):
                contracts.validate(dict(value,**{key:bad}),'metadata',(3,3),context)

    def test_allocation_free_operation_limits_alias_facts_and_exact_cap(self):
        recipe=selected_recipe()
        for stage in STAGES:
            operation=recipe.stage(stage).operation_id
            opts={'selected_allocations':(12,48),'input_allocations':(12,)}
            result=contracts.plan(operation,(3,4),recipe.parameters_for_runner(stage),**opts)
            self.assertGreaterEqual(result['cuda_bytes'],60+contracts.AUDIT_CUDA_BYTES)
            contracts.plan(operation,(3,4),recipe.parameters_for_runner(stage),
                cuda_limit=result['cuda_bytes'],selected_limit=60,**opts)
            for name,limit in (('cuda_limit',result['cuda_bytes']-1),('selected_limit',59)):
                with self.assertRaises(contracts.NativePaperReplayUnavailable):
                    contracts.plan(operation,(3,4),recipe.parameters_for_runner(stage),**{name:limit},**opts)
        with self.assertRaises(contracts.NativePaperReplayUnavailable):
            contracts.plan('smpts_paper.median',(1100,1000),{'ksize':31},**opts)
        with self.assertRaises(contracts.NativePaperReplayUnavailable):
            contracts.plan('smpts_paper.components',(1,7_272_728),{},**opts)
        with self.assertRaises(contracts.NativePaperReplayError):
            contracts.plan('smpts_paper.erosion',(3,4),{'ksize':2},**opts)
        mean=contracts.plan('smpts_paper.mean',(2080,3120),{'ksize':25,'offset':2.,'object_type':'light'},
            selected_allocations=(2080*3120,2080*3120),input_allocations=(2080*3120,))
        self.assertLessEqual(mean['cuda_bytes'],contracts.MAX_CUDA_BYTES)
        result=contracts.plan('smpts_paper.crops',(3,3),{},crop_pixels=18,**opts)
        self.assertEqual(result['crop_pixels'],18)


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class PaperStageReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary=tempfile.TemporaryDirectory(dir='artifacts');cls.root=Path(cls.temporary.name)
        mask,rgb=contact_fixture();gray=np.where(mask,220,10).astype(np.uint8)
        cls.cases={};cls.images={}
        for name,gray,rgb in (('full',gray,rgb),('singleton',np.full((1,1),220,np.uint8),np.full((1,1,3),220,np.uint8)),
                ('empty',np.full((2,3),10,np.uint8),np.full((2,3,3),10,np.uint8))):
            image=cls.root/(name+'.png');assert cv2.imwrite(str(image),rgb[:,:,::-1])
            recipe=selected_recipe();frame=CoordinateFrame('source.decoded',gray.shape,'top-left','x-right,y-down','pixel')
            saved=runner.run_native_recipe(recipe,torch.tensor(gray,device='cuda'),torch.tensor(rgb,device='cuda'),
                source_sha256=sha256(image.read_bytes()).hexdigest(),coordinate_frame=frame,
                preparation={'gray':'explicit test gray','rgb':'original test RGB'})
            assert saved.status=='executed',[(n.stage_id,n.status,n.reason) for n in saved.nodes]
            directory=cls.root/name;archives.export_native_recipe_result(directory,recipe,saved)
            cls.cases[name]=(recipe,saved,directory);cls.images[name]=image
        cls.full_mask,cls.full_rgb=contact_fixture()
        cls.full_gray=np.where(cls.full_mask,220,10).astype(np.uint8)

    @classmethod
    def tearDownClass(cls):cls.temporary.cleanup()

    def attempt(self,stage,case='full',directory=None,cancel=None,change=None):
        calls=[];selected=[]
        def profile(frame,event,arg):
            if event=='call' and frame.f_code is runner._dispatch.__code__:calls.append(frame.f_locals['operation'])
            elif event=='call' and frame.f_code is archives.load_native_recipe_result.__code__:selected.append(frame.f_locals['array_names'])
            elif event=='return' and change is not None and frame.f_code is archives.read_native_recipe_inventory.__code__ and arg is not None:change(*arg)
        previous=sys.getprofile();sys.setprofile(profile)
        try:result=replay.replay_native_stage(directory or self.cases[case][2],stage,cancel=cancel)
        finally:sys.setprofile(previous)
        return result,calls,selected

    def changed(self,**nodes):
        recipe,saved,_=self.cases['full']
        changed=replace(saved,nodes=tuple(nodes.get(node.stage_id,node) for node in saved.nodes))
        directory=self.root/('changed-'+str(len(tuple(self.root.iterdir()))))
        archives.export_native_recipe_result(directory,recipe,changed)
        return directory

    def test_all_nine_stage_instances_replay_only_selected_operation_and_preserve_archive(self):
        for case,(_,saved,directory) in self.cases.items():
            before={str(p.relative_to(directory)):sha256(p.read_bytes()).hexdigest() for p in directory.rglob('*') if p.is_file()}
            for stage,count in zip(STAGES,COUNTS):
                with self.subTest(case=case,stage=stage):
                    result,calls,selected=self.attempt(stage,case)
                    self.assertEqual(result.status,'matched',result.reason)
                    self.assertEqual(calls,[saved.node(stage).operation_id])
                    self.assertEqual(len(selected),1);self.assertEqual(len(selected[0]),count)
                    self.assertEqual(set(result.outputs),set(saved.node(stage).outputs))
                    self.assertFalse(result.scientifically_validated)
                    self.assertEqual(result.metadata['frame_scope'],'source')
                    if stage==paper.CROPS:self.assertFalse(result.outputs['metadata']['original_execution'])
            after={str(p.relative_to(directory)):sha256(p.read_bytes()).hexdigest() for p in directory.rglob('*') if p.is_file()}
            self.assertEqual(before,after)

    def test_full_graph_matches_independent_opencv_choices_and_packed_original_rgb(self):
        gray=self.full_gray;recipe,saved,_=self.cases['full']
        adaptive=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,25,2)
        median=cv2.medianBlur(adaptive,5)
        fixed=np.where(gray>100,255,0).astype(np.uint8);combined=cv2.bitwise_and(median,fixed)
        labels,rows=canonical_components(combined)
        selected=tuple(row['label_id'] for row in rows if np.prod(np.subtract(row['bounds_xyxy'][2:],row['bounds_xyxy'][:2]))>20)
        target=np.where(np.isin(labels,selected),255,0).astype(np.uint8)
        eroded=cv2.erode(target,np.ones((13,13),np.uint8),borderType=cv2.BORDER_REPLICATE)
        merged=np.where(np.isin(labels,selected),eroded,combined).astype(np.uint8)
        final,finalrows=canonical_components(merged)
        for stage,port,expected in ((paper.MEAN,'mask',adaptive),(paper.MEDIAN,'mask',median),
                (paper.GLOBAL,'mask',fixed),(paper.INTERSECTION,'mask',combined),
                (paper.COMPONENTS,'labels',labels),(paper.EROSION,'mask',merged),
                (paper.EROSION,'selected_mask',target),(paper.EROSION,'eroded_selected_mask',eroded),
                (paper.RELOCALIZE,'labels',final)):
            np.testing.assert_array_equal(saved.node(stage).outputs[port].cpu().numpy(),expected)
        self.assertEqual(saved.node(paper.COMPONENTS).outputs['components'],rows)
        self.assertEqual(saved.node(paper.CONTACTS).outputs['selected_parent_ids'],selected)
        packed=saved.node(paper.CROPS).outputs;offset=0;parents=set()
        for row,component in zip(packed['crops'],finalrows):
            x0,y0,x1,y1=row['bounds_xyxy'];mask=final[y0:y1,x0:x1]==row['label_id']
            parent=np.unique(labels[final==row['label_id']])
            self.assertEqual(len(parent),1);self.assertEqual(row['parent_id'],int(parent[0]))
            self.assertEqual(row['offset_pixels'],offset);parents.add(row['parent_id'])
            wanted=np.where(mask[...,None],self.full_rgb[y0:y1,x0:x1],0).reshape(-1)
            np.testing.assert_array_equal(packed['packed_rgb'][3*offset:3*(offset+mask.size)].cpu().numpy(),wanted)
            np.testing.assert_array_equal(packed['packed_mask'][offset:offset+mask.size].cpu().numpy(),mask.reshape(-1).astype(np.uint8)*255)
            offset+=mask.size
        self.assertEqual(packed['disappeared_parent_ids'],tuple(row['label_id'] for row in rows if row['label_id'] not in parents))

    def test_original_path_is_not_opened(self):
        image=self.images['full'];raw=image.read_bytes();image.unlink()
        try:
            for stage in STAGES:self.assertEqual(self.attempt(stage)[0].status,'matched')
        finally:image.write_bytes(raw)

    def test_well_typed_changed_pixels_compact_leaves_and_selection_report_mismatch(self):
        saved=self.cases['full'][1]
        for stage,port in ((paper.GLOBAL,'mask'),(paper.EROSION,'mask'),(paper.CROPS,'packed_rgb')):
            node=saved.node(stage);value=node.outputs[port].clone()
            value.view(-1)[0]=255 if int(value.view(-1)[0])==0 else 0
            result,calls,_=self.attempt(stage,directory=self.changed(**{stage:replace(node,outputs={**node.outputs,port:value})}))
            self.assertEqual(result.status,'mismatch',result.reason);self.assertEqual(len(calls),1)
            self.assertEqual(result.metadata['comparison']['mismatched_elements'],1)
        node=saved.node(paper.COMPONENTS);rows=list(node.outputs['components'])
        rows[0]=dict(rows[0],area_pixels=rows[0]['area_pixels']-1)
        result,_,_=self.attempt(paper.COMPONENTS,directory=self.changed(**{paper.COMPONENTS:replace(node,outputs={**node.outputs,'components':tuple(rows)})}))
        self.assertEqual(result.status,'mismatch',result.reason)
        self.assertEqual(result.metadata['comparison']['mismatched_elements'],1)
        node=saved.node(paper.CONTACTS)
        result,_,_=self.attempt(paper.CONTACTS,directory=self.changed(**{paper.CONTACTS:replace(node,outputs={'selected_parent_ids':()})}))
        self.assertEqual(result.status,'mismatch',result.reason)

    def test_bad_headers_aliases_crop_offsets_controls_and_producer_keys_fail_before_upload(self):
        def node(report,stage):return next(n for n in report['nodes'] if n['stage_id']==stage)
        def key(report,specs):node(report,paper.RELOCALIZE)['input_keys']['mask']='0'*64
        def header(report,specs):specs[dict(node(report,paper.COMPONENTS)['outputs'])['labels']['name']]['dtype']='|u1'
        def control(report,specs):next(item for item in node(report,paper.MEDIAN)['parameters']['items'] if item[0]=='ksize')[1]=3
        def alias(report,specs):
            values=node(report,paper.CROPS)['outputs']
            next(item for item in values if item[0]=='labels')[1]=dict(node(report,paper.COMPONENTS)['outputs'])['labels']
        for stage,change in ((paper.RELOCALIZE,key),(paper.CROPS,header),(paper.CROPS,control),(paper.CROPS,alias)):
            result,calls,selected=self.attempt(stage,change=change)
            self.assertEqual(result.status,'invalid_archive',result.reason);self.assertFalse(calls);self.assertFalse(selected)
        saved=self.cases['full'][1];node=saved.node(paper.CROPS)
        rows=list(node.outputs['crops']);rows[0]=dict(rows[0],offset_pixels=1)
        result,calls,selected=self.attempt(paper.CROPS,directory=self.changed(**{paper.CROPS:replace(node,outputs={**node.outputs,'crops':tuple(rows)})}))
        self.assertEqual(result.status,'invalid_archive',result.reason);self.assertFalse(calls);self.assertFalse(selected)

    def test_admission_cancellation_callback_exception_and_tensor_mutation_withhold_publication(self):
        for stage in STAGES:
            with patch.object(replay,'MAX_REPLAY_CUDA_BYTES',1),patch.object(replay,'_runtime') as runtime:
                result,calls,selected=self.attempt(stage)
                self.assertEqual(result.status,'unavailable',result.reason);runtime.assert_not_called()
                self.assertFalse(calls);self.assertFalse(selected)
            result,_,_=self.attempt(stage,cancel=lambda:True)
            self.assertEqual(result.status,'cancelled');self.assertFalse(result.outputs)
        self.assertEqual(self.attempt(paper.CROPS,cancel=lambda:1)[0].status,'failed')
        for port in ('labels','packed_rgb','packed_mask'):
            mutated=[]
            def cancel():
                frame=sys._getframe()
                while frame:
                    if frame.f_code is replay.replay_native_stage.__code__:
                        products=frame.f_locals.get('outputs')
                        if products and not mutated:
                            products[port].add_(1);mutated.append(True)
                            raise ValueError('caller exception after mutation')
                        break
                    frame=frame.f_back
                return False
            result,_,_=self.attempt(paper.CROPS,cancel=cancel)
            self.assertTrue(mutated);self.assertEqual(result.status,'failed',result.reason);self.assertFalse(result.outputs)
            self.assertIn('changed',result.reason)


    def test_explicit_frame_identity_and_overlapping_crop_boxes_above_source_area(self):
        mask=np.zeros((35,37),np.uint8);mask[:5,:]=mask[-5:,:]=255;mask[:,:5]=mask[:,-5:]=255
        mask[10:25,10:27]=255
        yy,xx=np.indices(mask.shape);rgb=np.stack((xx,yy,xx+yy),axis=2).astype(np.uint8)
        gray=np.where(mask,220,10).astype(np.uint8)
        recipe=parameter(selected_recipe(),paper.MEAN,'offset',255.)
        recipe=parameter(recipe,paper.MEDIAN,'ksize',1)
        recipe=parameter(recipe,paper.CONTACTS,'contact_box_area',100_000)
        frame=CoordinateFrame('explicit.test-frame',mask.shape,'top-left','x-right,y-down','pixel','c'*64)
        saved=runner.run_native_recipe(recipe,torch.tensor(gray,device='cuda'),torch.tensor(rgb,device='cuda'),
            source_sha256='d'*64,coordinate_frame=frame,preparation={'gray':'declared gray','rgb':'declared RGB'})
        self.assertEqual(saved.status,'executed')
        packed=saved.node(paper.CROPS).outputs
        self.assertGreater(packed['packed_mask'].numel(),mask.size)
        self.assertEqual(tuple(row['parent_id'] for row in packed['crops']),(1,2))
        directory=self.root/'overlapping';archives.export_native_recipe_result(directory,recipe,saved)
        for stage in (paper.MEAN,paper.CONTACTS,paper.CROPS):
            result,calls,_=self.attempt(stage,directory=directory)
            self.assertEqual(result.status,'matched',result.reason)
            self.assertEqual(CoordinateFrame.from_dict(dict(result.metadata['coordinate_frame'])),frame)
            self.assertEqual(calls,[saved.node(stage).operation_id])
        for row in packed['crops']:
            x0,y0,x1,y1=row['bounds_xyxy'];offset=row['offset_pixels'];count=np.prod(row['crop_shape'])
            expected=np.where((mask[y0:y1,x0:x1]>0)[...,None],rgb[y0:y1,x0:x1],0)
            # Other component pixels inside the box must be zeroed.
            labels=saved.labels[y0:y1,x0:x1].cpu().numpy()
            expected=np.where((labels==row['label_id'])[...,None],rgb[y0:y1,x0:x1],0)
            np.testing.assert_array_equal(packed['packed_rgb'][3*offset:3*(offset+count)].cpu().numpy(),expected.reshape(-1))

    def test_private_cuda_buffers_and_contract_state_mutation_with_callback_exceptions_withhold_outputs(self):
        for stage in (paper.COMPONENTS,paper.EROSION,paper.CROPS):
            watched=[];mutated=[]
            def profile(frame,event,arg):
                if event=='return' and frame.f_code is paper.paper._Guard.watch.__code__:
                    watched.extend(frame.f_locals['tensors'])
            def cancel():
                frame=sys._getframe()
                while frame:
                    if frame.f_code is replay.replay_native_stage.__code__:
                        owned=frame.f_locals.get('owned_storage',set())
                        candidate=next((v for v in watched if v.untyped_storage().data_ptr() not in owned),None)
                        if candidate is not None and not mutated:
                            candidate.add_(1);mutated.append(True)
                            raise ValueError('caller exception after private buffer mutation')
                        break
                    frame=frame.f_back
                return False
            old=sys.getprofile();sys.setprofile(profile)
            try:result=replay.replay_native_stage(self.cases['full'][2],stage,cancel=cancel)
            finally:sys.setprofile(old)
            self.assertTrue(mutated);self.assertEqual(result.status,'failed',result.reason);self.assertFalse(result.outputs)
            self.assertIn('changed',result.reason)
        changed=[];original=contracts.COMPARE_ELEMENTS
        def cancel():
            frame=sys._getframe()
            while frame:
                if frame.f_code is replay.replay_native_stage.__code__:
                    if frame.f_locals.get('outputs') and not changed:
                        contracts.COMPARE_ELEMENTS=17;changed.append(True)
                    break
                frame=frame.f_back
            return False
        try:result=self.attempt(paper.CROPS,cancel=cancel)[0]
        finally:contracts.COMPARE_ELEMENTS=original
        self.assertTrue(changed);self.assertEqual(result.status,'failed',result.reason);self.assertFalse(result.outputs)

    def test_actual_async_desktop_summaries_use_label_preview_and_preserve_active_run(self):
        app=QApplication.instance() or QApplication([])
        recipe,_,directory=self.cases['full']
        saved=self.cases['singleton'][1]
        dialog=NativeRecipeDialog(recipe)
        dialog.set_source_path(self.images['singleton']);dialog._completed(saved)
        dialog.resize(1050,650);dialog.show();app.processEvents()
        try:
            for stage,expected in ((paper.CONTACTS,'Selected contact parents'),(paper.CROPS,'Packed original RGB crops')):
                dialog._select_stage(stage);dialog.start_stage_replay(directory)
                deadline=time.monotonic()+90
                while dialog._worker is not None and time.monotonic()<deadline:
                    app.processEvents();time.sleep(.01)
                self.assertIsNone(dialog._worker);self.assertEqual(dialog.replay_result.status,'matched',dialog.replay_result.reason)
                self.assertIn(expected,dialog.facts.toPlainText());self.assertIs(dialog.result,saved)
            self.assertFalse(dialog.preview.pixmap().isNull())
            self.assertIn('Provisional paper reconstruction',dialog.facts.toPlainText())
            self.assertEqual(dialog.crop_selector.count(),len(dialog.replay_result.outputs['crops'])+1)
            dialog.crop_selector.setCurrentIndex(1);app.processEvents()
            self.assertIn('Saved-input replay',dialog.facts.toPlainText())
            pixel=dialog.preview.pixmap().toImage().pixelColor(0,0)
            expected=tuple(dialog.replay_result.outputs['packed_rgb'][:3].cpu().tolist())
            self.assertEqual((pixel.red(),pixel.green(),pixel.blue()),expected)
            self.assertNotEqual(expected,(220,220,220));self.assertIs(dialog.result,saved)
            dialog._select_stage(paper.CROPS)
            self.assertEqual(dialog.crop_selector.count(),len(saved.node(paper.CROPS).outputs['crops'])+1)
            dialog.crop_selector.setCurrentIndex(1);app.processEvents()
            pixel=dialog.preview.pixmap().toImage().pixelColor(0,0)
            self.assertEqual((pixel.red(),pixel.green(),pixel.blue()),(220,220,220))
            for failure in (ValueError('deliberate failed replay'),runner.NativeRecipeCancelled('deliberate cancellation')):
                dialog._replay_completed(failure);app.processEvents()
                self.assertFalse(dialog.crop_selector.isVisible());self.assertFalse(dialog._paper_crop_replay)
                self.assertIs(dialog.result,saved)
                dialog._select_stage(paper.CROPS);dialog.crop_selector.setCurrentIndex(1);app.processEvents()
                pixel=dialog.preview.pixmap().toImage().pixelColor(0,0)
                self.assertEqual((pixel.red(),pixel.green(),pixel.blue()),(220,220,220))
        finally:dialog.close();app.processEvents()


if __name__=='__main__':unittest.main()
