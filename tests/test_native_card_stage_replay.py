"""Saved direct card inputs, mixed frames, every product, guards and actual Qt."""
from dataclasses import replace
from hashlib import sha256
import os
from pathlib import Path
import shutil
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
from seedvision.benchmarking import native_card_recipe as card, native_card_replay as contracts
from seedvision.benchmarking import native_recipe as runner, native_stage_replay as replay
from seedvision.benchmarking import native_calibrated_seed_recipe as calibrated
from seedvision.benchmarking import native_watershed_diagnostic_recipe as diagnostic
from seedvision.export import native_results as archives
from seedvision.ui.native_recipe_dialog import NativeRecipeDialog
from tests.test_native_card_recipe import fixture, _parameter

BGR=card.PREFIX+'bgr'
ROI=(40,40,420,580)


def hashes(directory):
    return {str(path.relative_to(directory)):sha256(path.read_bytes()).hexdigest()
            for path in directory.rglob('*') if path.is_file()}


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class CardStageReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary=tempfile.TemporaryDirectory(dir='artifacts');cls.root=Path(cls.temporary.name)
        cls.image=cls.root/'card.png';assert cv2.imwrite(str(cls.image),fixture(varied=True))
        cls.cases={}
        for name,bounds in (('full',None),('roi',ROI)):
            recipe=_parameter(card.build_native_card_recipe(),card.PREPROCESS,'roi_bounds_xyxy',bounds)
            recipe=_parameter(recipe,card.SCALE,'color_chip_size',[7,11])
            saved=runner.run_native_recipe_path(recipe,cls.image)
            assert saved.status=='executed',[(node.stage_id,node.reason) for node in saved.nodes]
            directory=cls.root/name;archives.export_native_recipe_result(directory,recipe,saved)
            cls.cases[name]=(recipe,saved,directory)

    @classmethod
    def tearDownClass(cls):cls.temporary.cleanup()

    def attempt(self,stage,case='roi',directory=None,cancel=None,inventory_change=None):
        directory=directory or self.cases[case][2];calls=[];selected=[]
        def profile(frame,event,arg):
            if event=='call' and frame.f_code is runner._dispatch.__code__:calls.append(frame.f_locals['operation'])
            elif event=='call' and frame.f_code is archives.load_native_recipe_result.__code__:
                selected.append(frame.f_locals['array_names'])
            elif (event=='return' and inventory_change is not None and
                    frame.f_code is archives.read_native_recipe_inventory.__code__ and arg is not None):
                inventory_change(*arg)
        previous=sys.getprofile();sys.setprofile(profile)
        try:result=replay.replay_native_stage(directory,stage,cancel=cancel)
        finally:sys.setprofile(previous)
        return result,calls,selected

    def changed(self,case='roi',**nodes):
        recipe,saved,_=self.cases[case]
        changed=replace(saved,nodes=tuple(nodes.get(node.stage_id,node) for node in saved.nodes))
        directory=self.root/('changed-'+str(len(tuple(self.root.iterdir()))))
        archives.export_native_recipe_result(directory,recipe,changed)
        return directory

    def test_every_full_and_roi_card_stage_matches_all_outputs_and_direct_selection(self):
        counts={BGR:2,card.PREPROCESS:5,card.GEOMETRY:2,card.MASK:3,card.SCALE:0,
                card.MEANS:2,card.STANDARD:1,card.AFFINE:2}
        for case,(recipe,saved,directory) in self.cases.items():
            before=hashes(directory)
            for stage in recipe.stages[1:]:
                with self.subTest(case=case,stage=stage.stage_id):
                    result,calls,selected=self.attempt(stage.stage_id,case)
                    self.assertEqual(result.status,'matched',result.reason);self.assertEqual(calls,[stage.operation_id])
                    self.assertEqual(len(selected),1);self.assertEqual(len(selected[0]),counts[stage.stage_id])
                    self.assertEqual(set(result.outputs),set(saved.node(stage.stage_id).outputs))
                    self.assertEqual(set(result.metadata['comparison']['by_port']),set(result.outputs))
                    for name,value in result.outputs.items():
                        if torch.is_tensor(value):self.assertTrue(torch.equal(value,saved.node(stage.stage_id).outputs[name]))
                    self.assertFalse(result.scientifically_validated)
            self.assertEqual(hashes(directory),before)

    def test_scale_selects_no_arrays_but_binds_source_roi_and_geometry(self):
        result,calls,selected=self.attempt(card.SCALE)
        self.assertEqual(result.status,'matched',result.reason);self.assertEqual(selected,[()])
        self.assertEqual(result.metadata['plan']['selected_bytes'],0)
        self.assertEqual(result.metadata['plan']['cuda_bytes'],contracts.AUDIT_CUDA_BYTES)
        self.assertEqual(tuple(result.metadata['card_roi_bounds_xyxy']),ROI)
        self.assertIn(card.GEOMETRY,{item['producer_stage'] for item in result.metadata['frame_lineage']['card_ancestry']})
        self.assertEqual(calls,['color_card.scale'])

    def test_same_sized_differently_offset_and_explicit_full_rois_keep_distinct_frame_identities(self):
        recipe,_,_=self.cases['roi'];identities=[]
        for bounds in (ROI,(30,30,410,570),(0,0,460,620)):
            selected=_parameter(recipe,card.PREPROCESS,'roi_bounds_xyxy',bounds)
            saved=runner.run_native_recipe_path(selected,self.image)
            self.assertEqual(saved.status,'executed')
            directory=self.root/('offset-'+str(bounds[0]));archives.export_native_recipe_result(directory,selected,saved)
            result,_,_=self.attempt(card.PREPROCESS,directory=directory)
            self.assertEqual(result.status,'matched',result.reason)
            identities.append(result.metadata['card_roi_coordinate_frame']['transform_sha256'])
            self.assertEqual(result.metadata['frame_scope'],'card_roi')
            self.assertNotEqual(result.metadata['card_roi_coordinate_frame']['frame_id'],result.metadata['source_coordinate_frame']['frame_id'])
        self.assertEqual(len(set(identities)),3)

    def test_all_nine_actual_card_graph_variants_replay_canonical_shared_prefix(self):
        variants=[card.build_native_card_recipe()]
        for kind in ('binary','mean','otsu','range'):
            recipe=calibrated.build_native_calibrated_seed_recipe(kind)
            variants.extend((recipe,diagnostic.build_native_watershed_diagnostic_recipe(recipe)))
        self.assertEqual(len(variants),9)
        for index,recipe in enumerate(variants):
            with self.subTest(recipe=recipe.recipe_id):
                saved=runner.run_native_recipe_path(recipe,self.image)
                self.assertTrue(all(saved.node(stage.stage_id).status=='executed' for stage in card.build_native_card_recipe().stages[1:]))
                directory=self.root/f'variant-{index}';archives.export_native_recipe_result(directory,recipe,saved)
                for stage in (BGR,card.PREPROCESS,card.SCALE,card.AFFINE):
                    result,calls,_=self.attempt(stage,directory=directory)
                    self.assertEqual(result.status,'matched',result.reason)
                    self.assertEqual(calls,[recipe.stage(stage).operation_id])
        for recipe_id in diagnostic.BASE_IDS[:3]:
            recipe=diagnostic.canonical_base(recipe_id+diagnostic.SUFFIX)
            self.assertFalse(any(stage.operation_id in contracts.PORTS for stage in recipe.stages))

    def test_deleted_origin_image_is_not_resolved_or_opened(self):
        raw=self.image.read_bytes();self.image.unlink()
        try:
            for stage in (BGR,card.PREPROCESS,card.GEOMETRY,card.MASK,card.SCALE,card.MEANS,card.STANDARD,card.AFFINE):
                result,_,_=self.attempt(stage);self.assertEqual(result.status,'matched',result.reason)
        finally:self.image.write_bytes(raw)

    def test_valid_changed_compact_and_raster_outputs_are_mismatches(self):
        saved=self.cases['roi'][1]
        changes=((card.GEOMETRY,'geometry','median_color_chip_width'),(card.SCALE,'metadata','px_width'),
                 (card.MEANS,'source_matrix',None),(card.STANDARD,'target_matrix',None),
                 (card.AFFINE,'transformation_matrix',None))
        for stage,port,key in changes:
            node=saved.node(stage);value=node.outputs[port]
            if key is not None:
                changed=dict(value);changed[key]+=0.01
            else:
                changed=[list(row) for row in value];changed[0][-1]+=0.001
                changed=tuple(tuple(row) for row in changed)
            directory=self.changed(**{stage:replace(node,outputs={**node.outputs,port:changed})})
            result,_,_=self.attempt(stage,directory=directory)
            self.assertEqual(result.status,'mismatch',result.reason)
            self.assertEqual(result.metadata['comparison']['mismatched_elements'],1)
        for stage,port in ((BGR,'bgr'),(card.PREPROCESS,'gray'),(card.MASK,'labeled_mask'),(card.AFFINE,'corrected_rgb')):
            node=saved.node(stage);value=node.outputs[port].clone()
            if port=='labeled_mask':value[0,0]=10.0
            else:value.view(-1)[0]^=1
            result,_,_=self.attempt(stage,directory=self.changed(**{
                stage:replace(node,outputs={**node.outputs,port:value})}))
            self.assertEqual(result.status,'mismatch',result.reason)
            self.assertEqual(result.metadata['comparison']['mismatched_elements'],1)

    def test_producer_keys_controls_bounds_and_raster_headers_reject_before_load(self):
        def bad_key(report,specs):
            next(node for node in report['nodes'] if node['stage_id']==card.GEOMETRY)['input_keys']['threshold']='0'*64
        def bad_bounds(report,specs):
            binding=dict(next(node for node in report['nodes'] if node['stage_id']==card.PREPROCESS)['outputs'])['bounds']
            binding['items'][:]=[30,30,410,570]
        def bad_control(report,specs):
            node=next(node for node in report['nodes'] if node['stage_id']==card.PREPROCESS)
            next(item for item in node['parameters']['items'] if item[0]=='block_size')[1]=53
        def bad_header(report,specs):
            node=next(node for node in report['nodes'] if node['stage_id']==card.PREPROCESS)
            name=dict(node['outputs'])['roi_bgr']['name'];specs[name]['shape']=[620,460,3]
        for mutate in (bad_key,bad_bounds,bad_control,bad_header):
            result,calls,selected=self.attempt(card.SCALE,inventory_change=mutate)
            self.assertEqual(result.status,'invalid_archive',result.reason);self.assertFalse(calls);self.assertFalse(selected)

    def test_geometry_and_matrix_structural_errors_reject_before_load(self):
        saved=self.cases['roi'][1]
        for stage,port,value in ((card.GEOMETRY,'geometry',{**saved.node(card.GEOMETRY).outputs['geometry'],'detected_chips':True}),
            (card.MEANS,'source_matrix',((10.0,-1.0,0.0,0.0),)),
            (card.AFFINE,'transformation_matrix',((0.0,),))):
            node=saved.node(stage);directory=self.changed(**{stage:replace(node,outputs={**node.outputs,port:value})})
            result,calls,selected=self.attempt(stage,directory=directory)
            self.assertEqual(result.status,'invalid_archive',result.reason);self.assertFalse(calls);self.assertFalse(selected)

    def test_float64_label_validation_precedes_narrowing_and_dispatch(self):
        saved=self.cases['roi'][1];node=saved.node(card.MASK)
        for bad in (float('nan'),float('inf'),-10.0,.5,250.0,266.0):
            labels=node.outputs['labeled_mask'].clone();labels[0,0]=bad
            directory=self.changed(**{card.MASK:replace(node,outputs={**node.outputs,'labeled_mask':labels})})
            result,calls,_=self.attempt(card.MEANS,directory=directory)
            self.assertEqual(result.status,'invalid_archive',result.reason);self.assertFalse(calls)

    def test_empty_means_missing_background_and_no_card_are_distinct(self):
        saved=self.cases['roi'][1];mask=saved.node(card.MASK);means=saved.node(card.MEANS)
        zero=torch.zeros_like(mask.outputs['labeled_mask'])
        directory=self.changed(**{card.MASK:replace(mask,outputs={**mask.outputs,'labeled_mask':zero}),
            card.MEANS:replace(means,outputs={'source_matrix':()})})
        result,calls,_=self.attempt(card.MEANS,directory=directory)
        self.assertEqual(result.status,'matched',result.reason);self.assertEqual(result.outputs['source_matrix'],())
        full=torch.full_like(zero,10)
        result,_,_=self.attempt(card.MEANS,directory=self.changed(**{
            card.MASK:replace(mask,outputs={**mask.outputs,'labeled_mask':full})}))
        self.assertEqual(result.status,'unavailable',result.reason);self.assertFalse(result.outputs)
        prep=saved.node(card.PREPROCESS);threshold=torch.zeros_like(prep.outputs['threshold'])
        result,_,_=self.attempt(card.GEOMETRY,directory=self.changed(**{
            card.PREPROCESS:replace(prep,outputs={**prep.outputs,'threshold':threshold})}))
        self.assertEqual(result.status,'unavailable',result.reason);self.assertIn('No color card',result.reason)

    def test_23_detected_contours_and_radius_zero_clip_overlap_mask_paths(self):
        image=fixture(varied=True);image[235:286,155:206]=220
        path=self.root/'missing.png';cv2.imwrite(str(path),image)
        recipe=_parameter(self.cases['roi'][0],card.PREPROCESS,'adaptive_method',0)
        for radius in (0,20,80):
            selected=_parameter(recipe,card.MASK,'radius',radius)
            saved=runner.run_native_recipe_path(selected,path)
            self.assertEqual(saved.node(card.GEOMETRY).outputs['geometry']['detected_chips'],23)
            directory=self.root/f'missing-{radius}';archives.export_native_recipe_result(directory,selected,saved)
            for stage in (card.GEOMETRY,card.MASK):
                result,_,_=self.attempt(stage,directory=directory)
                self.assertEqual(result.status,'matched',result.reason)
            if radius==0:self.assertEqual(torch.count_nonzero(saved.node(card.MASK).outputs['labeled_mask']).item(),24)

    def test_singular_affine_inputs_and_ignored_chip_ids_match_saved_direct_fit(self):
        recipe,saved,_=self.cases['roi'];means=saved.node(card.MEANS);affine=saved.node(card.AFFINE)
        rows=tuple((-1000.0,.2,.3,.4) for _ in range(24))
        direct=card.execute_card_stage('color_card.affine',{'bgr':saved.node(BGR).outputs['bgr'],
            'source_matrix':rows,'target_matrix':saved.node(card.STANDARD).outputs['target_matrix']},{},None)
        directory=self.changed(**{card.MEANS:replace(means,outputs={'source_matrix':rows}),
            card.AFFINE:replace(affine,outputs=direct)})
        result,calls,_=self.attempt(card.AFFINE,directory=directory)
        self.assertEqual(result.status,'matched',result.reason);self.assertEqual(calls,['color_card.affine'])

    def test_lower_limits_reject_before_runtime_loading_or_upload(self):
        for stage in (BGR,card.PREPROCESS,card.GEOMETRY,card.MASK,card.SCALE,card.MEANS,card.STANDARD,card.AFFINE):
            with patch.object(replay,'MAX_REPLAY_CUDA_BYTES',contracts.AUDIT_CUDA_BYTES-1),patch.object(replay,'_runtime') as runtime:
                result,calls,selected=self.attempt(stage)
            self.assertEqual(result.status,'unavailable',result.reason);self.assertFalse(calls);self.assertFalse(selected)
            runtime.assert_not_called()
            if stage!=card.SCALE:
                with patch.object(replay,'MAX_TYPED_SELECTED_BYTES',1),patch.object(replay,'_runtime') as runtime:
                    result,calls,selected=self.attempt(stage)
                self.assertEqual(result.status,'unavailable',result.reason);self.assertFalse(calls);self.assertFalse(selected)
                runtime.assert_not_called()

    def test_runtime_incompatibility_precedes_version_provenance_validation(self):
        def changed(report,specs):
            report['runtime']['calculation_modules_sha256']='0'*64
            node=next(node for node in report['nodes'] if node['stage_id']==card.GEOMETRY)
            next(item for item in dict(node['outputs'])['geometry']['items'] if item[0]=='opencv_version')[1]='other'
        result,calls,selected=self.attempt(card.GEOMETRY,inventory_change=changed)
        self.assertEqual(result.status,'unavailable',result.reason);self.assertFalse(calls);self.assertFalse(selected)

    def test_actual_isolated_per_stage_peak_fits_admission_on_a_custom_stream(self):
        import gc
        for stage in (BGR,card.PREPROCESS,card.GEOMETRY,card.MASK,card.SCALE,card.MEANS,card.STANDARD,card.AFFINE):
            gc.collect();torch.cuda.synchronize();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
            with torch.cuda.stream(torch.cuda.Stream()):result,_,_=self.attempt(stage)
            torch.cuda.synchronize()
            self.assertEqual(result.status,'matched',result.reason)
            self.assertLessEqual(torch.cuda.max_memory_allocated()-base,result.metadata['plan']['cuda_bytes'],stage)
            del result

    def test_strict_final_cancellation_and_callback_errors_publish_nothing(self):
        for invalid in (1,np.bool_(False)):
            result,_,_=self.attempt(card.SCALE,cancel=lambda:invalid)
            self.assertEqual(result.status,'failed');self.assertFalse(result.outputs)
        def fail():raise ValueError('Caller failure is not source unavailability')
        result,_,_=self.attempt(card.SCALE,cancel=fail)
        self.assertEqual(result.status,'failed',result.reason);self.assertFalse(result.outputs)
        stopped=[]
        def stop():
            frame=sys._getframe(1)
            while frame:
                if frame.f_code is card.execute_card_stage.__code__ and frame.f_locals.get('operation')=='color_card.mask' and 'labels' in frame.f_locals:
                    stopped.append(True);return True
                frame=frame.f_back
            return False
        result,_,_=self.attempt(card.MASK,cancel=stop)
        self.assertTrue(stopped);self.assertEqual(result.status,'cancelled',result.reason);self.assertFalse(result.outputs)

    def test_callback_private_cuda_and_cpu_buffers_including_exception_paths(self):
        cases=((card.PREPROCESS,card.execute_card_stage,'local'),
            (card.PREPROCESS,card.preprocess._separable,'horizontal'),
            (card.GEOMETRY,card.contours.trace_binary_contours_cuda,'foreground_full'),
            (card.GEOMETRY,card.card._geometry,'indices'),(card.GEOMETRY,card.card._geometry,'samples'),
            (card.MASK,card.card._rasterize_geometry,'labels'),
            (card.MEANS,card.execute_card_stage,'mask'),(card.MEANS,card.correction.get_color_matrix_cuda,'counts'),
            (card.MEANS,card.correction.get_color_matrix_cuda,'values'),
            (card.STANDARD,card.execute_card_stage,'cuda_matrix'),
            (card.AFFINE,card.correction.affine_color_correction_cuda,'coefficients'),
            (card.AFFINE,card.correction._apply,'coeff'),(card.AFFINE,card.correction._apply,'basis'))
        for stage,function,field in cases:
            with self.subTest(stage=stage,field=field):
                changed=[]
                def callback():
                    frame=sys._getframe(1)
                    while frame:
                        value=frame.f_locals.get(field)
                        if not changed and frame.f_code is function.__code__ and isinstance(value,(torch.Tensor,np.ndarray)):
                            if torch.is_tensor(value):value.zero_()
                            else:
                                value.flat[0]=(float(value.flat[0])+1 if value.dtype.kind=='f'
                                               else int(value.flat[0])^1)
                            changed.append(field);raise RuntimeError('Caller exception after private mutation')
                        frame=frame.f_back
                    return False
                result,_,_=self.attempt(stage,cancel=callback)
                self.assertTrue(changed,field);self.assertEqual(result.status,'failed',result.reason);self.assertFalse(result.outputs)
                self.assertIn('changed',result.reason.lower())

    def test_private_array_headers_and_contour_forest_fields_are_guarded(self):
        cases=(('shape','corners'),('dtype','samples'),('contours','forest'),('hierarchy','forest'))
        for change,field in cases:
            with self.subTest(change=change):
                changed=[]
                def callback():
                    frame=sys._getframe(1)
                    while frame:
                        value=frame.f_locals.get(field)
                        if not changed and frame.f_code is card.card._geometry.__code__ and value is not None:
                            if change=='shape':value.shape=(value.size,)
                            elif change=='dtype':value.dtype=np.int8
                            elif change=='contours':object.__setattr__(value,'contours',())
                            else:object.__setattr__(value,'hierarchy',value.hierarchy.copy())
                            changed.append(change)
                            raise RuntimeError('Caller exception after private header or field replacement')
                        frame=frame.f_back
                    return False
                result,_,_=self.attempt(card.GEOMETRY,cancel=callback)
                self.assertTrue(changed,change);self.assertEqual(result.status,'failed',result.reason)
                self.assertFalse(result.outputs);self.assertIn('changed',result.reason.lower())

    def test_final_compact_and_raster_outputs_and_contract_constants_are_guarded(self):
        for stage in (card.GEOMETRY,card.MASK,card.SCALE,card.AFFINE):
            changed=[]
            def callback():
                frame=sys._getframe(1)
                while frame:
                    outputs=frame.f_locals.get('outputs',{})
                    if (not changed and frame.f_code.co_filename==card.execute_card_stage.__code__.co_filename
                            and frame.f_code.co_name=='finish' and outputs):
                        key=next(iter(outputs));value=outputs[key]
                        if torch.is_tensor(value):value.zero_()
                        else:outputs[key]={} if isinstance(value,dict) else ()
                        changed.append(stage)
                    frame=frame.f_back
                return False
            result,_,_=self.attempt(stage,cancel=callback)
            self.assertTrue(changed);self.assertEqual(result.status,'failed',result.reason);self.assertFalse(result.outputs)
        changed=[];original=contracts.COMPARE_ELEMENTS
        def callback():
            frame=sys._getframe(1)
            while frame:
                if not changed and frame.f_code is card.execute_card_stage.__code__:
                    contracts.COMPARE_ELEMENTS+=1;changed.append(True)
                frame=frame.f_back
            return False
        try:result,_,_=self.attempt(card.SCALE,cancel=callback)
        finally:contracts.COMPARE_ELEMENTS=original
        self.assertTrue(changed);self.assertEqual(result.status,'failed',result.reason);self.assertFalse(result.outputs)

    def test_public_card_boundaries_preserve_mutation_errors_and_final_metadata(self):
        cases=((card.card.detect_color_card_cuda,card.preprocess._separable,'horizontal'),
            (card.card.detect_color_card_cuda,card.contours.trace_binary_contours_cuda,'foreground_full'),
            (card.card.auto_correct_color_cuda,card.correction.get_color_matrix_cuda,'counts'),
            (card.card.auto_correct_color_cuda,card.correction.affine_color_correction_cuda,'coefficients'),
            (card.card.auto_correct_color_cuda,card.card.auto_correct_color_cuda,'target'),
            (card.card.detect_color_card_cuda,card.card.detect_color_card_cuda,'result'),
            (card.card.auto_correct_color_cuda,card.card.auto_correct_color_cuda,'result'))
        for operation,function,field in cases:
            with self.subTest(operation=operation.__name__,field=field):
                source=torch.as_tensor(fixture(varied=True),device='cuda');changed=[]
                def callback():
                    frame=sys._getframe(1)
                    while frame:
                        value=frame.f_locals.get(field)
                        if not changed and frame.f_code is function.__code__ and value is not None:
                            if torch.is_tensor(value):value.zero_()
                            elif isinstance(value,np.ndarray):value.flat[0]=float(value.flat[0])+1
                            elif operation is card.card.detect_color_card_cuda:
                                object.__setattr__(value.geometry,'median_color_chip_width',123.)
                            else:object.__setattr__(value.correction,'fit','changed')
                            changed.append(field);raise RuntimeError('Caller exception after private mutation')
                        frame=frame.f_back
                    return False
                with self.assertRaisesRegex(card.card.AutomaticCardMutation,'changed'):
                    operation(source,adaptive_method=0,cancel=callback)
                self.assertTrue(changed,field)
        failure=ValueError('Unrelated caller exception')
        def fail():raise failure
        for operation in (card.card.detect_color_card_cuda,card.card.auto_correct_color_cuda):
            with self.assertRaises(ValueError) as caught:operation(source,cancel=fail)
            self.assertIs(caught.exception,failure)
            with self.assertRaises(TypeError):operation(source,cancel=lambda:1)

    def test_corruption_of_an_unselected_payload_rejects_before_dispatch(self):
        directory=self.root/'corrupt';shutil.copytree(self.cases['roi'][2],directory)
        path=next(directory.rglob('*.npz'));raw=bytearray(path.read_bytes());raw[len(raw)//2]^=1;path.write_bytes(raw)
        result,calls,_=self.attempt(card.SCALE,directory=directory)
        self.assertEqual(result.status,'invalid_archive',result.reason);self.assertFalse(calls)

    def test_actual_async_card_replay_raster_and_compact_views_preserve_active_run(self):
        app=QApplication.instance() or QApplication([])
        recipe,saved,directory=self.cases['roi'];before=hashes(directory)
        dialog=NativeRecipeDialog(recipe);dialog.resize(1050,650);dialog.show()
        try:
            dialog.source_path=self.image;dialog._completed(saved)
            for stage in (card.PREPROCESS,card.GEOMETRY,card.MASK,card.SCALE,card.MEANS,card.STANDARD,card.AFFINE):
                dialog._select_stage(stage)
                with patch('seedvision.ui.native_recipe_dialog.QFileDialog.getExistingDirectory',return_value=str(directory)):
                    dialog._choose_stage_replay()
                deadline=time.monotonic()+30
                while dialog._worker is not None and time.monotonic()<deadline:app.processEvents();time.sleep(.01)
                self.assertIsNone(dialog._worker);self.assertIs(dialog.result,saved)
                self.assertEqual(dialog.replay_result.status,'matched',dialog.replay_result.reason)
                self.assertNotIn('tensor(',dialog.facts.toPlainText())
                if stage in (card.PREPROCESS,card.MASK,card.AFFINE):self.assertFalse(dialog.preview.pixmap().isNull())
                if stage==card.GEOMETRY:self.assertIn('24 sampling centres',dialog.facts.toPlainText())
                if stage==card.SCALE:self.assertIn('mm per pixel',dialog.facts.toPlainText())
            self.assertEqual(hashes(directory),before)
        finally:dialog.close();app.processEvents();dialog.deleteLater()


if __name__=='__main__':unittest.main()
