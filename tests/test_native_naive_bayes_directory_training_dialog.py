"""Directory training, strict portable provenance, and actual Qt/project saves."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import threading
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtCore import QSaveFile, QTimer
from PySide6.QtWidgets import QApplication, QDialog

from seedvision.benchmarking import naive_bayes_training_directory as directory
from seedvision.benchmarking import native_naive_bayes_recipe as factory
from seedvision.benchmarking import native_recipe as runner
from seedvision.export import native_results as outputs
from seedvision.export.project_native_graphs import export_project_native_graphs
from seedvision.model_io import naive_bayes_provenance as provenance
from seedvision.persistence import reference_models as assets
from seedvision.persistence.project_analysis import ProjectAnalysisStore, InvalidProjectAnalysis
from seedvision.segmentation.baseline import PipelineAnalysisCache
from seedvision.ui import native_naive_bayes_training_dialog as ui
from seedvision.ui import native_naive_bayes_dialog as picker
from tests import test_native_naive_bayes_training_dialog as sample_workflow
from tests.test_plantcv_naive_bayes import _image
from tests.test_plantcv_naive_bayes_sampling import coordinate_pair, pair

bind_recipe = sample_workflow.bind_recipe


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')


def selection_digest(value):
    selection = value['selection']
    selection['selection_sha256'] = sha256(canonical({key:item for key,item in selection.items()
        if key != 'selection_sha256'})).hexdigest()


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class NativeBayesDirectoryTrainingWorkflowTests(unittest.TestCase):
    wait = sample_workflow.NativeBayesTrainingWorkflowTests.wait
    tearDown = sample_workflow.NativeBayesTrainingWorkflowTests.tearDown
    make_project = sample_workflow.NativeBayesTrainingWorkflowTests.make_project

    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        import tempfile
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images, self.masks = self.root/'images-é-种子', self.root/'masks-α'
        self.images.mkdir()
        self.masks.mkdir()
        for index, name in enumerate(('z/same.png', 'a/same.png')):
            image, mask = pair(seed=101+index)
            path = self.images/name
            path.parent.mkdir()
            self.write_image(path, image)
            self.write_image(self.masks/'same.png', mask)
        self.completion = directory.train_bayes_directories(self.images, self.masks,
            rng=np.random.RandomState(17))
        self.model_path = self.root/'trained-directory.tsv'
        self.model_path.write_bytes(self.completion.tsv_bytes)
        self.selected = factory.model_source_for_trained_model(self.model_path,
            self.completion.tsv_bytes, self.completion.provenance_json)
        self.recipe = bind_recipe(self.selected)
        self.image_path = self.root/'seeds.png'
        self.write_image(self.image_path, _image())
        self.dialogs = []

    def write_image(self, path, image):
        ok, raw = cv2.imencode('.png', image)
        self.assertTrue(ok)
        path.write_bytes(raw.tobytes())

    def dialog(self, **kwargs):
        result = ui.NativeNaiveBayesTrainingDialog(**kwargs)
        self.dialogs.append(result)
        result.input_mode.setCurrentIndex(1)
        result.image_directory_edit.setText(str(self.images))
        result.mask_directory_edit.setText(str(self.masks))
        return result

    def prepared_dialog(self):
        result = self.dialog()
        result._run_mode = 'directories'
        result._run_path = (str(self.images.resolve()), str(self.masks.resolve()))
        result._completion = self.completion
        result.save_button.setEnabled(True)
        return result

    def test_actual_constant_directory_training_save_and_project_archive_without_sources(self):
        images,masks = self.root/'constant-images',self.root/'constant-masks'
        images.mkdir(); masks.mkdir()
        image = np.random.default_rng(24).integers(0,255,(10,12,3),dtype=np.uint8)
        image[:,:,0] = 255
        mask = np.zeros((10,12),dtype=np.uint8)
        mask.reshape(-1)[:60] = 255
        self.write_image(images/'seed.png',image)
        self.write_image(masks/'seed.png',mask)
        dialog = self.dialog()
        dialog.image_directory_edit.setText(str(images))
        dialog.mask_directory_edit.setText(str(masks))
        state = np.random.get_state()
        try:
            np.random.seed(17)
            dialog._start_training()
            self.wait(dialog)
        finally:
            np.random.set_state(state)
        self.assertIsNotNone(dialog._completion)
        self.assertEqual(dialog._completion.model.class_names,('plant','background'))
        for fits in dialog._completion.fit.channels:
            self.assertGreater(fits[2].mean,255)
        destination = self.root/'constant-directory.tsv'
        with patch.object(ui.QFileDialog,'getSaveFileName',return_value=(str(destination),'tsv')):
            dialog._save_model()
        self.assertEqual(dialog.result(),QDialog.DialogCode.Accepted)
        self.assertEqual(destination.read_bytes(),dialog._completion.tsv_bytes)
        sample_workflow.assert_constant_project_archive_after_removal(
            self,dialog.saved_model_source,(images,masks,destination))

    def test_actual_different_dimension_training_save_move_and_archive_without_sources(self):
        images,masks = self.root/'coordinate-images',self.root/'coordinate-masks'
        images.mkdir(); masks.mkdir()
        for index,(image_shape,mask_shape) in enumerate((((31,37),(20,19)),((20,19),(31,37)))):
            image,mask = coordinate_pair(image_shape,mask_shape,seed=991+index)
            self.write_image(images/f'{index}.png',image)
            self.write_image(masks/f'{index}.png',mask)
        dialog = self.dialog()
        dialog.image_directory_edit.setText(str(images))
        dialog.mask_directory_edit.setText(str(masks))
        state = np.random.get_state()
        try:
            np.random.seed(17)
            dialog._start_training()
            self.wait(dialog)
        finally:
            np.random.set_state(state)
        self.assertIsNotNone(dialog._completion,dialog.status_label.text())
        record = provenance.parse_training_record(dialog._completion.provenance_json,
            dialog._completion.model).metadata
        self.assertEqual(record['parser_id'],provenance.COORDINATE_DIRECTORY_PARSER_ID)
        self.assertTrue(all(o['image_shape'][:2]!=o['mask_shape'] for o in record['occurrences']))
        destination = self.root/'coordinate-directory.tsv'
        with patch.object(ui.QFileDialog,'getSaveFileName',return_value=(str(destination),'tsv')):
            dialog._save_model()
        self.assertEqual(dialog.result(),QDialog.DialogCode.Accepted)
        self.assertEqual(destination.read_bytes(),dialog._completion.tsv_bytes)
        sample_workflow.assert_constant_project_archive_after_removal(
            self,dialog.saved_model_source,(images,masks,destination))

    def test_v1_record_bytes_cache_and_project_archive_survive_v2_producer(self):
        legacy = json.loads(self.completion.provenance_json)
        self.assertEqual(legacy['parser_id'],provenance.COORDINATE_DIRECTORY_PARSER_ID)
        legacy['parser_id'] = provenance.DIRECTORY_PARSER_ID
        for occurrence in legacy['occurrences']:
            plan = occurrence['sampling_plan']
            plan.pop('mask_height'); plan.pop('mask_width')
            pixels = plan['height']*plan['width']
            tiles,n = plan['mask_tiles'],plan['samples_per_class']
            conversion,gather = plan['conversion_tile'],plan['gather_tile']
            gathers = 0 if not n else 6*((n+gather-1)//gather)
            plan['work'] = 96*pixels+8*tiles+2048*((pixels+conversion-1)//conversion)+12*n+256*gathers
            plan['cpu_metadata_bytes'] = 24*n+64*tiles+128*1024
            plan['peak_live_bytes'] = (7*pixels+4*(plan['foreground_pixels']+plan['background_pixels'])
                +16*tiles+6*n+128*conversion+16*gather+4*1024*1024)
        legacy['sampling_work'] = sum(o['sampling_plan']['work'] for o in legacy['occurrences'])
        payload = canonical(legacy)
        record = provenance.parse_training_record(payload,self.completion.model)
        self.assertEqual(record.json_text.encode('utf-8'),payload)
        selected = factory.model_source_for_trained_model(self.model_path,self.completion.tsv_bytes,payload)
        cache = PipelineAnalysisCache()
        first = runner.run_native_recipe_path(self.recipe,self.image_path,cache=cache)
        current = runner.run_native_recipe_path(bind_recipe(selected),self.image_path,cache=cache)
        self.assertTrue(all(node.reused for node in current.nodes))
        self.assertIs(first.node(factory.CLASSIFIER_STAGE_ID).outputs['class_masks'],
            current.node(factory.CLASSIFIER_STAGE_ID).outputs['class_masks'])
        sample_workflow.assert_constant_project_archive_after_removal(
            self,selected,(self.images,self.masks,self.model_path))

    def test_v2_exact_dimension_shape_profile_and_intersection_facts_are_required(self):
        image,mask = coordinate_pair((20,43),(31,37))
        self.write_image(self.images/'coordinate.png',image)
        self.write_image(self.masks/'coordinate.png',mask)
        result = directory.train_bayes_directories(self.images,self.masks,rng=np.random.RandomState(17))
        original = provenance.parse_training_record(result.provenance_json,result.model).metadata
        index = next(i for i,o in enumerate(original['occurrences'])
            if o['pair']['image_relative_path']=='coordinate.png')
        changes = (
            lambda v:v.__setitem__('parser_id',provenance.DIRECTORY_PARSER_ID),
            lambda v:v.__setitem__('parser_id','plantcv-4.11.3-ordered-binary-directories-v3'),
            lambda v:v['occurrences'][index]['sampling_plan'].pop('mask_height'),
            lambda v:v['occurrences'][index]['sampling_plan'].__setitem__('extra',0),
            lambda v:v['occurrences'][index]['sampling_plan'].__setitem__('mask_width',True),
            lambda v:v['occurrences'][index]['sampling_plan'].__setitem__('mask_height',20),
            lambda v:v['occurrences'][index]['mask_shape'].__setitem__(0,True),
            lambda v:v['occurrences'][index]['mask_shape'].__setitem__(0,16_000_000),
            lambda v:v['occurrences'][index]['image_shape'].__setitem__(0,16_000_000),
            lambda v:v['occurrences'][index]['sampling_plan'].__setitem__('background_pixels',371),
            lambda v:v['occurrences'][index]['sampling_plan'].__setitem__('cpu_metadata_bytes',1),
            lambda v:v['occurrences'][index]['sampling_plan'].__setitem__('work',1),
            lambda v:v['occurrences'][index]['sampling_plan'].__setitem__('peak_live_bytes',1),
        )
        for case,change in enumerate(changes):
            value = deepcopy(original)
            change(value)
            with self.subTest(case=case),self.assertRaises(ValueError):
                provenance.parse_training_record(canonical(value),result.model)

    def test_distinct_unicode_record_exact_binding_legacy_compatibility_and_independent_metadata(self):
        record = provenance.parse_training_record(self.completion.provenance_json, self.completion.model)
        self.assertEqual(record.to_binding(), self.selected[factory.TRAINING_FIELD])
        self.assertIn('种子', record.json_text)
        changed = record.metadata
        changed['occurrences'][0]['mask']['bytes'] += 1
        self.assertNotEqual(changed, record.metadata)
        plain = factory.model_source_for_path(self.model_path)
        self.assertEqual(set(plain), set(factory.MODEL_FIELDS))
        self.assertEqual(factory.load_selected_model(plain), factory.load_selected_model(self.selected))
        restored = type(self.recipe).from_json(self.recipe.to_json())
        self.assertEqual(restored, self.recipe)
        with self.assertRaises(ValueError):
            provenance.parse_training_record(json.dumps(record.metadata, sort_keys=True,
                separators=(',', ':')).encode(), self.completion.model)
        for kind, parser in (('native_bayes_sample_training', provenance.PARSER_ID),
                ('native_bayes_directory_training', provenance.PARSER_ID)):
            value = record.metadata
            value['kind'], value['parser_id'] = kind, parser
            with self.subTest(kind=kind, parser=parser), self.assertRaises(ValueError):
                provenance.parse_training_record(canonical(value), self.completion.model)

    def test_typed_directory_manifest_source_order_and_unique_bytes_reject_corruption(self):
        original = json.loads(self.completion.provenance_json)
        variants = []
        for change in (
                lambda v:v['selection'].__setitem__('selection_sha256', 'a'*64),
                lambda v:v['selection']['pairs'].reverse(),
                lambda v:v['selection']['pairs'][0].__setitem__('image_relative_path', '../same.png'),
                lambda v:v['selection']['pairs'][0].__setitem__('mask_relative_path', 'nested/same.png'),
                lambda v:v['selection']['pairs'][0].__setitem__('index', False),
                lambda v:v['selection']['walk'][0][2].append('wrong.png'),
                lambda v:v['selection']['skipped_missing_masks'].append('missing.png'),
                lambda v:v['occurrences'].reverse(),
                lambda v:v['occurrences'][0]['mask'].__setitem__('sha256', 'b'*64),
                lambda v:v['occurrences'][0]['image'].__setitem__('bytes', 64*1024*1024+1),
                lambda v:v.__setitem__('source_bytes', v['source_bytes']+1)):
            value = deepcopy(original)
            change(value)
            variants.append(value)
        # A skipped .jpeg must not become an eligible pair under a repaired hash.
        value = deepcopy(original)
        old = value['selection']['pairs'][0]['image_relative_path']
        parent, _ = old.rsplit('/', 1)
        value['selection']['pairs'][0]['image_relative_path'] = parent+'/same.jpeg'
        value['selection']['pairs'][0]['mask_relative_path'] = 'same.jpeg'
        for walk in value['selection']['walk']:
            if walk[0] == parent:
                walk[2][walk[2].index('same.png')] = 'same.jpeg'
        value['occurrences'][0]['pair'] = deepcopy(value['selection']['pairs'][0])
        variants.append(value)
        # Repair selection hashes except the digest-specific case, so these
        # cases exercise typed order/path rules rather than just byte identity.
        for index, value in enumerate(variants):
            if index:
                selection_digest(value)
            with self.subTest(index=index), self.assertRaises(ValueError):
                provenance.parse_training_record(canonical(value), self.completion.model)
        value = deepcopy(original)
        value['selection']['walk'][0][2].append('excluded.jpeg')
        selection_digest(value)
        # Excluded names are valid descriptive walk entries, without pairs.
        provenance.parse_training_record(canonical(value), self.completion.model)

    def test_typed_rng_plans_runtime_and_model_facts_reject_inconsistent_records(self):
        original = json.loads(self.completion.provenance_json)
        changes = (
            lambda v:v['rng_before'].__setitem__('position', True),
            lambda v:v['rng_before'].__setitem__('keys', '00'),
            lambda v:v['rng_after'].__setitem__('cached_gaussian', 'F'*16),
            lambda v:v['rng_after'].__setitem__('sha256', 'a'*64),
            lambda v:v['occurrences'][0].__setitem__('rng_before_sha256', 'b'*64),
            lambda v:v['occurrences'][1].__setitem__('rng_before_sha256', v['rng_before']['sha256']),
            lambda v:v['occurrences'][0].__setitem__('random_calls', True),
            lambda v:v['occurrences'][0].__setitem__('random_index_dtype', '<i2'),
            lambda v:v['occurrences'][0].__setitem__('draws_sha256', 'bad'),
            lambda v:v['occurrences'][0]['image_shape'].__setitem__(0, 1),
            lambda v:v['occurrences'][0]['sampling_plan'].__setitem__('foreground_pixels', True),
            lambda v:v['occurrences'][0]['sampling_plan'].__setitem__('background_pixels', 16_000_000),
            lambda v:v['occurrences'][0]['sampling_plan'].__setitem__('gather_tile', 0),
            lambda v:v['occurrences'][0]['sampling_plan'].__setitem__('conversion_tile', 0),
            lambda v:v['occurrences'][0]['sampling_plan'].__setitem__('work', 1),
            lambda v:v['occurrences'][0]['sampling_plan'].__setitem__('peak_live_bytes', 1),
            lambda v:v['occurrences'][0]['sampling_plan'].__setitem__('cpu_metadata_bytes', 1),
            lambda v:v['fit_preflight'].__setitem__('peak_live_bytes', 1),
            lambda v:v['fit_preflight'].__setitem__('work', v['fit_preflight']['work']+1),
            lambda v:v.__setitem__('sampling_work', True),
            lambda v:v.__setitem__('sample_counts', [111,110]),
            lambda v:v.__setitem__('model_sha256', 'a'*64),
            lambda v:v.__setitem__('tables_sha256', 'a'*64),
            lambda v:v['runtime'].__setitem__('opencv_build_sha256', 'bad'),
            lambda v:v['calculation'].pop(directory.__name__),
            lambda v:v['channel_fits'][0][0].__setitem__('numerical_profile', 'certified exact'),
            lambda v:v.__setitem__('scientific_validation', True),
        )
        for index, change in enumerate(changes):
            value = deepcopy(original)
            change(value)
            with self.subTest(index=index), self.assertRaises(ValueError):
                provenance.parse_training_record(canonical(value), self.completion.model)

    def test_zero_row_occurrence_preserves_zero_gather_and_rng_chain(self):
        image, mask = pair(foreground=9)
        self.write_image(self.images/'zero.png', image)
        self.write_image(self.masks/'zero.png', mask)
        result = directory.train_bayes_directories(self.images, self.masks, rng=np.random.RandomState(17))
        record = provenance.parse_training_record(result.provenance_json, result.model).metadata
        zero = next(o for o in record['occurrences'] if o['sampling_plan']['samples_per_class'] == 0)
        self.assertEqual(zero['sampling_plan']['gather_tile'], 0)
        self.assertEqual(zero['rng_before_sha256'], zero['rng_after_sha256'])
        zero['rng_after_sha256'] = 'a'*64
        with self.assertRaises(ValueError):
            provenance.parse_training_record(canonical(record), result.model)

    def test_source_impossible_completed_population_and_mixed_index_dtype_are_rejected(self):
        original = json.loads(self.completion.provenance_json)
        for background in (0,1):
            value = deepcopy(original)
            plan = value['occurrences'][0]['sampling_plan']
            plan['peak_live_bytes'] -= 4*(plan['background_pixels']-background)
            plan['background_pixels'] = background
            # Keep derived live/work/row facts consistent; the completed source
            # could not make its nonempty background draw from these populations.
            with self.subTest(background=background), self.assertRaisesRegex(ValueError, 'impossible'):
                provenance.parse_training_record(canonical(value), self.completion.model)
        value = deepcopy(original)
        value['occurrences'][1]['random_index_dtype'] = ('<i8' if
            value['occurrences'][0]['random_index_dtype'] == '<i4' else '<i4')
        with self.assertRaisesRegex(ValueError, 'dtype within one job'):
            provenance.parse_training_record(canonical(value), self.completion.model)

    def test_directory_annotation_reuses_predictions_and_refreshes_current_archive_metadata(self):
        value = json.loads(self.completion.provenance_json)
        value['selection']['image_root_origin'] = str(self.root/'former-images')
        selection_digest(value)
        selected = factory.model_source_for_trained_model(self.model_path, self.completion.tsv_bytes, canonical(value))
        changed = bind_recipe(selected)
        cache = PipelineAnalysisCache()
        first = runner.run_native_recipe_path(self.recipe, self.image_path, cache=cache)
        result = runner.run_native_recipe_path(changed, self.image_path, cache=cache)
        self.assertTrue(all(node.reused for node in result.nodes))
        self.assertIs(first.node(factory.CLASSIFIER_STAGE_ID).outputs['class_masks'],
            result.node(factory.CLASSIFIER_STAGE_ID).outputs['class_masks'])
        outputs.export_native_recipe_result(self.root/'changed-history', changed, result)
        report, _ = outputs.load_native_recipe_result(self.root/'changed-history')
        archived = report['recipe']['stages'][1]['parameters'][0]['value'][factory.TRAINING_FIELD]
        self.assertEqual(json.loads(archived['json'])['selection']['image_root_origin'], str(self.root/'former-images'))

    def test_real_async_directory_mode_global_stream_compact_facts_and_atomic_save(self):
        dialog = self.dialog()
        original = np.random.get_state()
        try:
            np.random.seed(17)
            dialog._start_training()
            for widget in (dialog.input_mode, dialog.samples_edit, dialog.image_directory_edit,
                    dialog.mask_directory_edit, dialog.choose_images_button, dialog.choose_masks_button,
                    dialog.train_button, dialog.save_button):
                self.assertFalse(widget.isEnabled())
            self.wait(dialog)
        finally:
            np.random.set_state(original)
        self.assertTrue(dialog.save_button.isEnabled(), dialog.status_label.text())
        self.assertIn('2 image/mask pairs', dialog.status_label.text())
        self.assertIn('plant: 110, background: 110', dialog.status_label.text())
        self.assertIn('empirical', dialog.status_label.text())
        self.assertIn('Keep', dialog.status_label.text())
        destination = self.root/'qt-directory.tsv'
        with patch.object(ui.QFileDialog, 'getSaveFileName', return_value=(str(destination), '')):
            dialog._save_model()
        self.assertEqual(destination.read_bytes(), self.completion.tsv_bytes)
        self.assertEqual(factory.load_selected_model(dialog.saved_model_source), self.completion.model)
        self.assertEqual(dialog.result(), QDialog.DialogCode.Accepted)
        self.assertEqual(dialog.saved_model_source[factory.TRAINING_FIELD]['json'], dialog._completion.provenance_json.decode())

    def test_input_mode_dispatch_frozen_roots_and_same_path_reselection_clear_old_completion(self):
        entered, release = threading.Event(), threading.Event()
        received = []
        def delayed(images, masks, *, cancel):
            received.append((images, masks))
            entered.set()
            release.wait(5)
            return self.completion
        dialog = self.dialog(directory_operation=delayed)
        dialog._start_training()
        self.assertTrue(entered.wait(5))
        self.assertEqual(received, [(str(self.images.resolve()), str(self.masks.resolve()))])
        dialog.input_mode.setCurrentIndex(0)
        self.assertTrue(dialog._cancel.is_set())
        release.set()
        self.wait(dialog)
        self.assertIsNone(dialog._completion)
        self.assertFalse(dialog.save_button.isEnabled())
        dialog.input_mode.setCurrentIndex(1)
        dialog._completion = self.completion
        dialog.save_button.setEnabled(True)
        with patch.object(ui.QFileDialog, 'getExistingDirectory', return_value=str(self.images)):
            dialog._choose_directory(dialog.image_directory_edit, 'Choose images')
        self.assertIsNone(dialog._completion)
        self.assertFalse(dialog.save_button.isEnabled())

    def test_directory_completion_requires_matching_mode_type_and_origins_without_dense_traversal(self):
        class DenseSentinel:
            def __getattribute__(self, name):
                raise AssertionError('UI inspected dense directory training products')
        dialog = self.prepared_dialog()
        compact = replace(self.completion, samples=DenseSentinel(), fit=DenseSentinel(), selection=DenseSentinel())
        self.assertEqual(dialog._validated_completion(compact)['sample_counts'], [110,110])
        dialog._run_path = (str(self.root/'other'), str(self.masks.resolve()))
        with self.assertRaisesRegex(ValueError, 'origins'):
            dialog._validated_completion(compact)
        dialog._run_mode = 'samples'
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            dialog._validated_completion(compact)

    def test_directory_stop_change_and_close_drain_without_late_selection(self):
        for action in ('stop','images','masks','reject','accept','close'):
            entered, release = threading.Event(), threading.Event()
            def delayed(images, masks, *, cancel):
                entered.set()
                release.wait(5)
                return self.completion
            dialog = self.dialog(directory_operation=delayed)
            dialog._start_training()
            self.assertTrue(entered.wait(5))
            finished = []
            dialog.finished.connect(finished.append)
            if action == 'stop':
                dialog._stop_training()
            elif action in ('images','masks'):
                edit = dialog.image_directory_edit if action == 'images' else dialog.mask_directory_edit
                edit.setText(str(self.root/'changed'))
                self.assertTrue(dialog._cancel.is_set())
            else:
                getattr(dialog, action)()
                self.assertEqual(finished, [])
            release.set()
            self.wait(dialog)
            with self.subTest(action=action):
                self.assertIsNone(dialog._completion)
                self.assertIsNone(dialog.saved_model_source)
                self.assertFalse(dialog.save_button.isEnabled())
                if action in ('reject','accept','close'):
                    self.assertEqual(finished, [QDialog.DialogCode.Rejected])

    def test_two_directory_jobs_serialize_global_stream_and_cancelled_queue_does_not_dispatch(self):
        first_entered, second_entered, release = threading.Event(), threading.Event(), threading.Event()
        def first(images, masks, *, cancel):
            first_entered.set()
            release.wait(5)
            return self.completion
        def second(images, masks, *, cancel):
            second_entered.set()
            return self.completion
        first_dialog = self.dialog(directory_operation=first)
        second_dialog = self.dialog(directory_operation=second)
        first_dialog._start_training()
        self.assertTrue(first_entered.wait(5))
        second_dialog._start_training()
        self.assertFalse(second_entered.wait(.15))
        second_dialog._stop_training()
        self.wait(second_dialog)
        self.assertFalse(second_entered.is_set())
        self.assertIsNone(second_dialog._completion)
        release.set()
        self.wait(first_dialog)
        self.assertIsNotNone(first_dialog._completion)
        second_dialog._start_training()
        self.wait(second_dialog)
        self.assertTrue(second_entered.is_set())
        self.assertIsNotNone(second_dialog._completion)

    def test_directory_atomic_short_write_commit_and_reload_failures_never_select_a_model(self):
        dialog = self.prepared_dialog()
        destination = self.root/'existing.tsv'
        destination.write_bytes(b'previous bytes')
        for mode in ('short','commit'):
            class FailedSave:
                def __init__(self, path): self.target = QSaveFile(path)
                def open(self, flags): return self.target.open(flags)
                def write(self, payload):
                    result = self.target.write(payload)
                    return result-1 if mode == 'short' else result
                def commit(self): return False
                def errorString(self): return 'injected failure'
                def cancelWriting(self): self.target.cancelWriting()
            with (patch.object(ui, 'QSaveFile', FailedSave),
                    patch.object(ui.QFileDialog, 'getSaveFileName', return_value=(str(destination), '')),
                    patch.object(ui.QMessageBox, 'warning')):
                dialog._save_model()
            self.assertEqual(destination.read_bytes(), b'previous bytes')
            self.assertIsNone(dialog.saved_model_source)
        with (patch.object(ui.QFileDialog, 'getSaveFileName', return_value=(str(destination), '')),
                patch.object(ui, 'model_source_for_trained_model', side_effect=ValueError('changed after commit')),
                patch.object(ui.QMessageBox, 'warning')):
            dialog._save_model()
        self.assertEqual(destination.read_bytes(), self.completion.tsv_bytes)
        self.assertIsNone(dialog.saved_model_source)

    def test_directory_record_project_move_and_actual_batch_archive_without_training_sources(self):
        workspace, store, master = self.make_project()
        self.assertTrue(self.images.resolve().is_relative_to(self.root.resolve()))
        self.assertTrue(self.masks.resolve().is_relative_to(self.root.resolve()))
        shutil.rmtree(self.images)
        shutil.rmtree(self.masks)
        self.model_path.unlink()
        moved = self.root/'moved-workspace'
        self.assertTrue(workspace.resolve().is_relative_to(self.root.resolve()))
        self.assertTrue(moved.resolve().is_relative_to(self.root.resolve()))
        workspace.rename(moved)
        loaded = ProjectAnalysisStore(moved).load(moved/master.name)
        self.assertFalse(loaded.issues)
        recipe = loaded.document.native_recipes[0]
        self.assertEqual(recipe, self.recipe)
        paths = assets.model_asset_paths(moved, recipe, loaded.document.reference_model_assets)
        result = runner.run_native_recipe_path(recipe, moved/'seeds.png', model_asset_paths=paths)
        self.assertEqual(result.status, 'executed')
        ledger = export_project_native_graphs((), moved/'seeds.png', self.root/'batch',
            expected_source_sha256=loaded.document.images[0].source.sha256, project_root=moved,
            reference_model_assets=loaded.document.reference_model_assets, native_recipes=(recipe,))
        row = json.loads(ledger.read_bytes())['graphs'][0]
        self.assertEqual(row['status'], 'executed', row.get('reason'))
        report, arrays = outputs.load_native_recipe_result(ledger.parent/Path(row['report']).parent)
        archived = report['recipe']['stages'][1]['parameters'][0]['value'][factory.TRAINING_FIELD]
        self.assertEqual(archived, self.selected[factory.TRAINING_FIELD])
        self.assertTrue(arrays)
        self.assertFalse(report['scientifically_validated'])

    def test_invalid_directory_record_is_rejected_before_model_or_project_asset_reads(self):
        invalid = deepcopy(self.selected)
        record = json.loads(invalid[factory.TRAINING_FIELD]['json'])
        record['occurrences'][0]['sampling_plan']['work'] = 1
        raw = canonical(record)
        invalid[factory.TRAINING_FIELD] = dict(json=raw.decode(), sha256=sha256(raw).hexdigest())
        with patch.object(factory, 'load_plantcv_pdf', side_effect=AssertionError('read before metadata check')):
            with self.assertRaises(ValueError):
                factory.load_selected_model(invalid)
        with patch.object(assets, 'verified_asset_bytes', side_effect=AssertionError('read before metadata check')):
            with self.assertRaises(InvalidProjectAnalysis):
                self.make_project(bind_recipe(invalid))

    def test_parent_close_drains_modal_directory_training_and_releases_completed_cuda_fit(self):
        parent = picker.NativeNaiveBayesDialog(self.recipe)
        self.dialogs.append(parent)
        release = threading.Event()
        def delayed(images, masks, *, cancel):
            release.wait(5)
            return self.completion
        child = self.dialog(parent=parent, directory_operation=delayed)
        finished = []
        parent.finished.connect(lambda code:finished.append((code,child._worker)))
        def start_and_close():
            child._start_training()
            parent.reject()
            self.assertEqual(finished, [])
            release.set()
        QTimer.singleShot(0, start_and_close)
        with patch.object(picker, 'NativeNaiveBayesTrainingDialog', return_value=child):
            parent._train_model()
        self.assertEqual(finished, [(QDialog.DialogCode.Rejected,None)])
        self.assertIsNone(child._completion)
        self.assertIsNone(child.saved_model_source)
        self.assertEqual(parent.recipe, self.recipe)


if __name__ == '__main__':
    unittest.main()
