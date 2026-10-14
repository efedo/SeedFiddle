"""Actual asynchronous calibrated capture and explicitly injected UI faults."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import cv2
import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from seedvision.benchmarking.checkpoints import CheckpointCancelled, CheckpointLimits, _thaw_json
from seedvision.benchmarking.reference_runner import PlantCVReferenceRunner, ReferenceExecutionError
from seedvision.ui import calibrated_public_capture_dialog as ui
from tests.test_native_calibrated_seed_recipe import fixture
from tests.test_reference_calibrated_seed import spec
from tests.test_reference_preprocessing import HAS_REFERENCE_ENV, REFERENCE_PYTHON


def wait_for_worker(app,dialog,seconds=180):
    deadline = time.monotonic()+seconds
    while dialog._worker is not None and time.monotonic()<deadline:
        app.processEvents()
        time.sleep(0.005)
    app.processEvents()
    if dialog._worker is not None:
        raise AssertionError('The same confirmed worker did not drain before the test observation deadline.')


@unittest.skipUnless(HAS_REFERENCE_ENV,'Pinned public PlantCV environment is required.')
class ActualCalibratedPublicDesktopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.temporary = tempfile.TemporaryDirectory(prefix='calibrated-public-qt-')
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.source = cls.root/'source.png'
        cls.bgr = fixture()
        ok,encoded = cv2.imencode('.png',cls.bgr)
        if not ok:
            raise AssertionError('Fixture encoding failed.')
        cls.source.write_bytes(encoded.tobytes())
        cls.selected = spec()
        cls.store = cls.root/'actual'
        cls.dialog = cls._make_dialog(cls.store)
        cls.addClassCleanup(cls.dialog.close)
        cls.dialog.show()
        cls.results,cls.requests = [],[]
        actual = PlantCVReferenceRunner.run
        def observe(runner,operation,arrays,parameters=None,**kwargs):
            result = actual(runner,operation,arrays,parameters,**kwargs)
            cls.results.append(result)
            cls.requests.append((operation,deepcopy(parameters)))
            return result
        cls.timer_ticks = 0
        def tick():
            cls.timer_ticks += 1
        timer = QTimer()
        timer.timeout.connect(tick)
        timer.start(5)
        with patch.object(PlantCVReferenceRunner,'run',observe):
            cls.dialog.start_capture()
            wait_for_worker(cls.app,cls.dialog)
        timer.stop()
        if cls.dialog._reply is None or cls.dialog._reply.session.verified is None:
            raise AssertionError(cls.dialog.status_label.text())
        cls.reply = cls.dialog._reply
        cls.partial_store = cls.root/'partial'
        cls.partial_dialog = cls._make_dialog(cls.partial_store)
        cls.addClassCleanup(cls.partial_dialog.close)
        with patch.object(PlantCVReferenceRunner,'run',cls._replay(fail_color=True)):
            cls.partial_dialog.start_capture()
            wait_for_worker(cls.app,cls.partial_dialog)
        if cls.partial_dialog._reply is None:
            raise AssertionError(cls.partial_dialog.status_label.text())
        cls.partial_reply = cls.partial_dialog._reply

    @classmethod
    def _make_dialog(cls,store):
        return ui.CalibratedPublicReferenceCaptureDialog(cls.selected.seed_recipe,source_path=cls.source,
            store=store,card_parameters=_thaw_json(cls.selected.card_parameters),
            seed_roi_xyxy=cls.selected.seed_roi_xyxy,runner_factory=lambda:PlantCVReferenceRunner(REFERENCE_PYTHON))

    @classmethod
    def _replay(cls,fail_color=False,fail_size=False):
        index = 0
        def run(runner,operation,arrays,parameters=None,**kwargs):
            nonlocal index
            if (operation,parameters) != cls.requests[index]:
                raise AssertionError('Replay calculation inputs differ from the actual selected capture.')
            result = cls.results[index]
            index += 1
            if fail_color and operation=='analysis.color':
                raise ReferenceExecutionError('Declared color fault after actual captured watershed context.')
            if fail_size and index==8:
                raise ReferenceExecutionError('Declared pixel-size fault; source-scaled size remains independent.')
            return replace(result,arrays={name:array.copy() for name,array in result.arrays.items()},
                identity=deepcopy(dict(result.identity)),effective_parameters=deepcopy(dict(result.effective_parameters)))
        return run

    def test_actual_async_capture_has_exact_selection_context_and_source_observations(self):
        self.assertEqual(len(self.requests),10)
        self.assertGreater(self.timer_ticks,0)
        session = self.reply.session
        self.assertEqual(session.spec,self.selected)
        self.assertEqual(self.requests[0][1],self.selected.to_dict()['card_parameters'])
        record = session.verified.record
        self.assertEqual(record['source_sha256'],sha256(self.source.read_bytes()).hexdigest())
        self.assertEqual(record['crop']['local_to_parent_xy'],(600,260))
        self.assertEqual(record['crop']['coordinate_frame']['shape'],(96,128))
        self.assertEqual(record['scaled_measurement']['units'],'mm')
        self.assertFalse(record['scientifically_validated'])
        np.testing.assert_array_equal(session.verified.checkpoints['parent'].arrays['rgb'],self.bgr[...,::-1])
        facts = json.loads(self.dialog.facts.toPlainText())
        self.assertIn('visualizations',facts['actual_source_metadata']['seed.color'])
        self.assertIn('observations',facts['actual_source_metadata']['scaled_size'])
        self.assertIn('Verified complete calibrated capture',self.dialog.status_label.text())

    def test_actual_saved_open_restores_controls_then_edit_invalidates_without_original_source(self):
        dialog = self._make_dialog(self.root/'unrelated-store')
        self.addCleanup(dialog.close)
        dialog.seed_roi_editor.setText('[0,0,2,2]')
        dialog._card_editors['radius'].setText('23')
        with patch.object(ui.inputs,'source_identity',side_effect=AssertionError('Saved open must not hash current image.')):
            dialog.open_saved(self.reply.session.manifest_path)
            wait_for_worker(self.app,dialog)
        self.assertEqual(dialog.selected_spec(),self.selected)
        self.assertEqual(dialog._recipe,self.selected.seed_recipe)
        self.assertIsNone(dialog._source_path)
        self.assertEqual(dialog._store,self.store.resolve())
        self.assertEqual(dialog._reply.session.manifest_sha256,self.reply.session.manifest_sha256)
        dialog.seed_roi_editor.setText('[601,260,729,356]')
        self.assertIsNone(dialog._reply)
        self.assertEqual(dialog.artifact_selector.count(),0)
        self.assertTrue(dialog.preview.pixmap().isNull())
        dialog.start_capture()
        self.assertIn('Select an existing source image',dialog.status_label.text())

    def test_declared_color_failure_keeps_verified_partial_phases_and_separate_scaled_units(self):
        reply = self.partial_reply
        self.assertIsNone(reply.session.verified)
        self.assertIsNone(reply.session.manifest_path)
        self.assertEqual(reply.session.outcome.child.calls[2].status,'failed')
        expected = {'correction','seed.watershed','seed.size','scaled_size',
                    *(f'seed.preprocessing.{index}' for index in range(5))}
        self.assertEqual(set(reply.source_metadata),expected)
        self.assertTrue(all(fact['status']=='checkpoint_verified' for fact in reply.checkpoint_facts.values()))
        self.assertEqual(reply.checkpoint_facts['seed.size']['measurement']['units'],'pixel')
        self.assertEqual(reply.checkpoint_facts['scaled_size']['measurement']['units'],'mm')
        self.assertTrue(any(artifact.key=='seed.watershed.diagnostic_bgr' for artifact in reply.artifacts))
        self.assertEqual(list(self.partial_store.glob('calibrated/chains/*.json')),[])
        self.assertIn('No complete capture manifest',self.partial_dialog.status_label.text())
        self.assertIn('Declared color fault',self.partial_dialog.facts.toPlainText())

    def test_declared_pixel_size_failure_still_displays_independent_scaled_size_and_color(self):
        dialog = self._make_dialog(self.root/'partial-size')
        self.addCleanup(dialog.close)
        with patch.object(PlantCVReferenceRunner,'run',self._replay(fail_size=True)):
            dialog.start_capture()
            wait_for_worker(self.app,dialog)
        self.assertIsNotNone(dialog._reply,dialog.status_label.text())
        reply = dialog._reply
        self.assertIsNone(reply.session.verified)
        self.assertEqual(reply.session.outcome.child.calls[1].status,'failed')
        self.assertIn('seed.color',reply.source_metadata,
            reply.checkpoint_facts.get('seed.color',{}))
        self.assertIn('scaled_size',reply.source_metadata)
        self.assertNotIn('seed.size',reply.source_metadata)

    def test_preview_budget_raw_hashes_BGR_conversion_and_compact_counts(self):
        for reply in (self.reply,self.partial_reply):
            pixels = sum(artifact.image.shape[0]*artifact.image.shape[1] for artifact in reply.artifacts
                         if artifact.image is not None)
            self.assertLessEqual(pixels,ui.MAX_TOTAL_PREVIEW_PIXELS)
            for value in reply.compact_arrays.values():
                self.assertLessEqual(value['displayed_elements'],4096)
                self.assertEqual(value['truncated'],value['displayed_elements']<value['original_elements'])
        verified = self.reply.session.verified
        array = verified.checkpoints['seed.watershed'].arrays['diagnostic_bgr']
        artifact = next(value for value in self.reply.artifacts if value.key=='seed.watershed.diagnostic_bgr')
        self.assertEqual(artifact.raw_sha256,sha256(array.tobytes(order='C')).hexdigest())
        np.testing.assert_array_equal(artifact.image[0,0],array[0,0,::-1])
        self.assertIn('BGR capture adapted to RGB',artifact.description)
        self.assertFalse(artifact.image.flags.writeable)

    def test_partial_budget_is_aggregate_and_corrupt_checkpoint_is_unavailable(self):
        cancel = threading.Event()
        total = self.partial_reply.retained_numeric_bytes
        limits = replace(CheckpointLimits(),max_total_array_bytes=total,max_array_bytes=total)
        reply = ui._display_reply(self.partial_reply.session,self.partial_store,cancel,limits=limits)
        self.assertEqual(reply.retained_numeric_bytes,total)
        reply = ui._display_reply(self.partial_reply.session,self.partial_store,cancel,
            limits=replace(limits,max_total_array_bytes=total-1,max_array_bytes=total-1))
        self.assertTrue(any(value['status']=='checkpoint_unavailable' for value in reply.checkpoint_facts.values()))
        checkpoint = self.partial_reply.session.outcome.scaled_size.checkpoint_reference
        path = self.partial_store/(checkpoint.checkpoint_sha256+'.checkpoint')/'manifest.json'
        original = path.read_bytes()
        try:
            path.write_bytes(b'declared checkpoint corruption')
            reply = ui._display_reply(self.partial_reply.session,self.partial_store,cancel)
            self.assertEqual(reply.checkpoint_facts['scaled_size']['status'],'checkpoint_unavailable')
            self.assertNotIn('scaled_size',reply.source_metadata)
            self.assertIn('seed.watershed',reply.source_metadata)
        finally:
            path.write_bytes(original)

    def test_actual_complete_and_partial_views_fit_and_capture_visual_evidence(self):
        directory = Path(__file__).resolve().parents[1]/'artifacts'/'calibrated-public-desktop-visual-v2'
        directory.mkdir(parents=True,exist_ok=True)
        evidence = []
        for name,dialog,key in (('complete',self.dialog,'correction.corrected_image'),
                                ('partial',self.partial_dialog,'seed.watershed.labels')):
            dialog.show()
            dialog.artifact_selector.setCurrentIndex(dialog.artifact_selector.findText(key))
            for width,height in ((1440,900),(1050,650)):
                dialog.resize(width,height)
                self.app.processEvents()
                self.assertEqual((dialog.width(),dialog.height()),(width,height))
                pixmap = dialog.preview.pixmap()
                rectangle = dialog.preview.contentsRect()
                self.assertFalse(pixmap.isNull())
                self.assertLessEqual(pixmap.width(),rectangle.width())
                self.assertLessEqual(pixmap.height(),rectangle.height())
                image = directory/f'{name}-{width}x{height}.png'
                self.assertTrue(dialog.grab().save(str(image)))
                evidence.append({'view':name,'size':[width,height],'path':str(image),
                    'artifact':key,'status':dialog.status_label.text()})
        (directory/'evidence.json').write_text(json.dumps({'views':evidence,
            'actual_timer_ticks':self.timer_ticks,'partial_source_fault_injected':True},indent=2)+'\n',encoding='utf-8')


class CalibratedPublicDesktopContractsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='calibrated-public-controls-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root/'source.png'
        self.source.write_bytes(b'controlled immutable worker source')
        self.selected = spec()

    def _dialog(self,**changes):
        values = {'source_path':self.source,'store':self.root/'store',
            'card_parameters':_thaw_json(self.selected.card_parameters),
            'seed_roi_xyxy':self.selected.seed_roi_xyxy,'runner_factory':lambda:object()}|changes
        dialog = ui.CalibratedPublicReferenceCaptureDialog(self.selected.seed_recipe,**values)
        self.addCleanup(dialog.close)
        return dialog

    def test_card_seed_and_child_controls_change_frozen_calculation_selection(self):
        dialog = self._dialog()
        original = dialog.selected_spec()
        self.assertEqual(original,self.selected)
        dialog._card_editors['radius'].setText('21')
        dialog._card_editors['roi_bounds_xyxy'].setText('[1,0,460,620]')
        dialog.seed_roi_editor.setText('[601,260,729,356]')
        key = next(key for key in dialog._editors if key[1]=='distance')
        dialog._editors[key].setText('13')
        selected = dialog.selected_spec()
        self.assertNotEqual(selected.sha256,original.sha256)
        self.assertEqual(selected.card_parameters['radius'],21)
        self.assertEqual(selected.card_parameters['roi_bounds_xyxy'],(1,0,460,620))
        self.assertEqual(selected.seed_roi_xyxy,(601,260,729,356))
        self.assertEqual(selected.seed_recipe.parameters_for_runner(key[0])['distance'],13)
        self.assertEqual(dialog._recipe,self.selected.seed_recipe)

    def test_missing_roi_and_malformed_controls_fail_before_worker_launch(self):
        for changes in ({'seed_roi_xyxy':None},{}):
            dialog = self._dialog(**changes)
            if not changes:
                dialog._card_editors['block_size'].setText('4')
            with patch.object(ui.QThreadPool,'globalInstance') as pool:
                dialog.start_capture()
            pool.assert_not_called()
            self.assertIn('Cannot start',dialog.status_label.text())

    def test_frozen_request_and_stale_selection_completion_are_discarded(self):
        dialog,entered,release = self._dialog(),threading.Event(),threading.Event()
        observed = []
        def controlled(selected,source,store,runner,**kwargs):
            observed.append(selected)
            entered.set()
            if not release.wait(5):
                raise RuntimeError('Test release expired.')
            raise RuntimeError('Stale declared failure.')
        with patch.object(ui,'capture_calibrated_reference_path',controlled):
            dialog.start_capture()
            self.assertTrue(entered.wait(2))
            self.assertFalse(dialog.parameter_scroll.isEnabled())
            self.assertFalse(dialog.run_button.isEnabled())
            dialog._card_editors['radius'].setText('22')
            self.assertTrue(dialog._cancel.is_set())
            self.assertEqual(observed[0],self.selected)
            self.assertNotEqual(dialog.selected_spec().sha256,observed[0].sha256)
            release.set()
            wait_for_worker(self.app,dialog,10)
        self.assertIn('discarded',dialog.status_label.text())
        self.assertIsNone(dialog._reply)
        self.assertTrue(dialog.run_button.isEnabled())

    def test_source_replacement_during_worker_discards_failure_and_previews(self):
        dialog,entered,release = self._dialog(),threading.Event(),threading.Event()
        def controlled(*args,**kwargs):
            entered.set()
            release.wait(5)
            raise ValueError('Old source failure must not attach.')
        with patch.object(ui,'capture_calibrated_reference_path',controlled):
            dialog.start_capture()
            self.assertTrue(entered.wait(2))
            self.source.write_bytes(b'replacement with another byte identity')
            release.set()
            wait_for_worker(self.app,dialog,10)
        self.assertIn('Source image changed',dialog.status_label.text())
        self.assertIsNone(dialog._reply)
        self.assertEqual(dialog.facts.toPlainText(),'')
        self.assertEqual(dialog.artifact_selector.count(),0)

    def test_stop_and_close_drain_same_worker_before_rerun(self):
        for close in (False,True):
            dialog,entered,release = self._dialog(),threading.Event(),threading.Event()
            def controlled(*args,**kwargs):
                entered.set()
                release.wait(5)
                raise CheckpointCancelled('Declared worker drain.')
            dialog.show()
            self.app.processEvents()
            with patch.object(ui,'capture_calibrated_reference_path',controlled):
                dialog.start_capture()
                self.assertTrue(entered.wait(2))
                worker = dialog._worker
                if close:
                    dialog.close()
                    self.assertTrue(dialog.isVisible())
                else:
                    dialog.stop_capture()
                self.assertTrue(dialog._cancel.is_set())
                dialog.start_capture()
                self.assertIs(dialog._worker,worker)
                release.set()
                wait_for_worker(self.app,dialog,10)
            self.assertIsNone(dialog._reply)
            if close:
                self.assertFalse(dialog.isVisible())
            else:
                self.assertIn('discarded',dialog.status_label.text())

    def test_saved_open_failure_is_async_and_does_not_restore_controls(self):
        dialog,entered,release = self._dialog(),threading.Event(),threading.Event()
        selected = dialog.selected_spec()
        def controlled(*args,**kwargs):
            entered.set()
            release.wait(5)
            raise ValueError('Declared saved revision mismatch.')
        with patch.object(ui,'open_calibrated_reference_capture',controlled):
            dialog.open_saved(self.root/'saved.json')
            self.assertTrue(entered.wait(2))
            self.assertFalse(dialog.open_button.isEnabled())
            release.set()
            wait_for_worker(self.app,dialog,10)
        self.assertIn('saved revision mismatch',dialog.status_label.text())
        self.assertEqual(dialog.selected_spec(),selected)
        self.assertEqual(dialog._source_path,self.source)
        self.assertIsNone(dialog._reply)

    def test_request_binding_rejects_wrong_spec_or_source_before_checkpoint_reads(self):
        from types import SimpleNamespace
        fake = SimpleNamespace(spec=self.selected,verified=None,outcome=SimpleNamespace(record={'source_sha256':'f'*64}))
        with patch.object(ui,'_partial_checkpoints',side_effect=AssertionError('checkpoint read')):
            with self.assertRaisesRegex(ValueError,'selection'):
                ui._display_reply(fake,self.root,threading.Event(),expected_spec=spec('mean'))
            with self.assertRaisesRegex(ValueError,'byte identity'):
                ui._display_reply(fake,self.root,threading.Event(),expected_source_sha256='a'*64)

    def test_graph_launch_eligibility_and_history_are_preserved(self):
        from seedvision.ui.reference_graph_dialog import ReferenceGraphDialog
        dialog = ReferenceGraphDialog()
        self.addCleanup(dialog.close)
        for recipe_id,enabled in (('plantcv_seed_binary',True),('plantcv_seed_mean',True),
                                  ('plantcv_seed_otsu',True),('plantcv_color_card_correct_color',False)):
            dialog.recipe_selector.setCurrentIndex(dialog.recipe_selector.findData(recipe_id))
            self.app.processEvents()
            self.assertEqual(dialog._calibrated_public_capture_button.isEnabled(),enabled)
        dialog.recipe_selector.setCurrentIndex(dialog.recipe_selector.findData('plantcv_seed_binary'))
        before,native = dialog.reference_graphs,dialog._native_recipes
        with patch.object(ui.CalibratedPublicReferenceCaptureDialog,'exec',return_value=0):
            dialog._open_calibrated_public_capture()
        self.assertEqual(dialog.reference_graphs,before)
        self.assertEqual(dialog._native_recipes,native)


if __name__ == '__main__':
    unittest.main()
