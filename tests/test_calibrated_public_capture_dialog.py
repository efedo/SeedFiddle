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



    def test_request_binding_rejects_wrong_spec_or_source_before_checkpoint_reads(self):
        from types import SimpleNamespace
        fake = SimpleNamespace(spec=self.selected,verified=None,outcome=SimpleNamespace(record={'source_sha256':'f'*64}))
        with patch.object(ui,'_partial_checkpoints',side_effect=AssertionError('checkpoint read')):
            with self.assertRaisesRegex(ValueError,'selection'):
                ui._display_reply(fake,self.root,threading.Event(),expected_spec=spec('mean'))
            with self.assertRaisesRegex(ValueError,'byte identity'):
                ui._display_reply(fake,self.root,threading.Event(),expected_source_sha256='a'*64)



if __name__ == '__main__':
    unittest.main()
