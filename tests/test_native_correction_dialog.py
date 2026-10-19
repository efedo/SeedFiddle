
from tests.catalogue_fixtures import CatalogueTestCase
import dataclasses
import hashlib
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch
from seedvision.benchmarking.native_correction_recipe import (
    NativeCorrectionCancelled, NativeCorrectionError, RECIPE_ID, STAGE_ID)
from seedvision.benchmarking.recipe_catalogue import build_recipe_catalogue
from seedvision.ui.native_correction_dialog import (
    MAX_PREPARED_CUDA_BYTES, NativeCorrectionDialog, NativeCorrectionInputs,
    prepare_native_correction_inputs)

_RECIPE = build_recipe_catalogue().recipe(RECIPE_ID)
_PORTS = ("target_rgb", "target_mask", "source_rgb", "source_mask")


def _fixture():
    rng = np.random.default_rng(309)
    source = rng.integers(20,235,(17,25,3),dtype=np.uint8)
    target = np.clip(source.astype(np.int16)+[9,-7,5],0,255).astype(np.uint8)
    source_mask, target_mask = np.zeros((17,25),np.uint8), np.zeros((17,25),np.uint8)
    for index in range(24):
        row, column = divmod(index,6)
        ys, xs = slice(1+row*4,4+row*4), slice(1+column*4,4+column*4)
        source[ys,xs] = rng.integers(20,235,3,dtype=np.uint8)
        target[ys,xs] = np.clip(source[ys,xs].astype(np.int16)+[9,-7,5],0,255)
        source_mask[ys,xs] = target_mask[ys,xs] = (index+1)*10
    return target, target_mask, source, source_mask


def _write_inputs(directory):
    paths = {}
    for name, value in zip(_PORTS,_fixture()):
        path = directory/(name+".png")
        image = cv2.cvtColor(value,cv2.COLOR_RGB2BGR) if name.endswith("_rgb") else value
        if not cv2.imwrite(str(path),image):
            raise AssertionError(f"Could not create PNG test input {path}")
        paths[name] = path
    return paths


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for correction preparation")
class NativeCorrectionPreparationTests(CatalogueTestCase):
    def test_four_png_inputs_decode_to_explicit_cuda_ports_and_exact_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = _write_inputs(Path(temporary))
            prepared = prepare_native_correction_inputs(paths)
            self.assertIsInstance(prepared,NativeCorrectionInputs)
            self.assertEqual(set(prepared.tensors),set(_PORTS))
            for name, expected in zip(_PORTS,_fixture()):
                tensor = prepared.tensors[name]
                self.assertEqual(tensor.device,prepared.device)
                self.assertEqual(tensor.device.type,"cuda")
                self.assertEqual(tensor.dtype,torch.uint8)
                np.testing.assert_array_equal(tensor.cpu().numpy(),expected)
                self.assertEqual(prepared.file_sha256s[name],hashlib.sha256(paths[name].read_bytes()).hexdigest())
                self.assertIn("sha256="+prepared.file_sha256s[name],prepared.preparation[name])
            self.assertEqual(prepared.coordinate_frames["target_rgb"],prepared.coordinate_frames["target_mask"])
            self.assertEqual(prepared.coordinate_frames["source_rgb"],prepared.coordinate_frames["source_mask"])
            self.assertIn("BGR on CPU",prepared.preparation["target_rgb"])
            self.assertIn("BGR-to-RGB",prepared.preparation["source_rgb"])
            self.assertIn("IMREAD_UNCHANGED",prepared.preparation["target_mask"])
            self.assertLessEqual(prepared.cuda_bytes,MAX_PREPARED_CUDA_BYTES)
            self.assertIs(prepare_native_correction_inputs(paths,previous=prepared),prepared)





class NativeCorrectionDialogTests(CatalogueTestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM","offscreen")
        from PySide6.QtWidgets import QApplication
        cls.application = QApplication.instance() or QApplication([])

    def wait_until(self,predicate,timeout=30):
        deadline = time.monotonic()+timeout
        while not predicate() and time.monotonic() < deadline:
            self.application.processEvents()
            time.sleep(.01)
        self.application.processEvents()
        self.assertTrue(predicate(),"Correction Qt worker did not reach the expected state.")

    def make_dialog(self,paths):
        dialog = NativeCorrectionDialog(_RECIPE)
        for name in _PORTS:
            dialog.set_input_path(name,paths[name])
        self.assertTrue(dialog.run_button.isEnabled())
        return dialog



    def test_close_cancels_worker_and_waits_for_it_to_drain(self):
        entered, release = threading.Event(), threading.Event()
        def controlled_prepare(paths,*,previous,device,cancel,**limits):
            entered.set()
            if not release.wait(10):
                raise AssertionError("Test did not release worker")
            if cancel():
                raise NativeCorrectionCancelled("Stopped")
            raise AssertionError("Closing did not request cancellation")
        with tempfile.TemporaryDirectory() as temporary:
            dialog = self.make_dialog(_write_inputs(Path(temporary)))
            dialog.show()
            try:
                with patch("seedvision.benchmarking.native_correction_inputs.prepare_native_correction_inputs",side_effect=controlled_prepare):
                    dialog.start_run()
                    self.wait_until(entered.is_set)
                    dialog.close()
                    self.assertTrue(dialog.isVisible())
                    self.assertTrue(dialog._cancel.is_set())
                    release.set()
                    self.wait_until(lambda:dialog._worker is None)
                    self.assertFalse(dialog.isVisible())
                    self.assertIsNone(dialog.result)
            finally:
                release.set()
                self.wait_until(lambda:dialog._worker is None)
                dialog.reject()
                dialog.deleteLater()





if __name__ == "__main__":
    unittest.main()
