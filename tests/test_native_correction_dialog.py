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
class NativeCorrectionPreparationTests(unittest.TestCase):
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

    def test_changed_file_tensor_version_or_requested_device_forces_fresh_preparation(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = _write_inputs(Path(temporary))
            first = prepare_native_correction_inputs(paths)
            first.tensors["source_rgb"][0,0,0] ^= 1
            second = prepare_native_correction_inputs(paths,previous=first)
            self.assertIsNot(second,first)
            self.assertNotEqual(second.tensor_signatures,first.tensor_signatures)
            other_device = dataclasses.replace(second,device=torch.device("cuda",torch.cuda.current_device()+1))
            third = prepare_native_correction_inputs(paths,previous=other_device)
            self.assertIsNot(third,other_device)
            self.assertEqual(third.device,torch.device("cuda",torch.cuda.current_device()))
            mask = cv2.imread(str(paths["source_mask"]),cv2.IMREAD_UNCHANGED)
            mask[0,0] = 10
            self.assertTrue(cv2.imwrite(str(paths["source_mask"]),mask))
            fourth = prepare_native_correction_inputs(paths,previous=third)
            self.assertIsNot(fourth,third)
            self.assertNotEqual(fourth.file_sha256s["source_mask"],third.file_sha256s["source_mask"])
            with self.assertRaisesRegex(NativeCorrectionError,"Requested CUDA device is unavailable"):
                prepare_native_correction_inputs(paths,previous=fourth,device=torch.device("cuda",torch.cuda.device_count()))

    def test_cpu_shape_and_pixel_preflight_happens_before_cuda_allocation(self):
        for kind in ("shape","pixels","height"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                paths = _write_inputs(Path(temporary))
                if kind == "shape":
                    self.assertTrue(cv2.imwrite(str(paths["target_mask"]),np.zeros((2,2),np.uint8)))
                    message = "shape must exactly match"
                elif kind == "height":
                    target = _fixture()[0]
                    taller = np.concatenate((target,target[:1]),axis=0)
                    self.assertTrue(cv2.imwrite(str(paths["target_rgb"]),cv2.cvtColor(taller,cv2.COLOR_RGB2BGR)))
                    self.assertTrue(cv2.imwrite(str(paths["target_mask"]),np.zeros((18,25),np.uint8)))
                    message = "equal target/source image heights"
                else:
                    message = "exceeds the 64-million-pixel"
                limit = 100 if kind == "pixels" else 64_000_000
                with patch("seedvision.ui.native_correction_dialog.MAX_COMBINED_PIXELS",limit), \
                        patch("seedvision.ui.native_correction_dialog.torch.as_tensor") as allocate:
                    with self.assertRaisesRegex(NativeCorrectionError,message):
                        prepare_native_correction_inputs(paths)
                    allocate.assert_not_called()

    def test_bad_mask_type_and_file_change_during_decode_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = _write_inputs(Path(temporary))
            self.assertTrue(cv2.imwrite(str(paths["target_mask"]),np.zeros((17,25,3),np.uint8)))
            with patch("seedvision.ui.native_correction_dialog.torch.as_tensor") as allocate:
                with self.assertRaisesRegex(NativeCorrectionError,"single-channel uint8 mask"):
                    prepare_native_correction_inputs(paths)
                allocate.assert_not_called()
        with tempfile.TemporaryDirectory() as temporary:
            paths = _write_inputs(Path(temporary))
            original, changed = cv2.imdecode, False
            def mutate(raw,flags):
                nonlocal changed
                value = original(raw,flags)
                if flags == cv2.IMREAD_UNCHANGED and not changed:
                    changed = True
                    mask = cv2.imread(str(paths['target_mask']), cv2.IMREAD_UNCHANGED)
                    mask[0,0] = 10
                    self.assertTrue(cv2.imwrite(str(paths['target_mask']),mask))
                return value
            with patch("seedvision.ui.native_correction_dialog.cv2.imdecode",side_effect=mutate):
                with self.assertRaisesRegex(NativeCorrectionError,"changed while OpenCV was decoding"):
                    prepare_native_correction_inputs(paths)


class NativeCorrectionDialogTests(unittest.TestCase):
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

    @unittest.skipUnless(torch.cuda.is_available(),"CUDA is required for native correction")
    def test_async_run_reuses_prepared_tensors_then_changed_mask_invalidates_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = _write_inputs(Path(temporary))
            dialog = self.make_dialog(paths)
            dialog.show()
            try:
                dialog.start_run()
                self.assertFalse(dialog.export_button.isEnabled())
                self.wait_until(lambda:dialog._worker is None)
                self.assertIsNotNone(dialog.result,dialog.status_label.text())
                self.assertEqual(dialog.result.status,"executed")
                first_prepared = dialog._prepared
                first_hash = first_prepared.file_sha256s["source_mask"]
                first_image = dialog.result.node(STAGE_ID).outputs["corrected_image"]
                self.assertEqual(dialog.cache.last_computed_nodes,(STAGE_ID,))
                self.assertFalse(dialog.preview.pixmap().isNull())
                dialog.start_run()
                self.wait_until(lambda:dialog._worker is None)
                self.assertIsNotNone(dialog.result,dialog.status_label.text())
                self.assertIs(dialog._prepared,first_prepared)
                self.assertEqual(dialog.cache.last_reused_nodes,(STAGE_ID,))
                self.assertIs(dialog.result.node(STAGE_ID).outputs["corrected_image"],first_image)
                changed = cv2.imread(str(paths["source_mask"]),cv2.IMREAD_UNCHANGED)
                changed[0,0] = 10
                self.assertTrue(cv2.imwrite(str(paths["source_mask"]),changed))
                dialog.start_run()
                self.wait_until(lambda:dialog._worker is None)
                self.assertIsNotNone(dialog.result,dialog.status_label.text())
                self.assertIsNot(dialog._prepared,first_prepared)
                self.assertNotEqual(dialog._prepared.file_sha256s["source_mask"],first_hash)
                self.assertEqual(dialog.cache.last_computed_nodes,(STAGE_ID,))
                self.assertEqual(dialog.cache.last_reused_nodes,())
                self.assertIn(dialog._prepared.file_sha256s["source_mask"],dialog.result.preparation["source_mask"])
            finally:
                dialog.reject()
                self.wait_until(lambda:dialog._worker is None)
                dialog.deleteLater()

    @unittest.skipUnless(torch.cuda.is_available(),"CUDA is required for native correction")
    def test_wrong_mask_shape_yields_no_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = _write_inputs(Path(temporary))
            self.assertTrue(cv2.imwrite(str(paths["target_mask"]),np.zeros((2,2),np.uint8)))
            dialog = self.make_dialog(paths)
            dialog.show()
            try:
                dialog.start_run()
                self.wait_until(lambda:dialog._worker is None)
                self.assertIsNone(dialog.result)
                self.assertIn("shape must exactly match",dialog.status_label.text())
                self.assertNotIn(f"reference.recipe:{RECIPE_ID}:{STAGE_ID}",dialog.cache.values)
            finally:
                dialog.reject()
                dialog.deleteLater()

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

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_async_export_records_four_inputs_and_locks_all_file_pickers(self):
        from seedvision.export.native_results import export_native_recipe_result, load_native_recipe_result
        entered, release = threading.Event(), threading.Event()
        def controlled_export(*args, **kwargs):
            entered.set()
            if not release.wait(10):
                raise AssertionError('Test did not release export')
            return export_native_recipe_result(*args, **kwargs)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dialog = self.make_dialog(_write_inputs(root))
            dialog.show()
            try:
                dialog.start_run()
                self.wait_until(lambda: dialog._worker is None)
                self.assertIsNotNone(dialog.result, dialog.status_label.text())
                with patch('seedvision.ui.native_recipe_dialog.QFileDialog.getExistingDirectory', return_value=str(root)), \
                        patch('seedvision.ui.native_recipe_dialog.export_native_recipe_result', side_effect=controlled_export):
                    dialog._export_run()
                    self.wait_until(entered.is_set)
                    self.assertTrue(dialog.stop_button.isEnabled())
                    for button in (*dialog._input_buttons.values(), dialog.run_button,
                                   dialog.save_button, dialog.export_button):
                        self.assertFalse(button.isEnabled())
                    with self.assertRaises(RuntimeError):
                        dialog.set_input_path('source_rgb', root/'other.png')
                    release.set()
                    self.wait_until(lambda: dialog._worker is None)
                reports = list(root.glob('*/native-results.json'))
                self.assertEqual(len(reports), 1, dialog.status_label.text())
                report, _ = load_native_recipe_result(reports[0].parent)
                self.assertEqual(set(report['correction_inputs']), set(_PORTS))
                self.assertEqual(report['recipe_identity'], dialog.result.recipe_identity.to_dict())
                self.assertTrue(all(button.isEnabled() for button in dialog._input_buttons.values()))
                self.assertTrue(dialog.export_button.isEnabled())
            finally:
                release.set()
                self.wait_until(lambda: dialog._worker is None)
                dialog.reject()
                dialog.deleteLater()

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_close_cancels_and_drains_correction_export_worker(self):
        from seedvision.benchmarking.native_recipe import NativeRecipeCancelled
        entered, release = threading.Event(), threading.Event()
        def controlled_export(*args, cancel, **kwargs):
            entered.set()
            if not release.wait(10):
                raise AssertionError('Test did not release export')
            if cancel():
                raise NativeRecipeCancelled('Stopped')
            raise AssertionError('Close did not cancel export')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dialog = self.make_dialog(_write_inputs(root))
            dialog.show()
            try:
                dialog.start_run()
                self.wait_until(lambda: dialog._worker is None)
                self.assertIsNotNone(dialog.result, dialog.status_label.text())
                with patch('seedvision.ui.native_recipe_dialog.QFileDialog.getExistingDirectory', return_value=str(root)), \
                        patch('seedvision.ui.native_recipe_dialog.export_native_recipe_result', side_effect=controlled_export):
                    dialog._export_run()
                    self.wait_until(entered.is_set)
                    dialog.close()
                    self.assertTrue(dialog.isVisible())
                    self.assertTrue(dialog._cancel.is_set())
                    release.set()
                    self.wait_until(lambda: dialog._worker is None)
                    self.assertFalse(dialog.isVisible())
                self.assertFalse(list(root.glob('*/native-results.json')))
            finally:
                release.set()
                self.wait_until(lambda: dialog._worker is None)
                dialog.reject()
                dialog.deleteLater()

    def test_selected_recipe_saves_and_launch_preserves_historical_attachment(self):
        from seedvision.benchmarking.recipes import RecipeSpec
        from seedvision.ui.reference_graph_dialog import ReferenceGraphDialog
        reference = ReferenceGraphDialog()
        try:
            index = reference.recipe_selector.findData(RECIPE_ID)
            self.assertGreaterEqual(index,0)
            reference.recipe_selector.setCurrentIndex(index)
            history = reference.reference_graphs
            self.assertTrue(reference._native_run_button.isEnabled())
            with patch("seedvision.ui.native_correction_dialog.NativeCorrectionDialog.exec",return_value=0) as run:
                reference._open_native_recipe()
            run.assert_called_once()
            self.assertEqual(reference.reference_graphs,history)
            with tempfile.TemporaryDirectory() as temporary:
                paths = _write_inputs(Path(temporary))
                dialog = self.make_dialog(paths)
                try:
                    output = Path(temporary)/"correction.seedfiddle-recipe.json"
                    with patch("seedvision.ui.native_recipe_dialog.QFileDialog.getSaveFileName",return_value=(str(output),"")):
                        dialog._save_recipe()
                    payload = output.read_bytes()
                    from seedvision.benchmarking.native_correction_recipe import build_native_correction_recipe
                    saved = RecipeSpec.from_json(payload)
                    self.assertEqual(saved, dialog._selected_recipe_with_layout())
                    self.assertEqual(saved.identity.analytical_dict(),
                                     build_native_correction_recipe(_RECIPE).identity.analytical_dict())
                    for path in paths.values():
                        self.assertNotIn(str(path).encode(),payload)
                    self.assertEqual(reference.reference_graphs,history)
                finally:
                    dialog.reject()
                    dialog.deleteLater()
        finally:
            reference.reject()
            reference.deleteLater()


if __name__ == "__main__":
    unittest.main()
