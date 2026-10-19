"""Exact-image correction Keep, worker handoff and reopened project pins."""
from dataclasses import replace
import os
from pathlib import Path
import shutil
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import numpy as np

from seedvision.benchmarking.native_correction_recipe import STAGE_ID
from seedvision.benchmarking.native_recipe import NativeRecipeCancelled
from seedvision.persistence import correction_inputs as assets
from seedvision.persistence.correction_session import (
    CorrectionKeepRequest, capture_correction_context, capture_correction_keep,
    validate_correction_keep_result,
)
from tests import test_project_correction_inputs as fixtures


class _Fixture:
    def setUp(self):
        # Import the fixture module, not its TestCase into discovery's namespace.
        self.fixture = fixtures.ProjectCorrectionInputTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.context = capture_correction_context(self.fixture.workspace, self.fixture.source)
        self.request = CorrectionKeepRequest(self.context, self.fixture.recipe, self.fixture.binding, 1)


class CorrectionSessionTests(_Fixture, unittest.TestCase):
    def test_context_uses_exact_locator_for_identical_bytes_and_preserves_loaded_record(self):
        f = self.fixture
        second = f.workspace/'second.png'
        shutil.copyfile(f.source, second)
        other = capture_correction_context(f.workspace, second)
        self.assertEqual(other.image.source.sha256, self.context.image.source.sha256)
        self.assertNotEqual(other.image.identifier, self.context.image.identifier)
        self.assertEqual(other.image.source.path, 'second.png')
        loaded = capture_correction_context(f.workspace, f.source, original_records=f.document.images)
        self.assertIs(loaded.image, f.document.images[0])
        f.source.write_bytes(f.source.read_bytes()[:-1]+b'x')
        with self.assertRaises(ValueError):
            capture_correction_context(f.workspace, f.source, original_records=f.document.images)




    def test_cancellation_and_wrong_capture_result_do_not_publish_master(self):
        with self.assertRaises(NativeRecipeCancelled):
            capture_correction_keep(self.request, cancel=lambda: True)
        captured = capture_correction_keep(self.request)
        bad = replace(captured, inputs=tuple(replace(item, decode=item.decode+' changed')
                                             for item in captured.inputs))
        with self.assertRaisesRegex(ValueError, 'frozen selection'):
            validate_correction_keep_result(self.request, bad)
        self.assertFalse(list(self.fixture.workspace.glob('*.seedfiddle-project.json')))


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for actual correction runs')
class CorrectionProjectDialogTests(_Fixture, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        from PySide6.QtWidgets import QApplication
        cls.application = QApplication.instance() or QApplication([])

    def wait_until(self, predicate, timeout=40):
        end = time.monotonic()+timeout
        while not predicate() and time.monotonic() < end:
            self.application.processEvents()
            time.sleep(.01)
        self.application.processEvents()
        self.assertTrue(predicate(), 'Correction worker did not finish.')

    def dialog(self, *, context=True, binding=None, callback=None, run=False):
        from seedvision.ui.native_correction_dialog import NativeCorrectionDialog
        dialog = NativeCorrectionDialog(self.fixture.recipe,
            project_context=self.context if context else None, project_binding=binding,
            keep_in_project=callback or (lambda request, cancel, current:
                capture_correction_keep(request, cancel=cancel.is_set)))
        if binding is None:
            for role, path in (self.fixture.paths | {'source_rgb': self.fixture.source}).items():
                dialog.set_input_path(role, path)
        dialog.show()
        self.addCleanup(self.dispose_dialog, dialog)
        if run:
            dialog.start_run()
            self.wait_until(lambda: dialog._worker is None)
            self.assertIsNotNone(dialog.result, dialog.status_label.text())
        return dialog

    def dispose_dialog(self, dialog):
        dialog.reject()
        self.wait_until(lambda: dialog._worker is None and not dialog._keeping)
        dialog.deleteLater()
        self.application.processEvents()

    def window(self):
        from tests import test_project_analysis_ui as ui
        f = self.fixture
        (f.workspace/'config').mkdir(exist_ok=True)
        shutil.copyfile(Path(__file__).parents[1]/'config'/'traits.json', f.workspace/'config'/'traits.json')
        ui.ProjectAnalysisUiTests.application = self.application
        window = ui.ProjectAnalysisUiTests._window(f.workspace)
        self.addCleanup(ui.ProjectAnalysisUiTests._dispose, window)
        window._add_image(f.source, open_now=True, mark_project_dirty=False)
        return window










    def test_window_preserves_distinct_same_byte_image_binding_and_history(self):
        window = self.window()
        first = capture_correction_keep(self.request)
        window._project_native_recipes = (self.fixture.recipe,)
        window._project_correction_bindings = (first,)
        window._project_original_image_records = {self.context.image.identifier: self.context.image}
        window._project_original_image_order = (self.context.image.identifier,)
        second = self.fixture.workspace/'second.png'
        shutil.copyfile(self.fixture.source, second)
        window._add_image(second, open_now=True, mark_project_dirty=False)
        context = window._capture_correction_project_context()
        binding = replace(self.fixture.binding, image_id=context.image.identifier)
        request = CorrectionKeepRequest(context, self.fixture.recipe, binding, 2)
        history = window._project_reference_graphs
        captured = window._keep_project_correction(request, threading.Event(), lambda: True)
        self.assertEqual(window._project_correction_bindings, (first, captured))
        self.assertNotEqual(first.image_id, captured.image_id)
        self.assertEqual(window._project_reference_graphs, history)
