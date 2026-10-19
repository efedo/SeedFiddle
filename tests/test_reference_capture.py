from __future__ import annotations

from tests.functional_units import functional_units

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
from tests.catalogue_fixtures import CatalogueTestCase
from unittest.mock import patch

import cv2
import numpy as np

from seedvision.benchmarking.checkpoints import (
    CheckpointCancelled, CheckpointLimits, RasterRole, _canonical_json, _thaw_json,
    load_checkpoint, publish_checkpoint,
)
from seedvision.benchmarking.reference_capture_session import (
    capture_reference_path, open_reference_capture, reference_measurement,
)
from seedvision.benchmarking.reference_runner import PlantCVReferenceRunner
from seedvision.benchmarking.reference_seed_chain import ReferenceRasterInput
from seedvision.benchmarking.reference_seed_chain_reader import read_reference_seed_chain
from tests.test_reference_preprocessing import HAS_REFERENCE_ENV, REFERENCE_PYTHON, _gray_fixture, _selected_recipe


@unittest.skipUnless(HAS_REFERENCE_ENV, "Pinned public PlantCV environment is required.")
class ReferenceCaptureContractsTests(CatalogueTestCase):
    """Actual selected binary chain; mutations exercise stored contracts only."""

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="public-capture-contracts-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.source = cls.root / "source.png"
        gray = _gray_fixture((48, 72))
        bgr = np.stack((np.minimum(gray, 173), np.maximum(gray, 61), gray), axis=-1).astype(np.uint8)
        bgr[0, 0] = (17, 31, 63)
        success, encoded = cv2.imencode(".png", bgr)
        if not success:
            raise AssertionError("Test image encoding failed.")
        cls.source.write_bytes(encoded.tobytes())
        cls.store = cls.root / "capture"
        cls.recipe = _selected_recipe("plantcv_seed_binary")
        cls.results, cls.operations = {}, []
        actual = PlantCVReferenceRunner.run

        def observe(runner, operation, arrays, parameters=None, **kwargs):
            result = actual(runner, operation, arrays, parameters, **kwargs)
            cls.results[operation] = result
            cls.operations.append(operation)
            return result

        with patch.object(PlantCVReferenceRunner, "run", observe):
            cls.session = capture_reference_path(cls.recipe, cls.source, cls.store, PlantCVReferenceRunner(REFERENCE_PYTHON))
        cls.manifest = cls.session.outcome.manifest_path
        if cls.session.verified is None:
            raise AssertionError([call.reason for call in cls.session.outcome.calls])

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def _arguments(self, digest=None, store=None):
        record = _thaw_json(self.session.verified.record)
        bindings = {}
        for name in ("gray", "rgb"):
            checkpoint = self.session.verified.checkpoints[f"input.{name}"]
            bindings[name] = ReferenceRasterInput(self.store / "inputs", checkpoint.checkpoint_sha256,
                checkpoint.identity, checkpoint.measurement, name, checkpoint.raster_roles[name])
        return {"output_root": store or self.store, "manifest_sha256": digest or self.session.verified.manifest_sha256,
                **bindings, "recipe": self.recipe, "expected_source_sha256": record["source_sha256"],
                "expected_coordinate_frame": bindings["rgb"].identity.coordinate_frame,
                "output_measurement": reference_measurement("public_reference_chain_output")}

    def _changed_manifest(self, mutate):
        record = _thaw_json(self.session.verified.record)
        mutate(record)
        raw = _canonical_json(record, name="test changed manifest")
        digest = sha256(raw).hexdigest()
        (self.store / "chains" / f"{digest}.json").write_bytes(raw)
        return digest

    def _replace_checkpoint(self, record, key, *, metadata_change=None, array_change=None, roles_change=None, execution_change=None):
        if key.startswith("preprocessing."):
            index = int(key.split(".")[1])
            entry = record["preprocessing"][index]["run_record"]
        else:
            entry = record["source_calls"][("watershed", "size", "color").index(key)]
        checkpoint = self.session.verified.checkpoints[key]
        metadata, arrays, roles = _thaw_json(checkpoint.metadata), dict(checkpoint.arrays), dict(checkpoint.raster_roles)
        if metadata_change:
            metadata_change(metadata)
        if array_change:
            array_change(arrays)
        if roles_change:
            roles_change(roles)
        execution = execution_change(checkpoint.execution) if execution_change else checkpoint.execution
        reference = publish_checkpoint(self.store, checkpoint.identity, arrays, execution=execution,
            measurement=checkpoint.measurement, metadata=metadata, raster_roles=roles, allow_cpu_transfer=True)
        entry["checkpoint_sha256"] = reference.checkpoint_sha256





    def test_manifest_revision_and_strict_json_caps_reject(self):
        (self.store / "chains" / f"{'a' * 64}.json").write_bytes(self.manifest.read_bytes())
        with self.assertRaises(ValueError):
            read_reference_seed_chain(**self._arguments("a" * 64))
        for raw in (b'{"schema":1,"schema":2}', b'{"schema":NaN}', b'{"schema":1e9999}'):
            digest = sha256(raw).hexdigest()
            (self.store / "chains" / f"{digest}.json").write_bytes(raw)
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                read_reference_seed_chain(**self._arguments(digest))
        with self.assertRaises(ValueError):
            read_reference_seed_chain(**self._arguments(), limits=replace(CheckpointLimits(), max_metadata_bytes=100))





    def _fake(self):
        # A plain function installed on the runner class receives its runner
        # instance. A bound TestCase method would shift all call arguments.
        def run(runner, operation, arrays, parameters=None, **kwargs):
            result = self.results[operation]
            return replace(result, arrays={name: a.copy() for name, a in result.arrays.items()},
                           effective_parameters=deepcopy(dict(result.effective_parameters)),
                           identity=deepcopy(dict(result.identity)))
        return run








@functional_units
class PublicCaptureDesktopTests(CatalogueTestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="public-capture-qt-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "image.png"
        self.source.write_bytes(b"controlled worker input")

    def _dialog(self):
        from seedvision.ui.public_reference_capture_dialog import PublicReferenceCaptureDialog
        dialog = PublicReferenceCaptureDialog(_selected_recipe("plantcv_seed_binary"), source_path=self.source, store=self.root / "store", runner_factory=lambda: object())
        self.addCleanup(dialog.close)
        return dialog

    def _drain(self, dialog):
        deadline = time.monotonic() + 5
        while dialog._worker is not None and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.005)
        self.assertIsNone(dialog._worker, "Worker did not drain.")
        self.app.processEvents()

    def check_test_selection_controls_create_new_recipe_and_preserve_original(self):
        from PySide6.QtWidgets import QLineEdit
        dialog = self._dialog()
        original = dialog._recipe
        key = next(key for key in dialog._editors if key[1] == "distance")
        editor = dialog._editors[key]
        self.assertIsInstance(editor, QLineEdit)
        editor.setText("23")
        selected = dialog.selected_recipe()
        self.assertEqual(dialog._recipe, original)
        self.assertNotEqual(selected.identity.settings_sha256, original.identity.settings_sha256)
        self.assertEqual(selected.parameters_for_runner(key[0])["distance"], 23)
        self.assertGreater(dialog._generation, 0)

    def check_test_invalid_parameter_does_not_start_worker(self):
        dialog = self._dialog()
        key = next(key for key in dialog._editors if key[1] == "ksize")
        dialog._editors[key].setText("true")
        with patch("seedvision.ui.public_reference_capture_dialog.QThreadPool.globalInstance") as pool:
            dialog.start_capture()
        pool.assert_not_called()
        self.assertIn("Cannot start", dialog.status_label.text())

    def check_test_worker_freezes_selection_and_stale_completion_is_discarded(self):
        dialog, entered, release = self._dialog(), threading.Event(), threading.Event()
        observed = []
        def controlled(recipe, source, store, runner, **kwargs):
            observed.append((recipe, source, kwargs["expected_stamp"]))
            entered.set()
            if not release.wait(5):
                raise RuntimeError("Test release expired.")
            raise RuntimeError("Old worker failure must be discarded.")
        with patch("seedvision.ui.public_reference_capture_dialog.capture_reference_path", controlled):
            dialog.start_capture()
            self.assertTrue(entered.wait(2))
            self.assertFalse(dialog.run_button.isEnabled())
            self.assertFalse(dialog.parameter_scroll.isEnabled())
            selected = observed[0][0]
            key = next(key for key in dialog._editors if key[1] == "distance")
            dialog._editors[key].setText("19")
            self.assertNotEqual(dialog.selected_recipe().identity.settings_sha256, selected.identity.settings_sha256)
            self.assertTrue(dialog._cancel.is_set())
            release.set()
            self._drain(dialog)
        self.assertIn("discarded", dialog.status_label.text())
        self.assertTrue(dialog.run_button.isEnabled())
        self.assertIsNone(dialog._reply)

    def check_test_source_changed_during_worker_cannot_show_result(self):
        dialog, entered, release = self._dialog(), threading.Event(), threading.Event()
        def controlled(*args, **kwargs):
            entered.set()
            release.wait(5)
            return None  # Must be discarded before result interpretation.
        with patch("seedvision.ui.public_reference_capture_dialog.capture_reference_path", controlled), patch(
                "seedvision.ui.public_reference_capture_dialog._display_artifacts", return_value=()):
            dialog.start_capture()
            self.assertTrue(entered.wait(2))
            self.source.write_bytes(b"changed source contents")
            release.set()
            self._drain(dialog)
        self.assertIn("Source image changed", dialog.status_label.text())
        self.assertIsNone(dialog._reply)

    def check_test_close_requests_cancel_and_defers_until_worker_drains(self):
        dialog, entered, release = self._dialog(), threading.Event(), threading.Event()
        def controlled(*args, **kwargs):
            entered.set()
            release.wait(5)
            raise CheckpointCancelled("Test worker acknowledged cancellation.")
        dialog.show()
        self.app.processEvents()
        with patch("seedvision.ui.public_reference_capture_dialog.capture_reference_path", controlled):
            dialog.start_capture()
            self.assertTrue(entered.wait(2))
            dialog.close()
            self.assertTrue(dialog.isVisible())
            self.assertTrue(dialog._cancel.is_set())
            self.assertTrue(dialog._closing)
            release.set()
            self._drain(dialog)
        self.assertFalse(dialog.isVisible())

    def check_test_stop_discards_worker_failure_and_drains_before_rerun(self):
        dialog, entered, release = self._dialog(), threading.Event(), threading.Event()
        def controlled(*args, **kwargs):
            entered.set()
            release.wait(5)
            raise RuntimeError("Stale failure after Stop.")
        with patch("seedvision.ui.public_reference_capture_dialog.capture_reference_path", controlled):
            dialog.start_capture()
            self.assertTrue(entered.wait(2))
            worker = dialog._worker
            dialog.stop_capture()
            dialog.start_capture()
            self.assertIs(dialog._worker, worker)
            self.assertTrue(dialog._cancel.is_set())
            release.set()
            self._drain(dialog)
        self.assertIsNone(dialog._reply)
        self.assertIn("discarded", dialog.status_label.text())

    def check_test_stored_manifest_load_is_async_and_failure_is_visible(self):
        dialog, entered, release = self._dialog(), threading.Event(), threading.Event()
        def controlled(path, **kwargs):
            entered.set()
            release.wait(5)
            raise ValueError("Stored revision binding differs.")
        with patch("seedvision.ui.public_reference_capture_dialog.open_reference_capture", controlled):
            dialog.open_saved(self.root / "saved.json")
            self.assertTrue(entered.wait(2))
            self.assertFalse(dialog.open_button.isEnabled())
            self.app.processEvents()
            release.set()
            self._drain(dialog)
        self.assertIn("Stored revision binding differs", dialog.status_label.text())

    def check_test_preview_is_bounded_and_BGR_capture_displays_correct_RGB(self):
        from seedvision.ui.public_reference_capture_dialog import preview_rgb
        bgr = np.empty((401, 619, 3), dtype=np.uint8)
        bgr[:] = (17, 31, 63)
        raw = sha256(bgr.tobytes()).hexdigest()
        display, description = preview_rgb(bgr, "diagnostic_bgr", max_pixels=4096)
        self.assertLessEqual(display.shape[0] * display.shape[1], 4096)
        np.testing.assert_array_equal(display[0, 0], [63, 31, 17])
        self.assertIn("BGR capture adapted to RGB", description)
        self.assertEqual(sha256(bgr.tobytes()).hexdigest(), raw)

    def test_workflow_lifecycle(self):
        """3 related PublicCaptureDesktopTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_selection_controls_create_new_recipe_and_preserve_original)
        self.scenario(self.check_test_source_changed_during_worker_cannot_show_result)
        self.scenario(self.check_test_preview_is_bounded_and_BGR_capture_displays_correct_RGB)

    def test_validation_and_recovery(self):
        """5 related PublicCaptureDesktopTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_invalid_parameter_does_not_start_worker)
        self.scenario(self.check_test_worker_freezes_selection_and_stale_completion_is_discarded)
        self.scenario(self.check_test_close_requests_cancel_and_defers_until_worker_drains)
        self.scenario(self.check_test_stop_discards_worker_failure_and_drains_before_rerun)
        self.scenario(self.check_test_stored_manifest_load_is_async_and_failure_is_visible)



if __name__ == "__main__":
    unittest.main()
