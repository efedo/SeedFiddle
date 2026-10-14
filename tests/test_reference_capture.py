from __future__ import annotations

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
class ReferenceCaptureContractsTests(unittest.TestCase):
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

    def test_actual_complete_capture_reopens_without_reading_original_image(self):
        self.assertEqual(len(self.operations), 8)
        self.assertEqual(self.operations[-3:], ["segmentation.watershed.capture", "analysis.size", "analysis.color"])
        with patch("seedvision.benchmarking.native_image_inputs.snapshot", side_effect=AssertionError("Stored revision must not decode current image.")):
            opened = open_reference_capture(self.manifest)
        self.assertEqual(opened.recipe, self.recipe)
        self.assertEqual(opened.verified.manifest_sha256, self.session.verified.manifest_sha256)
        self.assertEqual(len(opened.verified.checkpoints), 10)
        self.assertEqual(opened.verified.retained_array_bytes,
            sum(a.nbytes for c in opened.verified.checkpoints.values() for a in c.arrays.values()))
        self.assertFalse(opened.verified.record["scientifically_validated"])
        self.assertEqual(len({call["call_id"] for call in opened.verified.record["source_calls"]}), 3)
        for name in ("gray", "rgb"):
            self.assertIs(opened.verified.checkpoints[f"input.{name}"].metadata["actual_public_reader"], False)
        rgb = opened.verified.checkpoints["input.rgb"].arrays["rgb"]
        np.testing.assert_array_equal(rgb[0, 0], [63, 31, 17])
        for checkpoint in opened.verified.checkpoints.values():
            for array in checkpoint.arrays.values():
                self.assertFalse(array.flags.writeable)

    def test_manifest_fields_order_counts_and_call_id_reuse_reject(self):
        changes = {
            "source": lambda r: r.update(source_sha256="f" * 64),
            "frame": lambda r: r["coordinate_frame"].update(frame_id="wrong_frame"),
            "recipe": lambda r: r["recipe_identity"].update(settings_sha256="e" * 64),
            "count_bool": lambda r: r.update(runner_request_count_after_preprocessing=True),
            "partial": lambda r: r.update(status="partial"),
            "science": lambda r: r.update(scientifically_validated=True),
            "order": lambda r: r["source_calls"].reverse(),
            "missing": lambda r: r["preprocessing"].pop(),
            "duplicate_id": lambda r: r["source_calls"][1].update(call_id=r["source_calls"][0]["call_id"]),
            "wrong_input": lambda r: r["inputs"]["rgb"].update(array_name="gray"),
            "label_facts": lambda r: r["label_facts"].update(maximum_label_id=r["label_facts"]["maximum_label_id"] + 1),
            "n_labels": lambda r: r["source_calls"][2]["details"].update(effective_n_labels=999),
            "reason": lambda r: r["source_calls"][1].update(reason="failure"),
        }
        for name, mutate in changes.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                read_reference_seed_chain(**self._arguments(self._changed_manifest(mutate)))

    def test_republished_source_metadata_parameters_pins_and_context_reject(self):
        changes = {
            "RGB_input_digest": ("size", {"metadata_change": lambda m: m["input_array_sha256"].update(image="a" * 64)}),
            "effective": ("color", {"metadata_change": lambda m: m["effective_parameters"].update(n_labels=999)}),
            "origin": ("size", {"metadata_change": lambda m: m["array_origins"].update(labels_context="computed")}),
            "source_pin": ("color", {"metadata_change": lambda m: m["runner_identity"].update(identity_digest="a" * 64)}),
            "worker_id": ("watershed", {"metadata_change": lambda m: m["effective_parameters"]["capture"].update(call_id="wrong")}),
            "role": ("size", {"roles_change": lambda r: r.update(labels_context=RasterRole("raster", "<i4", "HW"))}),
            "context": ("color", {"array_change": lambda a: a.update(labels_context=np.zeros_like(a["labels_context"]))}),
            "runtime": ("size", {"execution_change": lambda e: replace(e, elapsed_ms=e.elapsed_ms + 1)}),
        }
        for name, (key, kwargs) in changes.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                digest = self._changed_manifest(lambda r: self._replace_checkpoint(r, key, **kwargs))
                read_reference_seed_chain(**self._arguments(digest))

    def test_republished_preprocessing_runtime_and_parameters_reject(self):
        for change in ({"execution_change": lambda e: replace(e, elapsed_ms=e.elapsed_ms + 1)},
                       {"metadata_change": lambda m: m["parameters"].update(ksize=7)}):
            with self.subTest(change=tuple(change)), self.assertRaises(ValueError):
                digest = self._changed_manifest(lambda r: self._replace_checkpoint(r, "preprocessing.0", **change))
                read_reference_seed_chain(**self._arguments(digest))

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

    def test_retained_budget_is_aggregate_and_cancel_returns_no_view(self):
        total = self.session.verified.retained_array_bytes
        verified = read_reference_seed_chain(**self._arguments(), limits=replace(CheckpointLimits(), max_total_array_bytes=total,
                                                                                max_array_bytes=total))
        self.assertEqual(verified.retained_array_bytes, total)
        with self.assertRaises(ValueError):
            read_reference_seed_chain(**self._arguments(), limits=replace(CheckpointLimits(), max_total_array_bytes=total - 1,
                                                                         max_array_bytes=total - 1))
        for boundary in (1, 4, 12, 22):
            count = 0
            def cancel():
                nonlocal count
                count += 1
                return count >= boundary
            with self.subTest(boundary=boundary), self.assertRaises(CheckpointCancelled):
                read_reference_seed_chain(**self._arguments(), cancel=cancel)

    def test_byte_corruption_is_rejected_without_changing_valid_store(self):
        with tempfile.TemporaryDirectory(prefix="capture-corrupt-") as directory:
            from shutil import copytree
            copy = Path(directory) / "copy"
            copytree(self.store, copy)
            checkpoint = self.session.verified.checkpoints["color"]
            # Corrupt only the copied content-addressed metadata, not fixtures.
            metadata = copy / f"{checkpoint.checkpoint_sha256}.checkpoint" / "manifest.json"
            metadata.write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                read_reference_seed_chain(**self._arguments(store=copy))

    def test_saved_recipe_binding_rejects_swapped_selection(self):
        digest = self.session.verified.manifest_sha256
        sidecar = self.manifest.with_name(f"{digest}.recipe.json")
        raw = sidecar.read_bytes()
        try:
            sidecar.write_text(_selected_recipe("plantcv_seed_mean").to_json(), encoding="utf-8")
            with self.assertRaises(ValueError):
                open_reference_capture(self.manifest)
        finally:
            sidecar.write_bytes(raw)

    def test_changed_source_is_rejected_before_preparation(self):
        with patch.object(PlantCVReferenceRunner, "run") as public, self.assertRaises(ValueError):
            capture_reference_path(self.recipe, self.source, self.root / "changed", PlantCVReferenceRunner(REFERENCE_PYTHON), expected_stamp=(0, 0, 0, 0, 0))
        public.assert_not_called()

    def _fake(self):
        # A plain function installed on the runner class receives its runner
        # instance. A bound TestCase method would shift all call arguments.
        def run(runner, operation, arrays, parameters=None, **kwargs):
            result = self.results[operation]
            return replace(result, arrays={name: a.copy() for name, a in result.arrays.items()},
                           effective_parameters=deepcopy(dict(result.effective_parameters)),
                           identity=deepcopy(dict(result.identity)))
        return run

    def test_cancel_during_read_retracts_only_own_manifest_and_retains_checkpoints(self):
        store, event = self.root / "cancelled", threading.Event()
        def cancelled_read(*args, **kwargs):
            event.set()
            raise CheckpointCancelled("Injected cancellation while verifying desktop revision.")
        with patch.object(PlantCVReferenceRunner, "run", self._fake()), patch(
                "seedvision.benchmarking.reference_capture_session.read_reference_seed_chain", cancelled_read), self.assertRaises(CheckpointCancelled):
            capture_reference_path(self.recipe, self.source, store, PlantCVReferenceRunner(REFERENCE_PYTHON), cancel=event)
        self.assertEqual(list((store / "chains").iterdir()), [])
        self.assertTrue(any(store.glob("*.checkpoint")))
        self.assertTrue(self.manifest.exists())

    def test_sidecar_write_failure_retracts_only_own_manifest(self):
        store = self.root / "write_failure"
        original = os.replace
        def fail_sidecar(source, destination):
            if str(destination).endswith(".recipe.json"):
                raise OSError("Injected sidecar publication failure.")
            return original(source, destination)
        with patch.object(PlantCVReferenceRunner, "run", self._fake()), patch(
                "seedvision.benchmarking.reference_capture_session.os.replace", fail_sidecar), self.assertRaises(OSError):
            capture_reference_path(self.recipe, self.source, store, PlantCVReferenceRunner(REFERENCE_PYTHON))
        self.assertEqual(list((store / "chains").iterdir()), [])
        self.assertTrue(self.manifest.exists())

    def test_cancel_after_verification_before_sidecar_commit_withholds_revision(self):
        store, event = self.root / "before_commit", threading.Event()
        actual = read_reference_seed_chain
        def cancel_after_read(*args, **kwargs):
            verified = actual(*args, **kwargs)
            event.set()
            return verified
        with patch.object(PlantCVReferenceRunner, "run", self._fake()), patch(
                "seedvision.benchmarking.reference_capture_session.read_reference_seed_chain", cancel_after_read), self.assertRaises(CheckpointCancelled):
            capture_reference_path(self.recipe, self.source, store, PlantCVReferenceRunner(REFERENCE_PYTHON), cancel=event)
        self.assertEqual(list((store / "chains").iterdir()), [])
        self.assertTrue(any(store.glob("*.checkpoint")))

    def test_cancel_after_sidecar_commit_preserves_completed_revision(self):
        store, event = self.root / "after_commit", threading.Event()
        actual = os.replace
        def cancel_after_replace(source, destination):
            value = actual(source, destination)
            if str(destination).endswith(".recipe.json"):
                event.set()
            return value
        with patch.object(PlantCVReferenceRunner, "run", self._fake()), patch(
                "seedvision.benchmarking.reference_capture_session.os.replace", cancel_after_replace):
            session = capture_reference_path(self.recipe, self.source, store, PlantCVReferenceRunner(REFERENCE_PYTHON), cancel=event)
        self.assertTrue(event.is_set())
        self.assertIsNotNone(session.verified)
        opened = open_reference_capture(session.outcome.manifest_path)
        self.assertEqual(opened.verified.manifest_sha256, session.verified.manifest_sha256)

    def test_replaced_manifest_survives_cancelled_attempt_cleanup(self):
        store, replaced = self.root / "replaced_manifest", []
        def replace_and_fail(output_root, digest, **kwargs):
            path = Path(output_root) / "chains" / f"{digest}.json"
            path.write_bytes(b"A replacement file does not belong to this attempt.")
            replaced.append(path)
            raise CheckpointCancelled("Injected cancellation after external replacement.")
        with patch.object(PlantCVReferenceRunner, "run", self._fake()), patch(
                "seedvision.benchmarking.reference_capture_session.read_reference_seed_chain", replace_and_fail), self.assertRaises(CheckpointCancelled):
            capture_reference_path(self.recipe, self.source, store, PlantCVReferenceRunner(REFERENCE_PYTHON))
        self.assertEqual(replaced[0].read_bytes(), b"A replacement file does not belong to this attempt.")
        self.assertTrue(self.manifest.exists())

    def test_actual_stored_capture_updates_desktop_only_after_worker_verification(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        from seedvision.ui.public_reference_capture_dialog import PublicReferenceCaptureDialog, MAX_TOTAL_PREVIEW_PIXELS
        app = QApplication.instance() or QApplication([])
        dialog = PublicReferenceCaptureDialog(self.recipe, source_path=self.source, store=self.store)
        self.addCleanup(dialog.close)
        dialog.open_saved(self.manifest)
        deadline = time.monotonic() + 10
        while dialog._worker is not None and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.005)
        self.assertIsNone(dialog._worker)
        self.assertIsNotNone(dialog._reply, dialog.status_label.text())
        self.assertEqual(dialog.selected_recipe(), self.recipe)
        self.assertIsNone(dialog._source_path)
        self.assertIn(self.session.verified.manifest_sha256, dialog.status_label.text())
        self.assertIn("actual_source_metadata", dialog.facts.toPlainText())
        self.assertLessEqual(sum(a.image.shape[0] * a.image.shape[1] for a in dialog._reply.artifacts if a.image is not None), MAX_TOTAL_PREVIEW_PIXELS)
        index = next(i for i, a in enumerate(dialog._reply.artifacts) if a.key == "watershed.diagnostic_bgr")
        dialog.artifact_selector.setCurrentIndex(index)
        self.assertIn("BGR capture adapted to RGB", dialog.preview_description.text())
        self.assertFalse(dialog.preview.pixmap().isNull())


class PublicCaptureDesktopTests(unittest.TestCase):
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

    def test_selection_controls_create_new_recipe_and_preserve_original(self):
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

    def test_invalid_parameter_does_not_start_worker(self):
        dialog = self._dialog()
        key = next(key for key in dialog._editors if key[1] == "ksize")
        dialog._editors[key].setText("true")
        with patch("seedvision.ui.public_reference_capture_dialog.QThreadPool.globalInstance") as pool:
            dialog.start_capture()
        pool.assert_not_called()
        self.assertIn("Cannot start", dialog.status_label.text())

    def test_worker_freezes_selection_and_stale_completion_is_discarded(self):
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

    def test_source_changed_during_worker_cannot_show_result(self):
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

    def test_close_requests_cancel_and_defers_until_worker_drains(self):
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

    def test_stop_discards_worker_failure_and_drains_before_rerun(self):
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

    def test_stored_manifest_load_is_async_and_failure_is_visible(self):
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

    def test_preview_is_bounded_and_BGR_capture_displays_correct_RGB(self):
        from seedvision.ui.public_reference_capture_dialog import preview_rgb
        bgr = np.empty((401, 619, 3), dtype=np.uint8)
        bgr[:] = (17, 31, 63)
        raw = sha256(bgr.tobytes()).hexdigest()
        display, description = preview_rgb(bgr, "diagnostic_bgr", max_pixels=4096)
        self.assertLessEqual(display.shape[0] * display.shape[1], 4096)
        np.testing.assert_array_equal(display[0, 0], [63, 31, 17])
        self.assertIn("BGR capture adapted to RGB", description)
        self.assertEqual(sha256(bgr.tobytes()).hexdigest(), raw)

    def test_graph_launch_eligibility_and_attachments_remain_unchanged(self):
        from seedvision.ui.reference_graph_dialog import ReferenceGraphDialog
        from seedvision.ui.public_reference_capture_dialog import PublicReferenceCaptureDialog
        dialog = ReferenceGraphDialog()
        self.addCleanup(dialog.close)
        for recipe_id, enabled in (("plantcv_seed_binary", True), ("plantcv_seed_mean", True),
                                   ("plantcv_seed_otsu", True), ("plantcv_color_card_correct_color", False)):
            dialog.recipe_selector.setCurrentIndex(dialog.recipe_selector.findData(recipe_id))
            self.app.processEvents()
            self.assertEqual(dialog._public_capture_button.isEnabled(), enabled)
        dialog.recipe_selector.setCurrentIndex(dialog.recipe_selector.findData("plantcv_seed_binary"))
        before = dialog.reference_graphs
        native = dialog._native_recipes
        with patch.object(PublicReferenceCaptureDialog, "exec", return_value=0):
            dialog._open_public_capture()
        self.assertEqual(dialog.reference_graphs, before)
        self.assertEqual(dialog._native_recipes, native)


if __name__ == "__main__":
    unittest.main()
