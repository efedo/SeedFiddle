from __future__ import annotations

from dataclasses import replace
import shutil
import threading
import tempfile
import time
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from seedvision.benchmarking.checkpoint_inspection import (
    InspectionStatus,
    inspect_reference_checkpoint,
)
from seedvision.benchmarking.checkpoints import (
    ExecutionStatus,
    RasterRole,
    load_checkpoint,
    publish_checkpoint,
)
from seedvision.benchmarking.recipes import (
    StageRunRecord,
    StageRunState,
    initial_stage_run,
)
from seedvision.benchmarking.reference_stage import (
    expected_reference_stage_identity,
    run_reference_stage,
)
from seedvision.persistence.reference_graphs import ReferenceGraphAttachment

from test_reference_stage import HAS_REFERENCE_ENV, _fixture, _recipe


@unittest.skipUnless(HAS_REFERENCE_ENV, "the ignored pinned PlantCV reference environment is unavailable")
class ReferenceCheckpointInspectionBridgeTests(unittest.TestCase):
    """Exercise inspection against a real output published by the pinned bridge."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        args, cls.recipe, cls.stage, cls.input_reference, _measurement = _fixture(
            str(cls.root / "bridge"), operation_id="threshold.otsu"
        )
        cls.outcome = run_reference_stage(**args)
        if cls.outcome.checkpoint_reference is None:
            raise AssertionError(f"Pinned PlantCV bridge did not publish: {cls.outcome.run_record}")
        cls.output_reference = cls.outcome.checkpoint_reference
        cls.store = cls.root / "store"
        cls.store.mkdir()
        for source_root, digest in (
            (args["input_root"], cls.input_reference.checkpoint_sha256),
            (args["output_root"], cls.output_reference.checkpoint_sha256),
        ):
            shutil.copytree(source_root / f"{digest}.checkpoint", cls.store / f"{digest}.checkpoint")
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtGui import QFont, QFontDatabase
        from PySide6.QtWidgets import QApplication
        cls.application = QApplication.instance() or QApplication([])
        font_id = QFontDatabase.addApplicationFont(r"C:\Windows\Fonts\segoeui.ttf")
        if font_id >= 0:
            families = QFontDatabase.applicationFontFamilies(font_id)
            if families:
                cls.application.setFont(QFont(families[0], 9))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_real_pinned_output_and_upstream_pixels_are_verified(self) -> None:
        result = inspect_reference_checkpoint(
            self.store, self.recipe, self.outcome.run_record
        )
        self.assertEqual(result.status, InspectionStatus.VERIFIED)
        self.assertEqual(result.checkpoint_sha256, self.output_reference.checkpoint_sha256)
        self.assertEqual(result.source_sha256, self.output_reference.identity.source_sha256)
        self.assertEqual(result.coordinate_frame, self.output_reference.identity.coordinate_frame)
        self.assertEqual(result.execution_status, self.output_reference.execution.status)
        self.assertEqual(set(result.preview_arrays), {"input.gray", "output.mask"})
        actual_output = load_checkpoint(
            self.store, self.output_reference.checkpoint_sha256
        )
        actual_input = load_checkpoint(
            self.store, self.input_reference.checkpoint_sha256
        )
        np.testing.assert_array_equal(result.preview_arrays["output.mask"], actual_output.arrays["mask"])
        np.testing.assert_array_equal(result.preview_arrays["input.gray"], actual_input.arrays["gray"])
        self.assertIn("observations", result.compact_metadata)
        self.assertIn("wall_runtime_ns", result.compact_metadata)
        self.assertFalse(actual_output.measurement.eligible)
        self.assertEqual(actual_output.metadata["scientific_validation_status"], "not_certified")

    def test_absent_store_empty_result_and_recorded_failure_remain_distinct(self) -> None:
        absent = inspect_reference_checkpoint(None, self.recipe, self.outcome.run_record)
        self.assertEqual(absent.status, InspectionStatus.STORE_UNBOUND)
        with tempfile.TemporaryDirectory() as missing:
            self.assertEqual(
                inspect_reference_checkpoint(missing, self.recipe, self.outcome.run_record).status,
                InspectionStatus.MISSING,
            )
        not_run = inspect_reference_checkpoint(
            self.store, self.recipe, initial_stage_run(self.recipe, self.stage.stage_id)
        )
        self.assertEqual(not_run.status, InspectionStatus.NOT_RUN)
        failure = StageRunRecord(
            self.stage.stage_id,
            self.recipe.identity,
            StageRunState.FAILED,
            upstream_checkpoint_sha256=self.input_reference.checkpoint_sha256,
            reason="recorded failure for inspection-state test",
        )
        failed = inspect_reference_checkpoint(self.store, self.recipe, failure)
        self.assertEqual(failed.status, InspectionStatus.RECORDED)
        self.assertIn("FAILED", failed.detail)
        self.assertFalse(failed.preview_arrays)

    def test_captured_environment_manifest_read_is_bounded_before_hashing(self) -> None:
        from seedvision.benchmarking import checkpoint_inspection as inspection_module

        checkpoint = load_checkpoint(
            self.store, self.output_reference.checkpoint_sha256
        )
        environment_path = (
            Path(inspection_module.__file__).resolve().parents[2]
            / "artifacts"
            / "reference-sources"
            / "plantcv-environment.json"
        )
        requested_sizes: list[int] = []

        class OversizedManifest:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, size: int = -1) -> bytes:
                requested_sizes.append(size)
                return b"x" * size

        original_open = Path.open

        def bounded_open(path, *args, **kwargs):
            if path == environment_path:
                return OversizedManifest()
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", new=bounded_open):
            with self.assertRaisesRegex(
                inspection_module.CheckpointMismatchError,
                "environment manifest hash differs",
            ):
                inspection_module._verify_runner_identity(
                    checkpoint.metadata, self.recipe
                )

        self.assertEqual(
            requested_sizes,
            [inspection_module.MAX_ENVIRONMENT_MANIFEST_BYTES + 1],
        )

    def test_valid_empty_mask_is_verified_as_empty_not_absent(self) -> None:
        empty_root = self.root / "empty-bridge"
        args, recipe, stage, upstream, _measurement = _fixture(
            str(empty_root), operation_id="threshold.otsu", empty=True
        )
        outcome = run_reference_stage(**args)
        self.assertEqual(outcome.run_record.state, StageRunState.EMPTY)
        self.assertEqual(outcome.checkpoint_reference.execution.status, ExecutionStatus.EMPTY)
        store = self.root / "empty-store"
        store.mkdir()
        for source_root, digest in (
            (args["input_root"], upstream.checkpoint_sha256),
            (args["output_root"], outcome.run_record.checkpoint_sha256),
        ):
            shutil.copytree(source_root / f"{digest}.checkpoint", store / f"{digest}.checkpoint")
        result = inspect_reference_checkpoint(store, recipe, outcome.run_record)
        self.assertEqual(result.status, InspectionStatus.VERIFIED)
        self.assertEqual(result.execution_status, ExecutionStatus.EMPTY)
        self.assertFalse(np.any(result.preview_arrays["output.mask"]))

    def test_valid_mean_stage_preserves_integer_setting_and_float_runner_offset(self) -> None:
        recipe, stage = _recipe("threshold.mean")
        selected = {"ksize": 3, "offset": 1, "object_type": "light"}
        mean_stage = replace(
            stage,
            parameters=tuple(
                replace(parameter, value=selected[parameter.parameter_id])
                if parameter.parameter_id in selected else parameter
                for parameter in stage.parameters
            ),
        )
        recipe = replace(
            recipe,
            stages=tuple(mean_stage if item.stage_id == stage.stage_id else item for item in recipe.stages),
        )
        bridge_root = self.root / "mean-bridge"
        args, recipe, stage, upstream, _measurement = _fixture(
            str(bridge_root), operation_id="threshold.mean", recipe_override=recipe
        )
        outcome = run_reference_stage(**args)
        self.assertEqual(outcome.run_record.state, StageRunState.EXECUTED)
        store = self.root / "mean-store"
        store.mkdir()
        for source_root, digest in (
            (args["input_root"], upstream.checkpoint_sha256),
            (args["output_root"], outcome.run_record.checkpoint_sha256),
        ):
            shutil.copytree(source_root / f"{digest}.checkpoint", store / f"{digest}.checkpoint")
        result = inspect_reference_checkpoint(store, recipe, outcome.run_record)
        self.assertEqual(result.status, InspectionStatus.VERIFIED, result.detail)
        self.assertEqual(result.compact_metadata["parameters"]["offset"], 1)
        self.assertIs(type(result.compact_metadata["parameters"]["offset"]), int)
        self.assertEqual(result.compact_metadata["effective_parameters"]["offset"], 1.0)
        self.assertIs(type(result.compact_metadata["effective_parameters"]["offset"]), float)

    def test_bridge_named_input_binding_and_additional_upstream_array_are_verified(self) -> None:
        original = load_checkpoint(
            self.store, self.input_reference.checkpoint_sha256
        )
        args, recipe, stage, _input, _measurement = _fixture(
            str(self.root / "named-input"), operation_id="threshold.otsu"
        )
        aliased_root = self.root / "aliased-input-checkpoints"
        aliased_role = RasterRole("image", np.dtype(np.uint8).str, "HW", (0, 255))
        alias_reference = publish_checkpoint(
            aliased_root,
            original.identity,
            {
                "gray_source": original.arrays["gray"],
                "auxiliary": np.asarray([7, 11], dtype=np.uint8),
            },
            execution=original.execution,
            measurement=original.measurement,
            raster_roles={"gray_source": aliased_role},
            metadata={"fixture_only": True},
            allow_cpu_transfer=True,
        )
        args.update({
            "input_root": aliased_root,
            "input_checkpoint_sha256": alias_reference.checkpoint_sha256,
            "expected_upstream_checkpoint_sha256": alias_reference.checkpoint_sha256,
            "expected_output_identity": expected_reference_stage_identity(
                original.identity, recipe, stage.stage_id, alias_reference.checkpoint_sha256
            ),
            "input_bindings": {"gray": "gray_source"},
            "input_roles": {"gray": aliased_role},
        })
        outcome = run_reference_stage(**args)
        self.assertEqual(outcome.run_record.state, StageRunState.EXECUTED)
        store = self.root / "aliased-store"
        store.mkdir()
        shutil.copytree(
            aliased_root / f"{alias_reference.checkpoint_sha256}.checkpoint",
            store / f"{alias_reference.checkpoint_sha256}.checkpoint",
        )
        shutil.copytree(
            args["output_root"] / f"{outcome.run_record.checkpoint_sha256}.checkpoint",
            store / f"{outcome.run_record.checkpoint_sha256}.checkpoint",
        )
        result = inspect_reference_checkpoint(store, recipe, outcome.run_record)
        self.assertEqual(result.status, InspectionStatus.VERIFIED, result.detail)
        self.assertIn(
            "upstream.auxiliary",
            {f"{item['artifact']}.{item['name']}" for item in result.array_inventory},
        )

    def _republished_with_metadata(self, name: str, mutate):
        source = load_checkpoint(self.store, self.output_reference.checkpoint_sha256)
        metadata = _plain(source.metadata)
        mutate(metadata)
        store = self.root / name
        store.mkdir()
        shutil.copytree(
            self.store / f"{self.input_reference.checkpoint_sha256}.checkpoint",
            store / f"{self.input_reference.checkpoint_sha256}.checkpoint",
        )
        reference = publish_checkpoint(
            store,
            source.identity,
            source.arrays,
            execution=source.execution,
            measurement=source.measurement,
            raster_roles=source.raster_roles,
            metadata=metadata,
            allow_cpu_transfer=True,
        )
        record = replace(self.outcome.run_record, checkpoint_sha256=reference.checkpoint_sha256)
        return store, record

    def test_hash_valid_artifacts_with_wrong_source_upstream_role_or_backend_fail_closed(self) -> None:
        mutations = (
            ("source", lambda value: value.update(source_sha256="f" * 64)),
            ("upstream", lambda value: value.update(upstream_checkpoint_sha256="e" * 64)),
            ("role", lambda value: value["output_raster_roles"]["mask"].update(kind="raster")),
            ("backend", lambda value: value["runner_identity"].update(version="9.9.9")),
            ("runner_digest", lambda value: value["runner_identity"].update(identity_digest="f" * 64)),
        )
        for name, mutate in mutations:
            with self.subTest(name=name):
                store, record = self._republished_with_metadata(f"bad-{name}", mutate)
                result = inspect_reference_checkpoint(store, self.recipe, record)
                self.assertEqual(result.status, InspectionStatus.MISMATCH)
                self.assertFalse(result.preview_arrays)

    def test_wrong_output_source_identity_and_recorded_upstream_chain_are_rejected(self) -> None:
        from seedvision.benchmarking.checkpoints import publish_checkpoint

        original = load_checkpoint(self.store, self.output_reference.checkpoint_sha256)
        other_source_identity = replace(original.identity, source_sha256="f" * 64)
        wrong_source_store = self.root / "wrong-output-source"
        wrong_source_store.mkdir()
        shutil.copytree(
            self.store / f"{self.input_reference.checkpoint_sha256}.checkpoint",
            wrong_source_store / f"{self.input_reference.checkpoint_sha256}.checkpoint",
        )
        wrong_source = publish_checkpoint(
            wrong_source_store,
            other_source_identity,
            original.arrays,
            execution=original.execution,
            measurement=original.measurement,
            raster_roles=original.raster_roles,
            metadata=_plain(original.metadata),
            allow_cpu_transfer=True,
        )
        wrong_source_record = replace(
            self.outcome.run_record, checkpoint_sha256=wrong_source.checkpoint_sha256
        )
        self.assertEqual(
            inspect_reference_checkpoint(wrong_source_store, self.recipe, wrong_source_record).status,
            InspectionStatus.MISMATCH,
        )

        alternate_input_store = self.root / "alternate-upstream-source"
        alternate_input_store.mkdir()
        original_input = load_checkpoint(self.store, self.input_reference.checkpoint_sha256)
        different_pixels = np.bitwise_xor(original_input.arrays["gray"], np.uint8(1))
        alternate_input = publish_checkpoint(
            alternate_input_store,
            original_input.identity,
            {"gray": different_pixels},
            execution=original_input.execution,
            measurement=original_input.measurement,
            raster_roles=original_input.raster_roles,
            metadata={"fixture_only": True},
            allow_cpu_transfer=True,
        )
        wrong_upstream_store = self.root / "wrong-upstream-record"
        wrong_upstream_store.mkdir()
        shutil.copytree(
            self.store / f"{self.output_reference.checkpoint_sha256}.checkpoint",
            wrong_upstream_store / f"{self.output_reference.checkpoint_sha256}.checkpoint",
        )
        shutil.copytree(
            alternate_input_store / f"{alternate_input.checkpoint_sha256}.checkpoint",
            wrong_upstream_store / f"{alternate_input.checkpoint_sha256}.checkpoint",
        )
        wrong_upstream_record = replace(
            self.outcome.run_record,
            upstream_checkpoint_sha256=alternate_input.checkpoint_sha256,
        )
        self.assertEqual(
            inspect_reference_checkpoint(wrong_upstream_store, self.recipe, wrong_upstream_record).status,
            InspectionStatus.MISMATCH,
        )

    def test_output_byte_tampering_and_stale_recipe_are_not_previewed(self) -> None:
        tampered = self.root / "tampered-store"
        tampered.mkdir()
        for digest in (self.input_reference.checkpoint_sha256, self.output_reference.checkpoint_sha256):
            shutil.copytree(self.store / f"{digest}.checkpoint", tampered / f"{digest}.checkpoint")
        output_archive = tampered / f"{self.output_reference.checkpoint_sha256}.checkpoint" / "arrays.npz"
        content = output_archive.read_bytes()
        output_archive.write_bytes(content[:-23])
        corrupted = inspect_reference_checkpoint(tampered, self.recipe, self.outcome.run_record)
        self.assertEqual(corrupted.status, InspectionStatus.MISMATCH)
        self.assertFalse(corrupted.preview_arrays)

        historical = replace(self.recipe, recipe_version="99.1")
        stale = inspect_reference_checkpoint(self.store, historical, self.outcome.run_record)
        self.assertEqual(stale.status, InspectionStatus.MISMATCH)
        self.assertFalse(stale.preview_arrays)

    def test_cancellation_before_load_returns_no_pixels(self) -> None:
        import threading

        cancel = threading.Event()
        cancel.set()
        result = inspect_reference_checkpoint(
            self.store, self.recipe, self.outcome.run_record, cancel=cancel
        )
        self.assertEqual(result.status, InspectionStatus.CANCELLED)
        self.assertFalse(result.preview_arrays)

    def _dialog_for_actual_output(self):
        from seedvision.benchmarking.recipes import initial_stage_run
        from seedvision.persistence.reference_graphs import ReferenceGraphAttachment
        from seedvision.ui.reference_graph_dialog import ReferenceGraphDialog

        runs = tuple(
            self.outcome.run_record if stage.stage_id == self.stage.stage_id
            else initial_stage_run(self.recipe, stage.stage_id)
            for stage in self.recipe.stages
        )
        attachment = ReferenceGraphAttachment(
            self.recipe, runs, selected_stage_id=self.stage.stage_id
        )
        dialog = ReferenceGraphDialog(reference_graphs=(attachment,))
        dialog.resize(1440, 900)
        dialog.show()
        self.application.processEvents()
        return dialog

    def _wait_for_status(self, dialog, expected: str, timeout: float = 12.0) -> None:
        from PySide6.QtTest import QTest

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.application.processEvents()
            if expected in dialog._inspection_status_label.text():
                return
            QTest.qWait(10)
        self.fail(
            f"Timed out waiting for {expected!r}; current status was "
            f"{dialog._inspection_status_label.text()!r}."
        )

    def test_dialog_requires_explicit_store_and_shows_verified_actual_pixels(self) -> None:
        from seedvision.segmentation.baseline import PipelineAnalysisCache

        dialog = self._dialog_for_actual_output()
        self.addCleanup(dialog.close)
        self.assertIsNone(dialog._checkpoint_store)
        self.assertIn("not selected", dialog._store_label.text())
        before = dialog.reference_graphs
        changed = []
        dialog.reference_graphs_changed.connect(changed.append)
        kept = np.arange(12, dtype=np.uint8).reshape(3, 4)
        cache = PipelineAnalysisCache(values={"kept": kept})

        dialog.bind_checkpoint_store(self.store)
        self._wait_for_status(dialog, "VERIFIED")
        self.assertEqual(dialog.reference_graphs, before)
        self.assertEqual(changed, [])
        self.assertIn(str(self.store.resolve()), dialog._store_label.text())
        self.assertIn(self.output_reference.identity.source_sha256, dialog._inspection_identity_label.text())
        self.assertIn("not independently verified", dialog._inspection_identity_label.text())
        self.assertIn("not overlaid", dialog._inspection_identity_label.text())
        output_preview = dialog._inspection_preview_labels["output.mask"]
        self.assertFalse(output_preview.pixmap().isNull())
        self.assertLessEqual(output_preview.pixmap().width(), output_preview.width())
        self.assertLessEqual(output_preview.pixmap().height(), output_preview.height())
        dialog._detail_tabs.setCurrentIndex(1)
        self.application.processEvents()
        self.assertGreater(
            dialog._inspection_preview_descriptions["output.mask"].geometry().top(),
            output_preview.geometry().bottom(),
        )
        self.assertIs(cache.values["kept"], kept)
        np.testing.assert_array_equal(cache.values["kept"], kept)
        self.assertIn("Checkpoint arrays: VERIFIED", dialog._facts_browser.toPlainText())

        from pathlib import Path
        render_root = Path("artifacts")
        render_root.mkdir(exist_ok=True)
        self.assertTrue(dialog.grab().save(str(render_root / "reference-checkpoint-inspection.png")))
        self.assertTrue(
            dialog._detail_tabs.currentWidget().grab().save(
                str(render_root / "reference-checkpoint-inspection-detail.png")
            )
        )

    def test_preview_axis_sampling_bounds_extreme_wide_and_tall_images(self) -> None:
        from seedvision.ui import reference_graph_dialog as dialog_module

        dialog = self._dialog_for_actual_output()
        self.addCleanup(dialog.close)
        with patch.object(dialog_module, "MAX_PREVIEW_PIXELS", 100):
            dialog._set_preview_array("input.gray", np.zeros((1, 10_000), dtype=np.uint8))
            wide = dialog._inspection_preview_descriptions["input.gray"].text()
            self.assertIn("10 preview pixels", wide)
            dialog._set_preview_array("output.mask", np.zeros((10_000, 1), dtype=np.uint8))
            tall = dialog._inspection_preview_descriptions["output.mask"].text()
            self.assertIn("10 preview pixels", tall)

    def test_stage_change_discards_stale_result_and_dialog_done_cancels_worker(self) -> None:
        from PySide6.QtCore import QThreadPool
        from PySide6.QtTest import QTest
        from seedvision.ui import reference_graph_dialog as dialog_module
        original_inspector = dialog_module.inspect_reference_checkpoint

        dialog = self._dialog_for_actual_output()
        self.addCleanup(dialog.close)
        started = threading.Event()
        release = threading.Event()
        received_cancels = []

        def blocked(_store, _recipe, _record, *, cancel=None):
            if _store is None:
                return original_inspector(_store, _recipe, _record, cancel=cancel)
            received_cancels.append(cancel)
            started.set()
            release.wait(5)
            return inspect_reference_checkpoint(None, self.recipe, initial_stage_run(self.recipe, self.stage.stage_id))

        with patch.object(dialog_module, "inspect_reference_checkpoint", side_effect=blocked):
            dialog.bind_checkpoint_store(self.store)
            self.assertTrue(started.wait(3))
            other = next(stage for stage in self.recipe.stages if stage.stage_id != self.stage.stage_id)
            dialog.canvas.select_node(other.stage_id)
            self.application.processEvents()
            self.assertTrue(received_cancels[0].is_set())
            release.set()
            self.assertTrue(QThreadPool.globalInstance().waitForDone(5000))
            self.application.processEvents()
            QTest.qWait(20)
            self.assertEqual(dialog.active_attachment.selected_stage_id, other.stage_id)
            self.assertIn("UNRESOLVED", dialog._inspection_status_label.text())
            self.assertEqual(dialog._inspection_result.status, InspectionStatus.RECORDED)

        second = self._dialog_for_actual_output()
        self.addCleanup(second.close)
        started.clear()
        release.clear()
        received_cancels.clear()
        with patch.object(dialog_module, "inspect_reference_checkpoint", side_effect=blocked):
            second.bind_checkpoint_store(self.store)
            self.assertTrue(started.wait(3))
            second.reject()
            self.assertTrue(received_cancels[0].is_set())
            prior_text = second._inspection_status_label.text()
            release.set()
            self.assertTrue(QThreadPool.globalInstance().waitForDone(5000))
            self.application.processEvents()
            QTest.qWait(20)
            self.assertEqual(second._inspection_status_label.text(), prior_text)
            self.assertFalse(second._inspection_workers)

    def test_finished_auto_deleted_worker_can_be_cancelled_before_slot_delivery(self) -> None:
        from PySide6.QtCore import QThreadPool

        dialog = self._dialog_for_actual_output()
        self.addCleanup(dialog.close)
        dialog.bind_checkpoint_store(self.store)

        # Keep the GUI event queue untouched while the real QRunnable finishes.
        # Its C++ object is auto-deleted before the queued completion slot runs.
        self.assertTrue(QThreadPool.globalInstance().waitForDone(30_000))
        self.assertIn("LOADING", dialog._inspection_status_label.text())

        dialog.clear_checkpoint_store()
        self.assertIn("STORE UNBOUND", dialog._inspection_status_label.text())
        self.application.processEvents()
        self.assertIn("STORE UNBOUND", dialog._inspection_status_label.text())

        closing = self._dialog_for_actual_output()
        self.addCleanup(closing.close)
        closing.bind_checkpoint_store(self.store)
        self.assertTrue(QThreadPool.globalInstance().waitForDone(30_000))
        prior_text = closing._inspection_status_label.text()
        closing.reject()
        self.application.processEvents()
        self.assertEqual(closing._inspection_status_label.text(), prior_text)


def _plain(value):
    if hasattr(value, "items"):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


if __name__ == "__main__":
    unittest.main()
