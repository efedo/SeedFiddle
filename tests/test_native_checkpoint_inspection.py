from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import os
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch

import tests.test_native_mask_stage as mask_fixtures
import tests.test_native_threshold_stage as threshold_fixtures
import tests.test_native_watershed_stage as watershed_fixtures
from seedvision.benchmarking import native_checkpoint_inspection as inspection_module
from seedvision.benchmarking.checkpoint_inspection import InspectionStatus, _plain
from seedvision.benchmarking.checkpoints import load_checkpoint, publish_checkpoint
from seedvision.benchmarking.native_checkpoint_inspection import inspect_native_checkpoint
from seedvision.benchmarking.recipes import initial_stage_run


@unittest.skipUnless(torch.cuda.is_available(), "native checkpoint inspection requires CUDA identity verification")
class NativeCheckpointInspectionTests(unittest.TestCase):
    def _case(self, operation="threshold.binary", *, empty=False):
        if operation.startswith("threshold."):
            temporary = self.enterContext(tempfile.TemporaryDirectory(prefix="native-inspection-"))
            args, _values, _parameters = threshold_fixtures._fixture(
                temporary, operation, values=np.zeros((13, 19), dtype=np.uint8) if empty else None,
            )
            args["output_root"] = args["input_root"]
            outcome = inspection_module.thresholds.run_native_threshold_stage(**args)
        elif operation in inspection_module.masks.SUPPORTED_NATIVE_MASK_OPERATIONS:
            helper = mask_fixtures.NativeMaskCudaStageTests()
            helper.device = torch.device("cuda", torch.cuda.current_device())
            helper.implementation = inspection_module.masks.native_mask_identity(helper.device)
            helper.setUp()
            self.addCleanup(helper.tearDown)
            parameters = {
                "filter.median_blur": {"ksize": [2, 3]},
                "morphology.fill": {"size": 3},
                "morphology.erode": {"ksize": 3, "iterations": 2},
                "morphology.dilate": {"ksize": 3, "iterations": 2},
            }[operation]
            case = helper._case(operation, parameters)
            case["output_root"] = case["input_root"]
            outcome = helper._run(case)
            args = case
        else:
            helper = watershed_fixtures.NativeWatershedCudaStageTests()
            helper.device = torch.device("cuda", torch.cuda.current_device())
            helper.implementation = inspection_module.watershed.native_watershed_identity(helper.device)
            helper.setUp()
            self.addCleanup(helper.tearDown)
            marker = operation == inspection_module.watershed.NATIVE_MARKER_OPERATION_ID
            values = np.zeros((19, 25), dtype=np.bool_ if marker else np.uint8)
            if not empty:
                if marker:
                    values[3, 4] = values[4, 5] = values[11, 16] = True
                else:
                    values[3:16, 4:21] = 255
            case = helper._case(operation, values)
            case["output_root"] = case["input_root"]
            outcome = helper._run(case)
            args = case
        self.assertIsNotNone(outcome.checkpoint_reference, outcome.run_record.reason)
        return args["input_root"], args["recipe"], outcome.run_record

    @staticmethod
    def _mutate(root, record, *, metadata=None, measurement=None, arrays=None):
        old = load_checkpoint(root, record.checkpoint_sha256)
        ref = publish_checkpoint(
            root, old.identity, dict(old.arrays) if arrays is None else arrays,
            execution=old.execution, measurement=old.measurement if measurement is None else measurement,
            raster_roles=dict(old.raster_roles), metadata=_plain(old.metadata) if metadata is None else metadata,
            allow_cpu_transfer=True,
        )
        return replace(record, checkpoint_sha256=ref.checkpoint_sha256)

    def test_actual_nine_native_operations_verify_their_archived_pixels(self):
        operations = (
            "threshold.binary", "threshold.mean", "threshold.otsu", "filter.median_blur",
            "morphology.fill", "morphology.erode", "morphology.dilate",
            inspection_module.watershed.NATIVE_DISTANCE_OPERATION_ID,
            inspection_module.watershed.NATIVE_MARKER_OPERATION_ID,
        )
        for operation in operations:
            with self.subTest(operation=operation):
                root, recipe, record = self._case(operation)
                result = inspect_native_checkpoint(root, recipe, record)
                self.assertEqual(result.status, InspectionStatus.VERIFIED, result.detail)
                self.assertEqual(result.checkpoint_sha256, record.checkpoint_sha256)
                output = load_checkpoint(root, record.checkpoint_sha256)
                name, value = next(iter(output.arrays.items()))
                np.testing.assert_array_equal(result.preview_arrays[f"output.{name}"], value)
                self.assertFalse(output.measurement.eligible)
                self.assertEqual(result.compact_metadata["reference_comparison_status"], "not_evaluated")
                metadata = _plain(output.metadata)
                metadata["native_supported_domain"] = "All image types and unlimited rasters are supported."
                altered = self._mutate(root, record, metadata=metadata)
                rejected = inspect_native_checkpoint(root, recipe, altered)
                self.assertEqual(rejected.status, InspectionStatus.MISMATCH, rejected.detail)
                self.assertIn("native_supported_domain", rejected.detail)
                self.assertFalse(rejected.preview_arrays)

    def test_empty_distance_is_available_but_empty_peak_markers_are_empty(self):
        for operation in (inspection_module.watershed.NATIVE_DISTANCE_OPERATION_ID, inspection_module.watershed.NATIVE_MARKER_OPERATION_ID):
            root, recipe, record = self._case(operation, empty=True)
            result = inspect_native_checkpoint(root, recipe, record)
            self.assertEqual(result.status, InspectionStatus.VERIFIED, result.detail)
            self.assertEqual(result.execution_status.value, "available" if operation.endswith("distance_map") else "empty")

    def test_unbound_missing_cancelled_and_absent_runtime_return_no_pixels(self):
        root, recipe, record = self._case()
        self.assertEqual(inspect_native_checkpoint(None, recipe, record).status, InspectionStatus.STORE_UNBOUND)
        self.assertEqual(inspect_native_checkpoint(root/"missing", recipe, record).status, InspectionStatus.MISSING)
        event = threading.Event()
        event.set()
        with patch.object(inspection_module, "load_checkpoint", side_effect=AssertionError("cancelled load")):
            result = inspect_native_checkpoint(root, recipe, record, cancel=event)
        self.assertEqual(result.status, InspectionStatus.CANCELLED)
        with patch.object(inspection_module.thresholds, "native_threshold_identity", side_effect=inspection_module.thresholds.NativeThresholdUnavailableError("test runtime unavailable")):
            result = inspect_native_checkpoint(root, recipe, record)
        self.assertEqual(result.status, InspectionStatus.UNSUPPORTED, result.detail)
        self.assertFalse(result.preview_arrays)
        self.assertEqual(inspect_native_checkpoint(root, recipe, initial_stage_run(recipe, record.stage_id)).status, InspectionStatus.NOT_RUN)

    def test_readdressed_bad_parameters_source_roles_environment_and_eligibility_reject(self):
        root, recipe, record = self._case()
        original = load_checkpoint(root, record.checkpoint_sha256)
        mutations = (
            ("selected_parameters", {"threshold": 0, "object_type": "light"}),
            ("source_sha256", "f"*64),
            ("input_bindings", {"gray": "missing"}),
            ("effective_parameters", {"threshold": 1, "object_type": "light"}),
            ("implementation_identity", dict(_plain(original.metadata["implementation_identity"]), device_name="different")),
            ("output_inventory", []),
        )
        for name, replacement in mutations:
            with self.subTest(field=name):
                metadata = _plain(original.metadata)
                metadata[name] = replacement
                altered = self._mutate(root, record, metadata=metadata)
                result = inspect_native_checkpoint(root, recipe, altered)
                self.assertEqual(result.status, InspectionStatus.MISMATCH, result.detail)
                self.assertFalse(result.preview_arrays)
        measurement = replace(original.measurement, eligible=True, reason="")
        metadata = _plain(original.metadata)
        metadata["measurement_contract"] = measurement.to_dict()
        result = inspect_native_checkpoint(root, recipe, self._mutate(root, record, measurement=measurement, metadata=metadata))
        self.assertEqual(result.status, InspectionStatus.MISMATCH, result.detail)

    def test_inspection_budget_rejects_before_cuda_identity_or_preview(self):
        root, recipe, record = self._case()
        with patch.object(inspection_module, "MAX_NATIVE_INSPECTED_ARRAY_BYTES", 16):
            with patch.object(inspection_module.thresholds, "native_threshold_identity", side_effect=AssertionError("late preflight")):
                result = inspect_native_checkpoint(root, recipe, record)
        self.assertEqual(result.status, InspectionStatus.MISMATCH, result.detail)
        self.assertFalse(result.preview_arrays)

    def test_changed_marker_support_and_distance_background_reject(self):
        for operation in (inspection_module.watershed.NATIVE_DISTANCE_OPERATION_ID, inspection_module.watershed.NATIVE_MARKER_OPERATION_ID):
            root, recipe, record = self._case(operation)
            output = load_checkpoint(root, record.checkpoint_sha256)
            name, pixels = next(iter(output.arrays.items()))
            changed = pixels.copy()
            changed[0, 0] = 1
            altered = self._mutate(root, record, arrays={name: changed})
            result = inspect_native_checkpoint(root, recipe, altered)
            self.assertEqual(result.status, InspectionStatus.MISMATCH, result.detail)
            self.assertFalse(result.preview_arrays)

    def test_actual_qt_async_native_distance_and_marker_inspection(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtGui import QFont, QFontDatabase
        from PySide6.QtTest import QTest
        from PySide6.QtWidgets import QApplication
        from seedvision.persistence.reference_graphs import ReferenceGraphAttachment
        from seedvision.ui.reference_graph_dialog import ReferenceGraphDialog
        app = QApplication.instance() or QApplication([])
        font_path = Path(r"C:\Windows\Fonts\segoeui.ttf")
        if font_path.is_file():
            font_id = QFontDatabase.addApplicationFont(str(font_path))
            families = QFontDatabase.applicationFontFamilies(font_id)
            if families:
                app.setFont(QFont(families[0], 9))
        for operation in (inspection_module.watershed.NATIVE_DISTANCE_OPERATION_ID, inspection_module.watershed.NATIVE_MARKER_OPERATION_ID):
            root, recipe, record = self._case(operation)
            runs = tuple(record if stage.stage_id == record.stage_id else initial_stage_run(recipe,stage.stage_id) for stage in recipe.stages)
            attachment = ReferenceGraphAttachment(recipe=recipe,stage_runs=runs,selected_stage_id=record.stage_id)
            dialog = ReferenceGraphDialog(reference_graphs=(attachment,))
            try:
                dialog.show()
                dialog.bind_checkpoint_store(root)
                deadline = time.monotonic()+12
                while dialog._inspection_result.status not in {InspectionStatus.VERIFIED,InspectionStatus.MISMATCH} and time.monotonic()<deadline:
                    app.processEvents()
                    QTest.qWait(10)
                self.assertEqual(dialog._inspection_result.status,InspectionStatus.VERIFIED,dialog._inspection_status_label.text())
                for key in ("input.gray","output.mask"):
                    self.assertIsNotNone(dialog._inspection_preview_labels[key].pixmap())
                    self.assertFalse(dialog._inspection_preview_labels[key].pixmap().isNull())
                caption = dialog._inspection_preview_descriptions["output.mask"].text()
                self.assertIn("Distance display" if operation.endswith("distance_map") else "Categorical label",caption)
                self.assertIn("current code/CUDA environment",dialog._inspection_status_label.text())
                self.assertIn("original public PlantCV call does not expose",dialog._facts_browser.toPlainText())
                self.assertIn("verified_native_implementation_identity",dialog._inspection_metadata_browser.toPlainText())
                self.assertIn("NATIVE RECIPE",dialog.attachment_label.text())
                dialog._detail_tabs.setCurrentIndex(1)
                app.processEvents()
                target=Path("artifacts")/("native-distance-inspection.png" if operation.endswith("distance_map") else "native-marker-inspection.png")
                self.assertTrue(dialog.grab().save(str(target)))
                for key in ("input.gray","output.mask"):
                    label=dialog._inspection_preview_labels[key]
                    caption_label=dialog._inspection_preview_descriptions[key]
                    self.assertFalse(label.geometry().intersects(caption_label.geometry()),f"{key}: image={label.geometry()}, caption={caption_label.geometry()}, group={label.parentWidget().geometry()}")
                    self.assertLessEqual(label.pixmap().height(),label.contentsRect().height())
                    self.assertLessEqual(label.pixmap().width(),label.contentsRect().width())
            finally:
                dialog.close()

    def test_native_catalogue_match_preserves_source_topology_and_parameter_contracts(self):
        from seedvision.benchmarking.recipe_catalogue import build_recipe_catalogue
        from seedvision.ui.reference_graph_dialog import _matches_native_catalogue
        _root, recipe, record = self._case()
        catalogue = build_recipe_catalogue().recipe(recipe.recipe_id)
        self.assertTrue(_matches_native_catalogue(recipe,catalogue))
        stage=recipe.stage(record.stage_id)
        changed=replace(stage,source_mappings=(replace(stage.source_mappings[0],source_sha256="f"*64),))
        altered=replace(recipe,stages=tuple(changed if item.stage_id==stage.stage_id else item for item in recipe.stages))
        self.assertFalse(_matches_native_catalogue(altered,catalogue))
        parameter=replace(stage.parameters[0],description="different calculation contract")
        changed=replace(stage,parameters=(parameter,*stage.parameters[1:]))
        altered=replace(recipe,stages=tuple(changed if item.stage_id==stage.stage_id else item for item in recipe.stages))
        self.assertFalse(_matches_native_catalogue(altered,catalogue))


if __name__ == "__main__":
    unittest.main()
