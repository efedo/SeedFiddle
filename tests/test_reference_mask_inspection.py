from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import time
import unittest
from unittest.mock import patch

import numpy as np

from seedvision.benchmarking.checkpoint_inspection import InspectionStatus, inspect_reference_checkpoint
from seedvision.benchmarking.checkpoints import ExecutionStatus, load_checkpoint, publish_checkpoint
from seedvision.benchmarking.reference_mask_stage import run_reference_mask_stage
from seedvision.benchmarking.recipes import initial_stage_run
import test_reference_mask_stage as fixtures
from test_reference_checkpoint_inspection import _plain


class SelectedRecipeInspectionTests(unittest.TestCase):
    def test_selected_values_are_allowed_but_changed_contracts_are_not(self):
        from seedvision.benchmarking.recipe_catalogue import build_recipe_catalogue
        from seedvision.ui.reference_graph_dialog import _matches_catalogue_with_selected_values
        canonical = build_recipe_catalogue().recipe("plantcv_seed_binary")
        median = next(stage for stage in canonical.stages if stage.operation_id == "filter.median_blur")
        selected_stage = replace(median, parameters=tuple(
            replace(parameter, value=3) if parameter.parameter_id == "ksize" else parameter
            for parameter in median.parameters
        ))
        selected = replace(canonical, stages=tuple(
            selected_stage if stage.stage_id == median.stage_id else stage for stage in canonical.stages
        ))
        self.assertNotEqual(selected.identity, canonical.identity)
        self.assertTrue(_matches_catalogue_with_selected_values(selected, canonical))
        self.assertFalse(_matches_catalogue_with_selected_values(selected, None))
        changed = replace(selected, provenance=replace(selected.provenance, backend_version="changed"))
        self.assertFalse(_matches_catalogue_with_selected_values(changed, canonical))
        altered_stage = replace(selected_stage, parameters=tuple(
            replace(parameter, description="Different declared meaning") for parameter in selected_stage.parameters
        ))
        altered = replace(selected, stages=tuple(
            altered_stage if stage.stage_id == median.stage_id else stage for stage in selected.stages
        ))
        self.assertFalse(_matches_catalogue_with_selected_values(altered, canonical))


@unittest.skipUnless(fixtures.HAS_REFERENCE_ENV, "Pinned original PlantCV environment unavailable")
class ReferenceMaskInspectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.ReferenceMaskStageTests.setUpClass()
        cls.addClassCleanup(fixtures.ReferenceMaskStageTests.tearDownClass)
        fixture = fixtures.ReferenceMaskStageTests()
        cls.cases = []
        for operation, parameters, array in (
            ("filter.median_blur", {"ksize": 3}, None),
            ("filter.median_blur", {"ksize": 3}, np.zeros((48, 72), np.uint8)),
            ("morphology.fill", {"size": 10}, None),
            ("morphology.erode", {"ksize": 3, "iterations": 2}, None),
            ("morphology.dilate", {"ksize": 3, "iterations": 2}, None),
        ):
            case = fixture._case(operation, parameters, array=array)
            case["output_root"] = case["input_root"]
            outcome = run_reference_mask_stage(**{k: v for k, v in case.items() if k != "input_reference"})
            if outcome.checkpoint_reference is None:
                raise AssertionError(outcome.run_record)
            cls.cases.append((case, outcome))

    def test_all_original_mask_stages_show_exact_pixels_and_zero_gray_is_available(self):
        for case, outcome in self.cases:
            with self.subTest(stage=case["stage_id"]):
                result = inspect_reference_checkpoint(case["input_root"], case["recipe"], outcome.run_record)
                self.assertEqual(result.status, InspectionStatus.VERIFIED, result.detail)
                output = load_checkpoint(case["output_root"], outcome.run_record.checkpoint_sha256)
                upstream = load_checkpoint(case["input_root"], case["input_checkpoint_sha256"])
                port, source_name = next(iter(case["input_bindings"].items()))
                output_name = next(iter(output.arrays))
                np.testing.assert_array_equal(result.preview_arrays[f"input.{port}"], upstream.arrays[source_name])
                np.testing.assert_array_equal(result.preview_arrays[f"output.{output_name}"], output.arrays[output_name])
                if output_name == "image":
                    self.assertEqual(result.execution_status, ExecutionStatus.AVAILABLE)

    def test_wrong_mask_stage_metadata_never_exposes_previews(self):
        for case, outcome in self.cases:
            original = load_checkpoint(case["output_root"], outcome.run_record.checkpoint_sha256)
            for field, replacement in (("format", "seedfiddle.reference-threshold-stage-output"), ("parameters", {})):
                metadata = _plain(original.metadata)
                metadata[field] = replacement
                changed = publish_checkpoint(
                    case["output_root"], original.identity, original.arrays,
                    execution=original.execution, measurement=original.measurement,
                    raster_roles=original.raster_roles, metadata=metadata, allow_cpu_transfer=True,
                )
                record = replace(outcome.run_record, checkpoint_sha256=changed.checkpoint_sha256)
                result = inspect_reference_checkpoint(case["output_root"], case["recipe"], record)
                self.assertEqual(result.status, InspectionStatus.MISMATCH)
                self.assertFalse(result.preview_arrays)

    def test_dialog_renders_median_image_and_fill_input_mask(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtGui import QFont, QFontDatabase
        from PySide6.QtTest import QTest
        from PySide6.QtWidgets import QApplication
        from seedvision.persistence.reference_graphs import ReferenceGraphAttachment
        from seedvision.ui.reference_graph_dialog import ReferenceGraphDialog
        app = QApplication.instance() or QApplication([])
        font_id = QFontDatabase.addApplicationFont(r"C:\Windows\Fonts\segoeui.ttf")
        if font_id >= 0:
            families = QFontDatabase.applicationFontFamilies(font_id)
            if families:
                app.setFont(QFont(families[0], 9))
        for index in (0, 2):
            case, outcome = self.cases[index]
            runs = tuple(
                outcome.run_record if stage.stage_id == case["stage_id"]
                else initial_stage_run(case["recipe"], stage.stage_id)
                for stage in case["recipe"].stages
            )
            attachment = ReferenceGraphAttachment(recipe=case["recipe"], stage_runs=runs, selected_stage_id=case["stage_id"])
            dialog = ReferenceGraphDialog(reference_graphs=(attachment,))
            try:
                dialog.show()
                with patch.object(dialog, "_set_preview_array", wraps=dialog._set_preview_array) as display:
                    dialog.bind_checkpoint_store(case["input_root"])
                    deadline = time.monotonic() + 12
                    while "VERIFIED" not in dialog._inspection_status_label.text() and time.monotonic() < deadline:
                        app.processEvents()
                        QTest.qWait(10)
                    self.assertIn("VERIFIED", dialog._inspection_status_label.text())
                    self.assertIn("SELECTED PARAMETERS", dialog.attachment_label.text())
                    self.assertEqual(display.call_count, 2)
                    self.assertEqual({call.args[0] for call in display.call_args_list}, {"input.gray", "output.mask"})
                app.processEvents()
                dialog._detail_tabs.setCurrentIndex(1)
                app.processEvents()
                if index == 0:
                    target = Path("artifacts/reference-median-inspection.png")
                    target.parent.mkdir(exist_ok=True)
                    self.assertTrue(dialog.grab().save(str(target)))
            finally:
                dialog.close()
