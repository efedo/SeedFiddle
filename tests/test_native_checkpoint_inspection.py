from __future__ import annotations

from tests.catalogue_fixtures import CatalogueTestCase

from tests.functional_units import functional_units

from dataclasses import replace
import tempfile
import threading
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
@functional_units
class NativeCheckpointInspectionTests(CatalogueTestCase):
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



    def check_test_unbound_missing_cancelled_and_absent_runtime_return_no_pixels(self):
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



    def check_test_changed_marker_support_and_distance_background_reject(self):
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


    def check_test_native_catalogue_match_preserves_source_topology_and_parameter_contracts(self):
        from seedvision.benchmarking.recipe_catalogue import build_recipe_catalogue
        from seedvision.benchmarking.recipe_matching import _matches_native_catalogue
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

    def test_validation_and_recovery(self):
        """2 related NativeCheckpointInspectionTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_unbound_missing_cancelled_and_absent_runtime_return_no_pixels)
        self.scenario(self.check_test_changed_marker_support_and_distance_background_reject)

    def test_workflow_lifecycle(self):
        """1 related NativeCheckpointInspectionTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_native_catalogue_match_preserves_source_topology_and_parameter_contracts)


if __name__ == "__main__":
    unittest.main()
