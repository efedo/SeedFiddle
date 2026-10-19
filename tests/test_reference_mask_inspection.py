from __future__ import annotations

from tests.catalogue_fixtures import CatalogueTestCase

from dataclasses import replace
import unittest

import numpy as np

from seedvision.benchmarking.checkpoint_inspection import InspectionStatus, inspect_reference_checkpoint
from seedvision.benchmarking.checkpoints import ExecutionStatus, load_checkpoint, publish_checkpoint
from seedvision.benchmarking.reference_mask_stage import run_reference_mask_stage
from seedvision.benchmarking.recipes import initial_stage_run
from tests import test_reference_mask_stage as fixtures
from tests.test_reference_checkpoint_inspection import _plain


class SelectedRecipeInspectionTests(CatalogueTestCase):
    def test_selected_values_are_allowed_but_changed_contracts_are_not(self):
        from seedvision.benchmarking.recipe_catalogue import build_recipe_catalogue
        from seedvision.benchmarking.recipe_matching import _matches_catalogue_with_selected_values
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
