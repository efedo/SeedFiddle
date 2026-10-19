from __future__ import annotations

from tests.functional_units import functional_units

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import threading
import unittest
from tests.catalogue_fixtures import CatalogueTestCase
from unittest.mock import patch

import numpy as np

from seedvision.benchmarking.checkpoints import (
    ExecutionRecord, ExecutionStatus, RasterRole, load_checkpoint, publish_checkpoint,
)
from seedvision.benchmarking.reference_runner import PlantCVReferenceRunner, ReferenceExecutionError
from seedvision.benchmarking.reference_seed_chain import (
    ReferenceRasterInput, ReferenceSeedChainError, run_reference_seed_chain,
)
from tests.test_reference_preprocessing import (
    HAS_REFERENCE_ENV, _ReferencePreprocessingFixtures, _gray_fixture, _measurement, _selected_recipe,
)


@unittest.skipUnless(HAS_REFERENCE_ENV, "Pinned public PlantCV environment is required.")
@functional_units
class ReferenceSeedChainTests(CatalogueTestCase):
    """One actual complete chain plus bounded fault injection at its API edges."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.fixtures = _ReferencePreprocessingFixtures()
        cls.fixtures.setUp()
        cls.source_results = {}
        cls.actual_operations = []
        cls.actual_arguments = cls._arguments()
        original = PlantCVReferenceRunner.run

        def observe(runner, operation, arrays, parameters=None, **kwargs):
            result = original(runner, operation, arrays, parameters, **kwargs)
            cls.source_results[operation] = result
            cls.actual_operations.append(operation)
            return result

        with patch.object(PlantCVReferenceRunner, "run", observe):
            cls.actual_outcome = run_reference_seed_chain(**cls.actual_arguments)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.fixtures.tearDown()

    @classmethod
    def _arguments(cls, recipe=None, *, empty=False):
        old = cls.fixtures._arguments(recipe, empty=empty)
        gray_identity = old["expected_input_identity"]
        rgb_identity = replace(gray_identity, stage_id="prepared_rgb_input", image_semantics={
            **dict(gray_identity.image_semantics), "color_order": "RGB",
            "channel_conversion": "caller_supplied_rgb",
        })
        gray = _gray_fixture(old["expected_coordinate_frame"].shape, empty=empty)
        rgb = np.stack((gray, np.maximum(gray, 61), np.minimum(gray, 173)), axis=-1).astype(np.uint8)
        rgb_role = RasterRole("image", np.dtype(np.uint8).str, "HWC", (0, 255))
        rgb_measurement = _measurement("prepared_rgb_independent_definition")
        reference = publish_checkpoint(
            old["input_root"], rgb_identity, {"rgb": rgb},
            execution=ExecutionRecord(ExecutionStatus.AVAILABLE), measurement=rgb_measurement,
            raster_roles={"rgb": rgb_role}, metadata={"fixture_only": True}, allow_cpu_transfer=True,
        )
        return {
            "gray": ReferenceRasterInput(old["input_root"], old["input_checkpoint_sha256"], gray_identity,
                                          old["expected_input_measurement"], "gray", old["input_roles"]["gray"]),
            "rgb": ReferenceRasterInput(old["input_root"], reference.checkpoint_sha256, rgb_identity,
                                         rgb_measurement, "rgb", rgb_role),
            **{key: old[key] for key in ("recipe", "runner", "output_root", "expected_source_sha256",
                                       "expected_coordinate_frame", "expected_recipe_identity", "output_measurement", "allow_cpu_transfer")},
        }

    def _fake(self, *, failure=None, capture_unavailable=False, empty=False, cancel_after=None, event=None, malformed=None):
        calls = []

        def run(runner, operation, arrays, parameters=None, **kwargs):
            calls.append((operation, dict(parameters or {})))
            if operation == failure:
                raise ReferenceExecutionError("Injected source-call failure.")
            original = self.source_results[operation]
            result = replace(original,
                arrays={name: value.copy() for name, value in original.arrays.items()},
                effective_parameters=deepcopy(dict(original.effective_parameters)),
                observations=deepcopy(dict(original.observations)), visualizations=deepcopy(dict(original.visualizations)),
                identity=deepcopy(dict(original.identity)))
            effective = dict(result.effective_parameters) | dict(parameters or {})
            if empty and operation.startswith("threshold."):
                result = replace(result, arrays={"mask": np.zeros_like(arrays["gray"])})
            elif empty and operation.startswith("morphology."):
                result = replace(result, arrays={"mask": np.zeros_like(arrays["gray"] if "gray" in arrays else arrays["mask"])})
            if operation == "segmentation.watershed.capture":
                if empty:
                    shape = arrays["mask"].shape
                    products = {name: np.zeros_like(value) for name, value in result.arrays.items()}
                    products["coordinates"] = np.empty((0, 2), dtype="<i8")
                    products["palette"] = np.zeros((1, 3), dtype=np.uint8)
                    products["diagnostic_bgr"] = arrays["rgb"][..., ::-1].copy()
                    products["diagnostic"] = arrays["rgb"].copy()
                    capture = effective["capture"]
                    from seedvision.benchmarking.reference_watershed_capture import EMPTY_COORDINATES
                    capture.update(empty_coordinates=dict(EMPTY_COORDINATES), estimated_object_count=0,
                                   positive_marker_ids=[], positive_label_ids=[], label_max=0)
                    for name, value in products.items():
                        capture["artifacts"][name].update(shape=list(value.shape), dtype=value.dtype.str,
                                                        nbytes=value.nbytes, sha256=sha256(value.tobytes()).hexdigest())
                    result = replace(result, arrays=products)
                if capture_unavailable:
                    effective["capture"] = {"schema": "seedfiddle.public-watershed-capture.v1",
                        "status": "unavailable", "reason": "Injected missing local.",
                        "call_id": "injected-worker-call", "scientifically_validated": False}
                    result = replace(result, arrays={"labels": result.arrays["labels"]})
                if malformed == "rgb_adapter":
                    result.arrays["diagnostic"][0, 0, 0] ^= np.uint8(1)
            if malformed == "parameters" and operation == "analysis.size":
                effective["n_labels"] += 1
            if event is not None and cancel_after == operation:
                event.set()
            return replace(result, effective_parameters=effective)

        return run, calls

    def check_test_actual_eight_public_calls_publish_bound_reloadable_chain(self):
        outcome = self.actual_outcome
        self.assertEqual(outcome.status, "executed", [call.reason for call in outcome.calls])
        self.assertEqual(len(self.actual_operations), 8)
        self.assertEqual(self.actual_operations[-3:], ["segmentation.watershed.capture", "analysis.size", "analysis.color"])
        raw = outcome.manifest_path.read_bytes()
        self.assertEqual(sha256(raw).hexdigest(), outcome.manifest_sha256)
        record = json.loads(raw)
        self.assertTrue(record["complete"])
        self.assertFalse(record["scientifically_validated"])
        self.assertEqual(record["runner_request_count_after_preprocessing"], 3)
        self.assertEqual(record["verified_public_result_count_after_preprocessing"], 3)
        ws = outcome.calls[0].checkpoint_reference
        loaded = load_checkpoint(self.actual_arguments["output_root"], ws.checkpoint_sha256,
                                 expected_identity=ws.identity, expected_measurement=self.actual_arguments["output_measurement"])
        labels_digest = sha256(loaded.arrays["labels"].tobytes()).hexdigest()
        np.testing.assert_array_equal(loaded.arrays["diagnostic"], loaded.arrays["diagnostic_bgr"][..., ::-1])
        for call in outcome.calls[1:]:
            reference = call.checkpoint_reference
            analysis = load_checkpoint(self.actual_arguments["output_root"], reference.checkpoint_sha256,
                                       expected_identity=reference.identity)
            self.assertEqual(analysis.metadata["input_array_sha256"]["labels"], labels_digest)
            self.assertEqual(analysis.metadata["array_origins"]["labels_context"], "input_context_not_computed_output")
            self.assertEqual(call.details["effective_n_labels"], int(loaded.arrays["labels"].max()))
            self.assertEqual(call.details["n_labels_rule"], "int(public_labels.max())")
            np.testing.assert_array_equal(analysis.arrays["labels_context"], loaded.arrays["labels"])

    def check_test_source_frame_channel_and_recipe_mismatches_reject_before_calls(self):
        arguments = self._arguments()
        rgb = arguments["rgb"]
        mismatches = [
            replace(rgb, identity=replace(rgb.identity, source_sha256="f" * 64)),
            replace(rgb, identity=replace(rgb.identity, coordinate_frame=replace(rgb.identity.coordinate_frame, frame_id="wrong_frame"))),
            replace(rgb, role=RasterRole("raster", "|u1", "HWC")),
            replace(rgb, array_name="absent"),
        ]
        with patch.object(PlantCVReferenceRunner, "run") as public:
            for mismatch in mismatches:
                with self.subTest(mismatch=mismatch), self.assertRaises(ValueError):
                    run_reference_seed_chain(**(arguments | {"rgb": mismatch}))
            with self.assertRaises(ReferenceSeedChainError):
                run_reference_seed_chain(**(arguments | {"expected_recipe_identity": replace(arguments["recipe"].identity, settings_sha256="e" * 64)}))
            public.assert_not_called()





    def check_test_cancel_after_each_call_withholds_later_calls_and_manifest(self):
        for boundary in ("filter.median_blur", "segmentation.watershed.capture", "analysis.size", "analysis.color"):
            with self.subTest(boundary=boundary):
                arguments = self._arguments()
                event = threading.Event()
                fake, calls = self._fake(cancel_after=boundary, event=event)
                with patch.object(PlantCVReferenceRunner, "run", fake):
                    outcome = run_reference_seed_chain(**arguments, cancel=event)
                self.assertEqual(outcome.status, "cancelled")
                self.assertEqual(calls[-1][0], boundary)
                self.assertIsNone(outcome.manifest_path)
                self.assertFalse((arguments["output_root"] / "chains").exists())


    def check_test_cancel_at_manifest_boundary_retains_checkpoints_without_complete_manifest(self):
        arguments = self._arguments()
        event = threading.Event()
        fake, _ = self._fake()
        from seedvision.benchmarking.reference_seed_chain import _manifest
        def cancel_manifest(root, record, limits, cancel):
            event.set()
            return _manifest(root, record, limits, cancel)
        with patch.object(PlantCVReferenceRunner, "run", fake), patch("seedvision.benchmarking.reference_seed_chain._manifest", cancel_manifest):
            outcome = run_reference_seed_chain(**arguments, cancel=event)
        self.assertEqual(outcome.status, "cancelled")
        self.assertFalse(outcome.record["complete"])
        self.assertTrue(all(call.checkpoint_reference is not None for call in outcome.calls))
        self.assertIsNone(outcome.manifest_path)
        self.assertEqual(list((arguments["output_root"] / "chains").iterdir()), [])



    def test_workflow_lifecycle(self):
        """1 related ReferenceSeedChainTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_actual_eight_public_calls_publish_bound_reloadable_chain)

    def test_validation_and_recovery(self):
        """3 related ReferenceSeedChainTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_source_frame_channel_and_recipe_mismatches_reject_before_calls)
        self.scenario(self.check_test_cancel_after_each_call_withholds_later_calls_and_manifest)
        self.scenario(self.check_test_cancel_at_manifest_boundary_retains_checkpoints_without_complete_manifest)


if __name__ == "__main__":
    unittest.main()
