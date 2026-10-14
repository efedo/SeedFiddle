"""Finite actual public-chain acceptance; expected artifacts are never refreshed."""
from dataclasses import dataclass, replace
from hashlib import sha256
import inspect
import json
from pathlib import Path
import re
import tempfile
import unittest

import cv2
import numpy as np

from seedvision.benchmarking.checkpoints import _canonical_json, _thaw_json, load_checkpoint
from seedvision.benchmarking.checkpoint_inspection import _verify_runner_identity
from seedvision.benchmarking.recipe_catalogue import build_recipe_catalogue
from seedvision.benchmarking.reference_capture_session import capture_reference_path, reference_measurement
from seedvision.benchmarking.reference_runner import PlantCVReferenceRunner
from tests.test_reference_preprocessing import HAS_REFERENCE_ENV, REFERENCE_PYTHON


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "artifacts/public-seed-acceptance-baseline-v1.json"
BASELINE_SHA256 = "5f84f7f8e8d9b71e6f0704a262c6002e4ed382a16cc13511566596b0cf87eb6a"
SCHEMA = "seedfiddle.public-seed-acceptance.v1"
RANDOM_PRODUCTS = frozenset({"palette", "diagnostic", "diagnostic_bgr"})


@dataclass(frozen=True)
class PublicProfile:
    case_id: str
    kind: str
    fixture: str
    polarity: str = "light"
    n_labels: int | None = None


def profiles():
    return (
        PublicProfile("binary-light-default", "binary", "chromatic"),
        PublicProfile("binary-dark-default", "binary", "chromatic", "dark"),
        PublicProfile("mean-even-light", "mean", "chromatic"),
        PublicProfile("otsu-light-default", "otsu", "chromatic"),
        PublicProfile("binary-light-explicit7", "binary", "chromatic", n_labels=7),
        PublicProfile("empty-default", "binary", "empty"),
        PublicProfile("empty-explicit7", "binary", "empty", n_labels=7),
        PublicProfile("gray-color-failure", "binary", "gray_failure"),
    )


def fixture_bytes(name):
    if name == "gray_failure":
        gray = np.full((48, 72), 32, dtype=np.uint8)
        gray[9:38, 14:57] = 205
        gray[24, 36] = 255
        image = np.repeat(gray[..., None], 3, axis=2)
        image[0, 0] = (17, 31, 63)
    else:
        y, x = np.ogrid[:96, :128]
        gray = np.full((96, 128), 35, dtype=np.uint8)
        selected = ((x - 45) ** 2 / 20 ** 2 + (y - 40) ** 2 / 14 ** 2 < 1) | ((x - 77) ** 2 / 19 ** 2 + (y - 41) ** 2 / 13 ** 2 < 1)
        gray[selected] = 210
        gray[10, 10] = 255
        image = np.clip(gray[..., None].astype(np.int16) + [-5, -3, 7], 0, 255).astype(np.uint8)
        if name == "empty":
            image.fill(0)
        elif name != "chromatic":
            raise ValueError("Unknown public acceptance fixture.")
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise AssertionError("Acceptance fixture encoding failed.")
    return encoded.tobytes()


def selected_recipe(profile):
    recipe = build_recipe_catalogue().recipe("plantcv_seed_" + profile.kind)
    values = {
        "filter.median_blur": {"ksize": 3},
        "threshold.binary": {"threshold": 128 if profile.fixture == "gray_failure" else 100, "object_type": profile.polarity},
        "threshold.mean": {"ksize": 8, "offset": -2, "object_type": profile.polarity},
        "threshold.otsu": {"object_type": profile.polarity},
        "morphology.fill": {"size": 5 if profile.fixture == "gray_failure" else 20},
        "morphology.erode": {"ksize": 3, "iterations": 1},
        "morphology.dilate": {"ksize": 3, "iterations": 1},
        "reference.internal.watershed.local_maxima": {"distance": 10 if profile.fixture == "gray_failure" else 8},
        "analysis.size": {"n_labels": profile.n_labels},
        "analysis.color": {"n_labels": profile.n_labels, "colorspaces": "all"},
    }
    return replace(recipe, stages=tuple(replace(stage, parameters=tuple(
        replace(parameter, value=values.get(stage.operation_id, {}).get(parameter.parameter_id, parameter.value))
        for parameter in stage.parameters)) for stage in recipe.stages))


def array_descriptor(array, *, pixels=True):
    value = {"shape": list(array.shape), "dtype": array.dtype.str, "nbytes": array.nbytes}
    if pixels:
        value["sha256"] = sha256(array.tobytes(order="C")).hexdigest()
    return value


def checkpoint_semantics(checkpoint, recipe):
    metadata = _thaw_json(checkpoint.metadata)
    if "runner_identity" in metadata:
        _verify_runner_identity(checkpoint.metadata, recipe)
    identity = checkpoint.identity.to_dict()
    # Published stage addresses include measured runtimes and per-call UUIDs.
    # Their actual lineage is verified separately, never made a repeatability hash.
    identity.pop("upstream_checkpoint_sha256")
    effective = metadata.get("effective_parameters", {})
    if "capture" in effective:
        capture = effective["capture"]
        if capture["status"] != "available":
            raise AssertionError("Required actual watershed locals are unavailable.")
        if not re.fullmatch(r"[0-9a-f]{32}", capture["call_id"]):
            raise AssertionError("Actual watershed RPC identity is invalid.")
        np.testing.assert_array_equal(checkpoint.arrays["diagnostic"], checkpoint.arrays["diagnostic_bgr"][..., ::-1])
        capture.pop("call_id")
        for name in RANDOM_PRODUCTS:
            capture["artifacts"][name].pop("sha256")
        # Public palette membership is repeatable; its shuffled order is not.
        capture["palette_sorted_RGB_triples"] = sorted(checkpoint.arrays["palette"].tolist())
    return {
        "identity_without_transient_upstream_address": identity,
        "execution_status": checkpoint.execution.status.value,
        "measurement": checkpoint.measurement.to_dict(),
        "raster_roles": {name: role.to_dict() for name, role in checkpoint.raster_roles.items()},
        "arrays": {name: array_descriptor(array, pixels=not ("capture" in effective and name in RANDOM_PRODUCTS))
                   for name, array in sorted(checkpoint.arrays.items())},
        "requested_parameters": metadata.get("requested_parameters", metadata.get("parameters")),
        "parameters_passed": metadata.get("parameters_passed", metadata.get("parameters")),
        "effective_parameters": effective,
        "observations": metadata.get("observations"),
        "visualizations": metadata.get("visualizations"),
        "channel_semantics": metadata.get("channel_semantics"),
        "array_origins": metadata.get("array_origins"),
        "preparation": metadata.get("preparation"),
        "actual_public_reader": metadata.get("actual_public_reader"),
    }


def project_session(profile, session, store):
    outcome = session.outcome
    if outcome is None or len(outcome.calls) != 3:
        raise AssertionError("An actual new source-chain attempt is required.")
    record = _thaw_json(outcome.record)
    ids = [call.call_id for call in outcome.calls]
    if len(set(ids)) != 3 or any(not re.fullmatch(r"[0-9a-f]{32}", value) for value in ids):
        raise AssertionError("Each coordinator call requires its own distinct ID.")
    if bool(outcome.manifest_sha256) != record["complete"] or bool(session.verified) != record["complete"]:
        raise AssertionError("Partial or empty runs cannot expose a complete manifest.")
    recipe = selected_recipe(profile)
    if recipe != session.recipe or recipe.identity.to_dict() != record["recipe_identity"]:
        raise AssertionError("Selected recipe differs from the actual source attempt.")
    semantic = {"case_id": profile.case_id, "fixture": profile.fixture, "status": outcome.status,
        "complete_manifest": record["complete"], "recipe_identity": record["recipe_identity"],
        "source_sha256": record["source_sha256"], "coordinate_frame": record["coordinate_frame"],
        "parameters": {stage.stage_id: recipe.parameters_for_runner(stage.stage_id) for stage in recipe.stages},
        "inputs": {}, "preprocessing": [], "source_calls": [], "label_facts": record["label_facts"],
        "runner_requests_after_preprocessing": record["runner_request_count_after_preprocessing"],
        "verified_public_results_after_preprocessing": record["verified_public_result_count_after_preprocessing"],
        "scientifically_validated": False}
    provenance = {"case_id": profile.case_id, "manifest_sha256": outcome.manifest_sha256,
                  "checkpoint_references": {}, "source_calls": [call.to_dict() for call in outcome.calls]}
    for name, binding in record["inputs"].items():
        checkpoint = load_checkpoint(Path(store) / "inputs", binding["checkpoint_sha256"])
        if checkpoint.identity.to_dict() != binding["identity"] or checkpoint.measurement.to_dict() != binding["measurement"]:
            raise AssertionError("Prepared checkpoint binding differs.")
        semantic["inputs"][name] = checkpoint_semantics(checkpoint, recipe)
        provenance["checkpoint_references"]["input." + name] = checkpoint.checkpoint_sha256
    measurement = reference_measurement("public_reference_chain_output")
    for index, stage in enumerate(outcome.preprocessing.stages):
        reference = stage.checkpoint_reference
        entry = {"operation": stage.stage.operation_id, "status": stage.status.value, "reason": stage.run_record.reason}
        if reference is not None:
            checkpoint = load_checkpoint(store, reference.checkpoint_sha256,
                                         expected_identity=reference.identity, expected_measurement=measurement)
            entry["checkpoint"] = checkpoint_semantics(checkpoint, recipe)
            provenance["checkpoint_references"][f"preprocessing.{index}"] = checkpoint.checkpoint_sha256
        semantic["preprocessing"].append(entry)
    labels = None
    for call in outcome.calls:
        details = _thaw_json(call.details)
        entry = {"operation": call.operation, "status": call.status, "invoked": call.invoked, "reason": call.reason,
                 "requested_parameters": details.get("requested_parameters"), "parameters_passed": details.get("parameters_passed"),
                 "n_labels_rule": details.get("n_labels_rule"), "effective_n_labels": details.get("effective_n_labels")}
        reference = call.checkpoint_reference
        if reference is not None:
            checkpoint = load_checkpoint(store, reference.checkpoint_sha256,
                                         expected_identity=reference.identity, expected_measurement=measurement)
            if (checkpoint.metadata["call_id"] != call.call_id or checkpoint.metadata["operation"] != call.operation
                    or checkpoint.metadata["source_sha256"] != record["source_sha256"]):
                raise AssertionError("Public checkpoint call/source binding differs.")
            entry["checkpoint"] = checkpoint_semantics(checkpoint, recipe)
            provenance["checkpoint_references"][call.operation] = checkpoint.checkpoint_sha256
            if call.operation == "segmentation.watershed.capture":
                labels = checkpoint.arrays["labels"].copy()
                if int(labels.max()) != record["label_facts"]["maximum_label_id"]:
                    raise AssertionError("Actual maximum label differs.")
            else:
                np.testing.assert_array_equal(checkpoint.arrays["labels_context"], labels)
                expected_n = int(labels.max()) if profile.n_labels is None else profile.n_labels
                if checkpoint.metadata["parameters_passed"]["n_labels"] != expected_n:
                    raise AssertionError("Actual public loop bound differs from authored selection.")
        semantic["source_calls"].append(entry)
    provenance["retained_reader_numeric_bytes"] = session.verified.retained_array_bytes if session.verified else None
    return semantic, provenance


def contract_hash():
    source = "\n".join(inspect.getsource(function) for function in (
        profiles, fixture_bytes, selected_recipe, array_descriptor, checkpoint_semantics, project_session))
    return sha256(source.encode("utf-8")).hexdigest()


def capture_matrix(root, *, capture=None, progress=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    cases, provenance = [], []
    for profile in profiles():
        source = root / (profile.fixture + ".png")
        raw = fixture_bytes(profile.fixture)
        source.write_bytes(raw)
        store = root / profile.case_id
        if progress:
            progress(profile.case_id, "started")
        session = (capture(profile, source, store) if capture else capture_reference_path(
            selected_recipe(profile), source, store, PlantCVReferenceRunner(REFERENCE_PYTHON)))
        semantic, run = project_session(profile, session, store)
        if semantic["source_sha256"] != sha256(raw).hexdigest():
            raise AssertionError("Encoded source bytes differ from the prepared chain identity.")
        cases.append(semantic)
        provenance.append(run)
        if progress:
            progress(profile.case_id, semantic["status"])
    record = {"schema": SCHEMA, "contract_sha256": contract_hash(), "scientifically_validated": False,
              "case_count": len(cases), "cases": cases,
              "repeatability_boundary": "Exact deterministic numeric arrays/observations/parameters; per-run UUID/runtime/checkpoint addresses and watershed shuffled palette/display ordering excluded. Per-call BGR/RGB and palette membership verified."}
    return record, provenance


def read_baseline(path=BASELINE):
    raw = Path(path).read_bytes()
    if sha256(raw).hexdigest() != BASELINE_SHA256:
        raise ValueError("The fixed public acceptance baseline is missing or changed; never refresh expected evidence.")
    record = json.loads(raw)
    if record.get("schema") != SCHEMA or record.get("contract_sha256") != contract_hash():
        raise ValueError("Public acceptance fixture/projection contract differs from its frozen record.")
    return record


class PublicSeedAcceptanceInventoryTests(unittest.TestCase):
    def test_frozen_case_inventory_source_recipe_and_outcome_claims(self):
        record = read_baseline()
        self.assertEqual(record["case_count"], 8)
        self.assertEqual([c["case_id"] for c in record["cases"]], [p.case_id for p in profiles()])
        self.assertFalse(record["scientifically_validated"])
        for profile, case in zip(profiles(), record["cases"], strict=True):
            self.assertEqual(case["source_sha256"], sha256(fixture_bytes(profile.fixture)).hexdigest())
            self.assertEqual(case["recipe_identity"], selected_recipe(profile).identity.to_dict())
            self.assertEqual(len(case["preprocessing"]), 5)
            self.assertEqual(len(case["source_calls"]), 3)
            self.assertEqual(case["complete_manifest"], case["status"] == "executed")
        empty = next(c for c in record["cases"] if c["case_id"] == "empty-default")
        self.assertEqual(empty["status"], "empty")
        self.assertEqual([c["invoked"] for c in empty["source_calls"]], [True, False, False])
        self.assertEqual([empty["runner_requests_after_preprocessing"], empty["verified_public_results_after_preprocessing"]], [1, 1])
        for case in record["cases"][:4]:
            self.assertEqual(case["status"], "executed")
            self.assertEqual([case["runner_requests_after_preprocessing"], case["verified_public_results_after_preprocessing"]], [3, 3])
        mean = record["cases"][2]["preprocessing"][1]["checkpoint"]
        self.assertEqual(mean["requested_parameters"]["ksize"], 8)
        self.assertEqual(mean["effective_parameters"]["ksize"], 9)
        for case in (record["cases"][4], record["cases"][6]):
            self.assertEqual(case["status"], "partial")
            self.assertEqual([case["runner_requests_after_preprocessing"], case["verified_public_results_after_preprocessing"]], [3, 2])
            for call in case["source_calls"][1:]:
                self.assertTrue(call["invoked"])
                self.assertEqual(call["parameters_passed"]["n_labels"], 7)
            self.assertEqual(case["source_calls"][1]["status"], "executed")
            self.assertEqual(case["source_calls"][2]["status"], "failed")
            self.assertIn("UnboundLocalError", case["source_calls"][2]["reason"])
        failure = next(c for c in record["cases"] if c["case_id"] == "gray-color-failure")
        self.assertEqual(failure["status"], "partial")
        self.assertEqual(failure["source_calls"][2]["status"], "failed")
        self.assertIn("numpy.float16", failure["source_calls"][2]["reason"])

    def test_changed_expected_bytes_are_rejected_without_refresh(self):
        raw = BASELINE.read_bytes()
        with tempfile.TemporaryDirectory(prefix="public-baseline-reject-") as directory:
            path = Path(directory) / "changed.json"
            path.write_bytes(raw + b" ")
            with self.assertRaisesRegex(ValueError, "missing or changed"):
                read_baseline(path)
        self.assertEqual(BASELINE.read_bytes(), raw)

    def _assert_projection_sensitivity(self, checkpoint, recipe):
        # Controlled copies test only normalization. They are never published,
        # passed off as verified source checkpoints, or counted as public calls.
        expected = checkpoint_semantics(checkpoint, recipe)
        arrays = dict(checkpoint.arrays)
        distance = arrays["distance"].copy()
        distance[0, 0] += np.float32(1)
        arrays["distance"] = distance
        self.assertNotEqual(checkpoint_semantics(replace(checkpoint, arrays=arrays), recipe), expected)
        arrays = dict(checkpoint.arrays)
        palette = arrays["palette"].copy()
        palette[0, 0] ^= np.uint8(1)
        arrays["palette"] = palette
        self.assertNotEqual(checkpoint_semantics(replace(checkpoint, arrays=arrays), recipe), expected)
        arrays = dict(checkpoint.arrays)
        arrays["palette"] = arrays["palette"][::-1].copy()
        self.assertEqual(checkpoint_semantics(replace(checkpoint, arrays=arrays), recipe), expected)
        arrays = dict(checkpoint.arrays)
        diagnostic = arrays["diagnostic"].copy()
        diagnostic[0, 0, 0] ^= np.uint8(1)
        arrays["diagnostic"] = diagnostic
        with self.assertRaises(AssertionError):
            checkpoint_semantics(replace(checkpoint, arrays=arrays), recipe)
        metadata = _thaw_json(checkpoint.metadata)
        metadata["call_id"] = "f" * 32
        metadata["effective_parameters"]["capture"]["call_id"] = "e" * 32
        metadata["wall_ns"] = 1
        changed = replace(checkpoint, metadata=metadata,
                          identity=replace(checkpoint.identity, upstream_checkpoint_sha256="0" * 64))
        self.assertEqual(checkpoint_semantics(changed, recipe), expected)

    @unittest.skipUnless(HAS_REFERENCE_ENV, "Pinned public PlantCV environment is required.")
    def test_all_selected_public_chains_match_frozen_numeric_and_outcome_record(self):
        expected = read_baseline()  # Fail closed before any call; no regeneration.
        def capture(profile, source, store):
            session = capture_reference_path(selected_recipe(profile), source, store, PlantCVReferenceRunner(REFERENCE_PYTHON))
            if profile.case_id == "binary-light-default":
                self._assert_projection_sensitivity(session.verified.checkpoints["watershed"], session.recipe)
            return session
        with tempfile.TemporaryDirectory(prefix="public-seed-acceptance-") as directory:
            actual, provenance = capture_matrix(directory, capture=capture)
        self.assertEqual(_canonical_json(actual, name="fresh public record"), _canonical_json(expected, name="fixed public record"))
        self.assertEqual(len(provenance), 8)
        for run in provenance:
            ids = [call["call_id"] for call in run["source_calls"]]
            self.assertEqual(len(set(ids)), 3)


if __name__ == "__main__":
    unittest.main()
