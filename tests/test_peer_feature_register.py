"""Structural checks for the PlantCV/SMPTS operation-level feature register."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
REGISTER_PATH = ROOT / "docs" / "plantcv_smpts_features.json"


class PeerFeatureRegisterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.register = json.loads(REGISTER_PATH.read_text(encoding="utf-8"))

    def test_all_families_and_operation_rows_have_independent_status_and_contracts(self) -> None:
        features = self.register["features"]
        self.assertEqual([f"F{index:02}" for index in range(1, 23)], [item["id"] for item in features])
        operation_ids = []
        functionality = set(self.register["status_vocabulary"]["functionality"])
        conformance = set(self.register["status_vocabulary"]["conformance"])
        science = set(self.register["status_vocabulary"]["science"])
        for feature in features:
            self.assertTrue(feature["operations"], feature["id"])
            self.assertIn(feature["family_status"], functionality, feature["id"])
            for operation in feature["operations"]:
                operation_ids.append(operation["id"])
                self.assertTrue(operation["contract"]["inputs"], operation["id"])
                self.assertTrue(operation["contract"]["outputs"], operation["id"])
                self.assertTrue(operation["source_functions"], operation["id"])
                self.assertTrue(operation["native_mapping"] is not None, operation["id"])
                self.assertIn(operation["reuse_gap"], {"reused_native", "partial", "gap", "unresolved"})
                self.assertIn(operation["status"]["functionality"], functionality, operation["id"])
                self.assertIn(operation["status"]["conformance"], conformance, operation["id"])
                self.assertIn(operation["status"]["science"], science, operation["id"])
                self.assertTrue(operation["acceptance"], operation["id"])
                self.assertIn("existing", operation["tests"], operation["id"])
                self.assertIn("required", operation["tests"], operation["id"])
                self.assertTrue(all(test_id.startswith("planned:") for test_id in operation["tests"]["required"]))
        self.assertEqual(len(operation_ids), len(set(operation_ids)))

    def test_every_existing_test_reference_resolves_without_importing_test_modules(self) -> None:
        parsed_test_functions: dict[str, set[str]] = {}
        for feature in self.register["features"]:
            for operation in feature["operations"]:
                for test_id in operation["tests"]["existing"]:
                    relative_path, separator, test_name = test_id.partition("::")
                    self.assertTrue(separator, test_id)
                    path = ROOT / Path(relative_path)
                    self.assertTrue(path.is_file(), test_id)
                    if relative_path not in parsed_test_functions:
                        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative_path)
                        parsed_test_functions[relative_path] = {
                            node.name
                            for node in ast.walk(tree)
                            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                        }
                    self.assertIn(test_name, parsed_test_functions[relative_path], test_id)

    def test_core_recipe_cannot_omit_required_reference_stages_or_claim_validation(self) -> None:
        self.assertEqual([f"F{index:02}" for index in range(1, 23)], [item["id"] for item in self.register["features"]])
        plantcv = self.register["selected_scope"]["plantcv_core_recipe"]
        joined = " ".join(plantcv["required_stage_groups"]).casefold()
        for required in ("color-card", "threshold", "median", "morphology", "fill", "watershed", "analyze.size", "analyze.color"):
            self.assertIn(required, joined)
        smpts = self.register["selected_scope"]["smpts_core_recipe"]
        smpts_stages = " ".join(smpts["required_stage_groups"]).casefold()
        for required in ("threshold branch a", "threshold branch b", "filtering", "mask combination", "contact screening", "erosion", "extraction"):
            self.assertIn(required, smpts_stages)
        self.assertEqual(self.register["reference_pins"]["smpts_paper"]["original_code"]["status"], "unavailable")
        self.assertFalse(any(
            operation["status"]["science"] == "validated"
            for feature in self.register["features"]
            for operation in feature["operations"]
        ))

    def test_plantcv_source_and_binary_lock_pins_match_canonical_records(self) -> None:
        pin = self.register["reference_pins"]["plantcv"]
        source = json.loads((ROOT / "config" / "peer_reference_sources.json").read_text(encoding="utf-8"))
        self.assertEqual(pin["version"], source["version"])
        self.assertEqual(pin["commit"], source["commit"])
        source_files = {item["path"]: item for item in source["files"]}
        downloaded_root = ROOT / "artifacts" / "reference-sources" / "plantcv-v4.11.3"
        for item in pin["files"]:
            self.assertRegex(item["sha256"], r"^[0-9a-f]{64}$", item["path"])
            self.assertIn(item["path"], source_files, item["path"])
            self.assertEqual(item["sha256"], source_files[item["path"]]["sha256"], item["path"])
            local_source = downloaded_root / Path(item["path"])
            if local_source.is_file():
                self.assertEqual(hashlib.sha256(local_source.read_bytes()).hexdigest(), item["sha256"], item["path"])

        lock_path = ROOT / pin["dependency_lock"]
        lock_bytes = lock_path.read_bytes()
        lock_sha256 = hashlib.sha256(lock_bytes).hexdigest()
        self.assertEqual(pin["dependency_lock_sha256"], lock_sha256)
        self.assertEqual(source["environment"]["lockfile_sha256"], lock_sha256)
        self.assertIn("--only-binary=:all:", lock_bytes.decode("utf-8"))
        packages = {item["name"].casefold(): item for item in source["environment"]["packages"]}
        self.assertEqual(len(packages), 67)
        flyr = packages["flyr"]
        retained_wheel = flyr["retained_artifact"]
        self.assertEqual(flyr["archive_sha256"], retained_wheel["sha256"])
        self.assertIn(f"flyr=={flyr['version']} --hash=sha256:{retained_wheel['sha256']}".encode(), lock_bytes)
        wheel_path = ROOT / Path(retained_wheel["path"])
        if wheel_path.is_file():
            self.assertEqual(hashlib.sha256(wheel_path.read_bytes()).hexdigest(), retained_wheel["sha256"])

    def test_baseline_keeps_runtime_and_raw_checkout_identity_explicit(self) -> None:
        baseline = json.loads((ROOT / "config" / "peer_baseline.json").read_text(encoding="utf-8"))
        self.assertFalse(baseline["scientifically_validated"])
        runtime = baseline["runtime_environment"]
        self.assertEqual(runtime["implementation"], "CPython")
        self.assertTrue(runtime["packages"])
        self.assertIn("torch_cuda_version", runtime)
        self.assertTrue(runtime["device"]["name"])
        checkout = baseline["checkout_identity"]
        self.assertIn("raw", checkout["hash_policy"].casefold())
        self.assertEqual(set(checkout["files"]), set(baseline["implementation"]["files"]))
        for relative, record in checkout["files"].items():
            self.assertRegex(record["checked_out_sha256"], r"^[0-9a-f]{64}$", relative)
            self.assertRegex(record["git_blob_sha256"], r"^[0-9a-f]{64}$", relative)
            self.assertEqual(record["checked_out_sha256"], baseline["implementation"]["files"][relative])

    def test_unambiguous_f08_rectangle_geometry_symbol_resolves(self) -> None:
        operation = next(
            operation
            for feature in self.register["features"] if feature["id"] == "F08"
            for operation in feature["operations"] if operation["id"] == "F08.2"
        )
        mapping, = operation["native_mapping"]
        self.assertEqual(mapping["path"], "seedvision/segmentation/procedural_fit.py")
        tree = ast.parse((ROOT / mapping["path"]).read_text(encoding="utf-8"), filename=mapping["path"])
        self.assertTrue(any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == mapping["symbol"]
            for node in ast.walk(tree)
        ), mapping["symbol"])


if __name__ == "__main__":
    unittest.main()
