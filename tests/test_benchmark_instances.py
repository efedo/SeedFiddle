"""The instance benchmark harness scores source-frame masks and states the reference caveat."""
from __future__ import annotations

from tests.functional_units import functional_units

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load_harness():
    spec = importlib.util.spec_from_file_location("benchmark_instances", ROOT / "scripts" / "benchmark_instances.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _disc(labels, centre, radius, value):
    cv2.circle(labels, centre, radius, int(value), -1)


@functional_units
class BenchmarkInstanceHarnessTests(unittest.TestCase):
    def test_benchmark_harness(self) -> None:
        self.scenario(self.check_scores_methods_and_restricts_to_dish_region)
        self.scenario(self.check_rejects_wrong_frame_and_missing_predictions)

    def _fixture(self, root: Path):
        shape = (120, 160)
        reference = np.zeros(shape, np.uint16)
        for index, centre in enumerate(((40, 40), (80, 40), (60, 80)), start=1):
            _disc(reference, centre, 12, index)
        _disc(reference, (140, 105), 8, 9)  # isolated seed outside the dish
        image = np.full(shape + (3,), 200, np.uint8)
        (root / "images").mkdir()
        cv2.imwrite(str(root / "images" / "IMG_T.JPG"), image)
        references = root / "seed-instance-references"
        references.mkdir()
        cv2.imwrite(str(references / "IMG_T.instances.png"), reference)
        manifest = {"version": 1, "entries": [{
            "image_path": "images/IMG_T.JPG", "image_sha256": sha256((root / "images" / "IMG_T.JPG").read_bytes()).hexdigest(),
            "width": shape[1], "height": shape[0], "mask_path": "IMG_T.instances.png", "coordinate_space": "source",
            "mask_sha256": sha256((references / "IMG_T.instances.png").read_bytes()).hexdigest(),
            "instance_count": 4, "dish_instance_count": 3, "external_reference_count": 1,
            "reviewed": False, "notes": "synthetic"}]}
        (references / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        good = np.zeros(shape, np.uint16)
        for index, centre in enumerate(((41, 40), (80, 41), (60, 80)), start=1):
            _disc(good, centre, 12, index * 7)
        merged = np.zeros(shape, np.uint16)
        _disc(merged, (40, 40), 12, 3)
        _disc(merged, (80, 40), 12, 3)  # one label spanning two reference seeds
        _disc(merged, (60, 80), 12, 5)
        _disc(merged, (140, 105), 8, 6)  # also labels the external seed
        for name, labels in (("good", good), ("merged", merged)):
            (root / name).mkdir()
            cv2.imwrite(str(root / name / "IMG_T.instances.png"), labels)
        (root / "good" / "IMG_T.run.json").write_text(json.dumps({"elapsed_seconds": 1.5, "warnings": ["w"]}), encoding="utf-8")
        regions = root / "regions"
        regions.mkdir()
        (regions / "IMG_T.region.json").write_text(json.dumps(
            {"stem": "IMG_T", "coordinate_space": "source", "polygon_xy": [[5, 5], [115, 5], [115, 115], [5, 115]]}),
            encoding="utf-8")
        return references, regions

    def check_scores_methods_and_restricts_to_dish_region(self) -> None:
        harness = _load_harness()
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            references, regions = self._fixture(root)
            report = harness.run_benchmark(
                references, {"good": root / "good", "merged": root / "merged"}, regions=regions,
                review=["IMG_T"], review_dir=root / "review", image_root=root)
            self.assertIn("reviewed: false", report["caveat"])
            self.assertEqual(report["unreviewed_reference_images"], 1)
            record = report["images"][0]
            self.assertEqual(record["reference_instances_outside_dish"], 1)
            good = record["methods"]["good"]
            self.assertEqual(good["full_frame"]["true_positives"], 3)
            self.assertEqual(good["full_frame"]["false_negatives"], 1)
            self.assertEqual(good["dish_region"]["false_negatives"], 0)
            self.assertEqual(good["dish_region"]["f1"], 1.0)
            self.assertEqual(good["run"], {"elapsed_seconds": 1.5, "warnings": ["w"]})
            merged = record["methods"]["merged"]
            self.assertEqual(merged["full_frame"]["merged_instances"], 1)
            self.assertEqual(merged["full_frame"]["count_error"], -1)
            self.assertEqual(merged["dish_region"]["prediction_instances_outside_dish"], 1)
            self.assertEqual(merged["dish_region"]["predicted_instances"], 2)
            aggregate = report["aggregate"]["good"]["dish_region"]
            self.assertEqual(aggregate["pooled_f1"], 1.0)
            self.assertEqual(aggregate["total_merged_instances"], 0)
            self.assertEqual(report["aggregate"]["merged"]["full_frame"]["total_merged_instances"], 1)
            self.assertEqual(len(report["review_overlays"]), 2)
            overlay = cv2.imread(report["review_overlays"][0])
            self.assertEqual(overlay.shape, (120, 320, 3))
            for method in ("good", "merged"):
                self.assertTrue((root / "review" / f"IMG_T.reference-vs-{method}.jpg").is_file())
            text = json.dumps(report)
            self.assertIn("not accuracy against human ground truth", text)

    def check_rejects_wrong_frame_and_missing_predictions(self) -> None:
        harness = _load_harness()
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            references, regions = self._fixture(root)
            (root / "absent").mkdir()
            report = harness.run_benchmark(references, {"absent": root / "absent"}, regions=regions, image_root=root)
            self.assertEqual(report["images"][0]["methods"]["absent"]["status"], "missing")
            self.assertEqual(report["aggregate"], {"absent": {}})
            (root / "wrong").mkdir()
            cv2.imwrite(str(root / "wrong" / "IMG_T.instances.png"), np.zeros((60, 80), np.uint16))
            with self.assertRaisesRegex(ValueError, "reference frame"):
                harness.run_benchmark(references, {"wrong": root / "wrong"}, image_root=root)
            cv2.imwrite(str(root / "wrong" / "IMG_T.instances.png"), np.zeros((120, 160), np.uint8))
            with self.assertRaisesRegex(ValueError, "uint16"):
                harness.run_benchmark(references, {"wrong": root / "wrong"}, image_root=root)
            with self.assertRaises(SystemExit):
                harness.main(["--predictions", "bad-spec"])
