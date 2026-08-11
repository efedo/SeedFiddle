from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from seedvision.learning.decode import (
    HybridDecodeSettings,
    StarDistDecodeSettings,
    UNetWatershedSettings,
    decode_stardist,
    decode_unet_stardist_hybrid,
    decode_unet_watershed,
)
from seedvision.learning.data import (
    LearningManifest,
    SeedTileDataset,
    audit_manifest,
    export_learning_sample,
    rotation_ray_mapping,
)
from seedvision.learning.features import colour_only_feature_spec
from seedvision.learning.losses import multi_head_unet_loss, stardist_loss
from seedvision.learning.metrics import evaluate_binary_probability, evaluate_instances
from seedvision.learning.models import MultiHeadSeedUNet, SeedStarDist2D
from seedvision.learning.targets import build_dense_targets, build_stardist_targets


def _scene(shape=(112, 144)):
    labels = np.zeros(shape, dtype=np.int32)
    for identifier, centre, axes, angle in (
        (1, (38, 56), (23, 16), -12),
        (2, (72, 55), (22, 16), 8),
        (3, (107, 57), (23, 17), -6),
    ):
        cv2.ellipse(labels, centre, axes, angle, 0, 360, identifier, -1)
    return labels


def _logit(values):
    values = np.clip(values, 1e-4, 1.0 - 1e-4)
    return np.log(values / (1.0 - values)).astype(np.float32)


class LearnedTargetTests(unittest.TestCase):
    def test_dense_targets_keep_missing_pattern_annotations_unknown(self) -> None:
        labels = _scene()
        targets = build_dense_targets(labels)
        self.assertEqual(targets.interior.shape, labels.shape)
        self.assertGreater(float(targets.physical_boundary.sum()), 0.0)
        self.assertGreater(float(targets.centre.max()), 0.99)
        self.assertGreater(float(targets.distance.max()), 0.99)
        self.assertEqual(float(targets.pattern_boundary.sum()), 0.0)
        self.assertEqual(float(targets.pattern_valid.sum()), 0.0)

        pattern = np.zeros_like(labels, dtype=np.uint8)
        pattern[45:68, 37:40] = 1
        reviewed = labels > 0
        annotated = build_dense_targets(
            labels, pattern_boundary=pattern, pattern_valid=reviewed
        )
        self.assertGreater(float(annotated.pattern_boundary.sum()), 0.0)
        self.assertEqual(
            int(annotated.pattern_valid.sum()), int(np.count_nonzero(reviewed))
        )

    def test_stardist_targets_have_radial_geometry_at_seed_centres(self) -> None:
        labels = np.zeros((65, 65), dtype=np.int32)
        cv2.circle(labels, (32, 32), 15, 1, -1)
        targets = build_stardist_targets(labels, ray_count=16, maximum_distance=32)
        centre_distances = targets.radial_distances[:, 32, 32]
        self.assertTrue(np.all(targets.radial_valid[:, 32, 32] > 0))
        self.assertLess(float(np.max(np.abs(centre_distances - 15.5))), 1.1)
        self.assertGreater(float(targets.object_probability[32, 32]), 0.99)

    def test_quarter_turn_maps_new_rays_to_correct_old_directions(self) -> None:
        mapping = rotation_ray_mapping(16, 1)
        self.assertEqual(int(mapping[0]), 4)
        self.assertEqual(int(mapping[12]), 0)


class LearningDataTests(unittest.TestCase):
    def test_exported_human_sample_round_trips_and_builds_cached_targets(self) -> None:
        labels = _scene((80, 96)).astype(np.uint16)
        image = np.full((*labels.shape, 3), 180, dtype=np.uint8)
        spec = colour_only_feature_spec(include_species_planes=False)
        features = np.zeros((spec.input_channels, *labels.shape), dtype=np.float32)
        features[3] = 1.0
        pattern = np.zeros_like(labels, dtype=np.uint8)
        pattern[35:45, 20:40] = labels[35:45, 20:40] > 0
        pattern_valid = np.uint8(labels > 0)
        with TemporaryDirectory() as temporary:
            manifest_path = Path(temporary) / "manifest.json"
            export_learning_sample(
                manifest_path,
                dataset_id="unit-test",
                feature_spec=spec,
                identifier="capture 01",
                features=features,
                labels=labels,
                image_bgr=image,
                species="soybean",
                group="lot-a/capture-1",
                split="train",
                reviewed=True,
                pattern_boundary=pattern,
                pattern_valid=pattern_valid,
                annotation_author="Test reviewer",
                annotation_revision="1",
            )
            # A complete manifest needs validation data as well; use a distinct
            # capture group so leakage auditing remains meaningful.
            export_learning_sample(
                manifest_path,
                dataset_id="unit-test",
                feature_spec=spec,
                identifier="capture 02",
                features=features,
                labels=labels,
                image_bgr=image,
                species="soybean",
                group="lot-b/capture-2",
                split="validation",
                reviewed=True,
                annotation_author="Test reviewer",
            )
            manifest = LearningManifest.load(manifest_path)
            self.assertEqual(len(manifest.samples), 2)
            audit = audit_manifest(manifest_path)
            self.assertTrue(audit["valid"], audit["errors"])
            dataset = SeedTileDataset(
                manifest_path,
                split="train",
                family="stardist",
                tile_size=64,
                tiles_per_sample=1,
                ray_count=16,
            )
            item = dataset[0]
            self.assertEqual(tuple(item["features"].shape), (4, 64, 64))
            cache_files = tuple((Path(temporary) / ".seedfiddle-cache").glob("*.npz"))
            self.assertEqual(len(cache_files), 1)


class LearnedModelTests(unittest.TestCase):
    def test_models_emit_expected_full_resolution_heads_and_backpropagate(self) -> None:
        import torch

        values = torch.randn(2, 7, 64, 80)
        labels = np.stack((_scene((64, 80)), _scene((64, 80))), axis=0)
        dense = [build_dense_targets(item) for item in labels]
        targets = {
            "interior": torch.from_numpy(np.stack([item.interior for item in dense]))[:, None],
            "physical_boundary": torch.from_numpy(
                np.stack([item.physical_boundary for item in dense])
            )[:, None],
            "pattern_boundary": torch.from_numpy(
                np.stack([item.pattern_boundary for item in dense])
            )[:, None],
            "pattern_valid": torch.from_numpy(
                np.stack([item.pattern_valid for item in dense])
            )[:, None],
            "centre": torch.from_numpy(np.stack([item.centre for item in dense]))[:, None],
            "distance": torch.from_numpy(np.stack([item.distance for item in dense]))[:, None],
            "valid": torch.from_numpy(np.stack([item.valid for item in dense]))[:, None],
        }
        unet = MultiHeadSeedUNet(7, base_channels=8, depth=2)
        outputs = unet(values)
        self.assertEqual(outputs["interior_logits"].shape, (2, 1, 64, 80))
        self.assertEqual(outputs["uncertainty_logits"].shape, (2, 5, 64, 80))
        loss, terms = multi_head_unet_loss(outputs, targets)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("pattern_boundary", terms)
        loss.backward()

        star = SeedStarDist2D(7, ray_count=16, base_channels=8, depth=2)
        star_outputs = star(values)
        self.assertEqual(star_outputs["radial_distances"].shape, (2, 16, 64, 80))
        star_target = build_stardist_targets(labels[0], ray_count=16, maximum_distance=96)
        star_targets = {
            "object_probability": torch.from_numpy(star_target.object_probability)[None, None].repeat(2, 1, 1, 1),
            "radial_distances": torch.from_numpy(star_target.radial_distances)[None].repeat(2, 1, 1, 1),
            "radial_valid": torch.from_numpy(star_target.radial_valid)[None].repeat(2, 1, 1, 1),
            "valid": torch.from_numpy(star_target.valid)[None, None].repeat(2, 1, 1, 1),
        }
        star_total, _ = stardist_loss(star_outputs, star_targets)
        self.assertTrue(torch.isfinite(star_total))
        star_total.backward()


class LearnedDecoderTests(unittest.TestCase):
    def test_pattern_aware_watershed_decodes_ideal_touching_instances(self) -> None:
        import torch

        labels = _scene()
        pattern = np.zeros_like(labels, dtype=np.uint8)
        pattern[42:70, 31:45] = labels[42:70, 31:45] == 1
        targets = build_dense_targets(
            labels, pattern_boundary=pattern, pattern_valid=labels > 0
        )
        outputs = {
            "interior_logits": torch.from_numpy(_logit(targets.interior))[None, None],
            "physical_boundary_logits": torch.from_numpy(
                _logit(np.clip(targets.physical_boundary, 0.001, 0.999))
            )[None, None],
            "pattern_boundary_logits": torch.from_numpy(
                _logit(np.clip(targets.pattern_boundary, 0.001, 0.999))
            )[None, None],
            "centre_logits": torch.from_numpy(_logit(np.clip(targets.centre, 0.001, 0.999)))[None, None],
            "distance": torch.from_numpy(targets.distance)[None, None],
            "uncertainty_logits": torch.full((1, 5, *labels.shape), -5.0),
        }
        result = decode_unet_watershed(
            outputs,
            settings=UNetWatershedSettings(
                centre_threshold=0.25,
                centre_minimum_separation_px=18.0,
                minimum_instance_area_px=100,
            ),
        )
        self.assertEqual(result.count, 3)
        metrics = evaluate_instances(labels, result.labels, iou_threshold=0.50)
        self.assertEqual(metrics.true_positives, 3)
        self.assertGreater(metrics.f1, 0.99)

    def test_stardist_decodes_ideal_radial_targets(self) -> None:
        import torch

        labels = _scene()
        targets = build_stardist_targets(labels, ray_count=32, maximum_distance=70)
        outputs = {
            "object_logits": torch.from_numpy(
                _logit(np.clip(targets.object_probability, 0.001, 0.999))
            )[None, None],
            "radial_distances": torch.from_numpy(targets.radial_distances)[None],
            "radial_uncertainty_logits": torch.full((1, 1, *labels.shape), -5.0),
        }
        result = decode_stardist(
            outputs,
            settings=StarDistDecodeSettings(
                object_threshold=0.40,
                nms_iou_threshold=0.25,
                minimum_instance_area_px=100,
            ),
        )
        self.assertEqual(result.count, 3)
        metrics = evaluate_instances(labels, result.labels, iou_threshold=0.50)
        self.assertEqual(metrics.true_positives, 3)
        self.assertGreater(metrics.mean_matched_iou, 0.85)

    def test_hybrid_rejects_stardist_candidates_outside_unet_interior(self) -> None:
        import torch

        labels = _scene()
        dense = build_dense_targets(labels)
        star = build_stardist_targets(labels, ray_count=32, maximum_distance=70)
        unet_outputs = {
            "interior_logits": torch.from_numpy(_logit(dense.interior))[None, None],
            "physical_boundary_logits": torch.from_numpy(
                _logit(np.clip(dense.physical_boundary, 0.001, 0.999))
            )[None, None],
            "pattern_boundary_logits": torch.full((1, 1, *labels.shape), -8.0),
            "centre_logits": torch.full((1, 1, *labels.shape), -8.0),
            "distance": torch.from_numpy(dense.distance)[None, None],
            "uncertainty_logits": torch.full((1, 5, *labels.shape), -5.0),
        }
        object_probability = star.object_probability.copy()
        object_probability[8, 8] = 0.999
        radial = star.radial_distances.copy()
        radial[:, 8, 8] = 12.0
        star_outputs = {
            "object_logits": torch.from_numpy(
                _logit(np.clip(object_probability, 0.001, 0.999))
            )[None, None],
            "radial_distances": torch.from_numpy(radial)[None],
            "radial_uncertainty_logits": torch.full((1, 1, *labels.shape), -5.0),
        }
        result = decode_unet_stardist_hybrid(
            unet_outputs,
            star_outputs,
            settings=HybridDecodeSettings(
                unet=UNetWatershedSettings(
                    centre_threshold=0.95,
                    centre_minimum_separation_px=18.0,
                    minimum_instance_area_px=100,
                ),
                stardist=StarDistDecodeSettings(
                    object_threshold=0.4,
                    nms_iou_threshold=0.25,
                    minimum_instance_area_px=100,
                ),
            ),
        )
        self.assertEqual(result.count, 3)
        self.assertEqual(result.labels[8, 8], 0)
        self.assertGreater(evaluate_instances(labels, result.labels).f1, 0.99)

    def test_instance_metrics_report_perfect_prediction(self) -> None:
        labels = _scene()
        metrics = evaluate_instances(labels, labels, boundary_tolerance_px=1.0)
        self.assertEqual(metrics.true_instances, 3)
        self.assertEqual(metrics.true_positives, 3)
        self.assertAlmostEqual(metrics.panoptic_quality, 1.0)
        self.assertAlmostEqual(metrics.boundary_f1, 1.0)
        self.assertEqual(metrics.count_error, 0)

    def test_binary_probability_metrics_are_perfect_for_exact_scores(self) -> None:
        truth = np.asarray([[0, 0, 1, 1]], dtype=np.uint8)
        probability = np.asarray([[0.01, 0.1, 0.9, 0.99]], dtype=np.float32)
        metrics = evaluate_binary_probability(probability, truth)
        self.assertAlmostEqual(metrics.f1, 1.0)
        self.assertAlmostEqual(metrics.roc_auc, 1.0)
        self.assertAlmostEqual(metrics.average_precision, 1.0)


if __name__ == "__main__":
    unittest.main()
