"""The three-arm training experiment: radial weighting, shared network, budget and reports."""
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from seedvision.learning.contracts import FeatureStackSpec, ModelFamily
from seedvision.learning.data import SeedTileDataset, export_learning_sample
from seedvision.learning.losses import multi_head_unet_loss, shared_network_loss, stardist_loss
from seedvision.learning.training_experiment import (
    ExperimentConfiguration,
    adoption_verdict,
    describe_evidence,
    run_training_experiment,
    training_configuration,
)
from tests.functional_units import functional_units


SPEC = FeatureStackSpec(include_species_planes=False)
SHARED = ModelFamily.SHARED_UNET_STARDIST.value
UNET = ModelFamily.UNET_WATERSHED.value
STARDIST = ModelFamily.STARDIST.value
QUICK = ExperimentConfiguration(
    steps=2, validate_every=1, seeds=(5,), batch_size=2, tile_size=64,
    validation_tiles_per_sample=1, base_channels=8, depth=1, ray_count=8,
    evaluation_tile_size=64, evaluation_overlap=8, device="cpu", allow_unreviewed=True,
)


def _labels(offset: int) -> np.ndarray:
    labels = np.zeros((96, 96), np.uint16)
    for index, (row, column) in enumerate(((20, 20), (20, 60), (60, 24), (64, 64)), start=1):
        rows, columns = np.ogrid[:96, :96]
        selected = (rows - row - offset) ** 2 + (columns - column) ** 2 <= 11**2
        labels[selected] = index
    return labels


def _export(manifest: Path, identifier: str, split: str, reviewed: bool, offset: int = 0) -> None:
    labels = _labels(offset)
    features = np.zeros((SPEC.input_channels, 96, 96), np.float32)
    features[0] = labels > 0
    export_learning_sample(
        manifest, dataset_id="experiment", feature_spec=SPEC, identifier=identifier,
        features=features, labels=labels, image_bgr=None, species="soybean",
        group=f"capture-{identifier}", split=split, reviewed=reviewed,
        annotation_author="Test reviewer" if reviewed else None,
    )


def _summary_row(arm, output, separate, spread=0.01):
    metric = {"mean": separate, "spread": spread}
    return {"arm": arm, "output": output, "seeds": 3,
            "metrics": {"f1_iou50": metric, "f1_iou75": metric}}


@functional_units
class TrainingExperimentTests(unittest.TestCase):
    def test_training_experiment(self):
        """Training experiment: 12 scenarios from the loss to the final report."""
        self.scenario(self.check_radial_error_counts_in_proportion_to_object_probability)
        self.scenario(self.check_training_tiles_change_each_epoch_and_repeat_within_one)
        self.scenario(self.check_shared_targets_equal_both_separate_datasets)
        self.scenario(self.check_shared_model_starts_like_the_separate_unet_and_round_trips)
        self.scenario(self.check_shared_loss_sums_both_losses_and_trains_every_head)
        self.scenario(self.check_a_shared_checkpoint_must_name_the_output_to_score)
        self.scenario(self.check_step_budget_is_met_for_any_image_count)
        self.scenario(self.check_verdict_needs_reviewed_test_images_and_three_seeds)
        self.scenario(self.check_verdict_reports_when_joint_training_falls_short)
        self.scenario(self.check_unreviewed_run_is_labelled_not_validation_and_resumes)
        self.scenario(self.check_reviewed_test_images_are_scored_with_the_validation_decoder)
        self.scenario(self.check_launcher_forwards_and_runs_the_experiment)

    def check_radial_error_counts_in_proportion_to_object_probability(self):
        import torch

        probability = torch.tensor([[[[1.0, 0.1, 0.0]]]])
        expected = torch.full((1, 8, 1, 3), 10.0)

        def radial_term(column):
            predicted = expected.clone()
            predicted[:, :, 0, column] = 20.0
            outputs = {
                "object_logits": torch.zeros(1, 1, 1, 3),
                "radial_distances": predicted,
                "radial_uncertainty_logits": torch.zeros(1, 1, 1, 3),
            }
            targets = {
                "valid": torch.ones(1, 1, 1, 3),
                "object_probability": probability,
                "radial_distances": expected,
                "radial_valid": torch.ones(1, 8, 1, 3),
            }
            return float(stardist_loss(outputs, targets)[1]["radial"])

        self.assertAlmostEqual(radial_term(0) / radial_term(1), 10.0, places=4)
        self.assertEqual(radial_term(2), 0.0)

    def check_training_tiles_change_each_epoch_and_repeat_within_one(self):
        with TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            _export(manifest, "a", "train", True)
            dataset = SeedTileDataset(
                manifest, split="train", family=UNET, tile_size=32, tiles_per_sample=8,
                augment=True, random_seed=3,
            )

            def tiles(epoch):
                dataset.set_epoch(epoch)
                return np.stack([dataset[index]["targets"]["interior"].numpy() for index in range(8)])

            first = tiles(1)
            np.testing.assert_array_equal(first, tiles(1))
            self.assertFalse(np.array_equal(first, tiles(2)))

    def check_shared_targets_equal_both_separate_datasets(self):
        with TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            _export(manifest, "a", "train", True)
            datasets = {
                family: SeedTileDataset(
                    manifest, split="train", family=family, tile_size=48, tiles_per_sample=12,
                    ray_count=8, augment=True, random_seed=9,
                )
                for family in (UNET, STARDIST, SHARED)
            }
            for index in range(12):
                shared = datasets[SHARED][index]["targets"]
                for family in (UNET, STARDIST):
                    separate = datasets[family][index]["targets"]
                    self.assertLessEqual(set(separate), set(shared))
                    for name, values in separate.items():
                        np.testing.assert_array_equal(values.numpy(), shared[name].numpy(), err_msg=name)
                self.assertEqual(len(shared), 10)
            caches = {path.name.split(".")[1] for path in (Path(directory) / ".seedfiddle-cache").glob("*.npz")}
            self.assertEqual(caches, {"unet", "stardist8", "shared8"})

    def check_shared_model_starts_like_the_separate_unet_and_round_trips(self):
        import torch
        from seedvision.learning.checkpoint import load_checkpoint, save_checkpoint
        from seedvision.learning.models import MultiHeadSeedUNet, SeedStarDist2D, SharedSeedUNet

        def seeded(constructor, **settings):
            torch.manual_seed(11)
            return constructor(SPEC.input_channels, base_channels=8, depth=2, **settings)

        shared, unet = seeded(SharedSeedUNet, ray_count=8), seeded(MultiHeadSeedUNet)
        star = seeded(SeedStarDist2D, ray_count=8)
        weights = shared.state_dict()
        for name, value in unet.state_dict().items():
            self.assertTrue(torch.equal(value, weights[name]), name)
        for name, value in star.state_dict().items():
            if name.startswith("backbone."):
                self.assertTrue(torch.equal(value, weights[name]), name)
        values = torch.zeros(1, SPEC.input_channels, 32, 32)
        outputs = shared(values)
        self.assertEqual(set(outputs), set(unet(values)) | set(star(values)))
        self.assertEqual(outputs["radial_distances"].shape, (1, 8, 32, 32))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "shared.pt"
            save_checkpoint(path, shared, family=SHARED, feature_spec=SPEC)
            loaded, _spec, payload = load_checkpoint(path)
            self.assertEqual(payload["family"], SHARED)
            self.assertEqual(loaded.family, SHARED)
            for name, value in loaded.state_dict().items():
                self.assertTrue(torch.equal(value, weights[name]), name)

    def check_shared_loss_sums_both_losses_and_trains_every_head(self):
        import torch
        from seedvision.learning.models import SharedSeedUNet

        with TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            _export(manifest, "a", "train", True)
            dataset = SeedTileDataset(manifest, split="train", family=SHARED, tile_size=48, ray_count=8)
            item = dataset[0]
        features = item["features"][None]
        targets = {name: value[None] for name, value in item["targets"].items()}
        torch.manual_seed(2)
        model = SharedSeedUNet(SPEC.input_channels, ray_count=8, base_channels=8, depth=1)
        outputs = model(features)
        total, terms = shared_network_loss(outputs, targets, unet_weights={"pattern_boundary": 0.5})
        unet_total, unet_terms = multi_head_unet_loss(outputs, targets, weights={"pattern_boundary": 0.5})
        star_total, star_terms = stardist_loss(outputs, targets)
        self.assertAlmostEqual(total.item(), (unet_total + star_total).item(), places=4)
        self.assertAlmostEqual(terms["unet_total"].item(), unet_total.item(), places=5)
        self.assertAlmostEqual(terms["stardist_radial"].item(), star_terms["radial"].item(), places=5)
        self.assertAlmostEqual(terms["unet_centre"].item(), unet_terms["centre"].item(), places=5)
        total.backward()
        for head in ("interior_head", "centre_distance_head", "object_head", "radial_head",
                     "radial_uncertainty_head"):
            gradient = getattr(model, head)[-1].weight.grad
            self.assertGreater(float(gradient.abs().sum()), 0.0, head)

    def check_a_shared_checkpoint_must_name_the_output_to_score(self):
        from seedvision.learning.checkpoint import save_checkpoint
        from seedvision.learning.evaluation import decoded_family, evaluate_checkpoint
        from seedvision.learning.models import SharedSeedUNet

        family = ModelFamily
        self.assertIs(decoded_family(family.UNET_WATERSHED), family.UNET_WATERSHED)
        self.assertIs(decoded_family(family.SHARED_UNET_STARDIST, "stardist"), family.STARDIST)
        with self.assertRaisesRegex(ValueError, "no stardist output"):
            decoded_family(family.UNET_WATERSHED, "stardist")
        with self.assertRaisesRegex(ValueError, "no shared_unet_stardist output"):
            decoded_family(family.SHARED_UNET_STARDIST, family.SHARED_UNET_STARDIST)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            _export(manifest, "a", "test", True)
            path = root / "shared.pt"
            model = SharedSeedUNet(SPEC.input_channels, ray_count=8, base_channels=8, depth=1)
            save_checkpoint(path, model, family=SHARED, feature_spec=SPEC)
            settings = {"device": "cpu", "tile_size": 64, "overlap": 8}
            with self.assertRaisesRegex(ValueError, "needs output"):
                evaluate_checkpoint(path, manifest, root / "none", **settings)
            for output in (UNET, STARDIST):
                report = evaluate_checkpoint(path, manifest, root / output, output=output, **settings)
                self.assertEqual(report["family"], output)
                self.assertEqual(report["checkpoint_family"], SHARED)
                self.assertTrue((root / output / "a.prediction.png").is_file())

    def check_step_budget_is_met_for_any_image_count(self):
        budget = ExperimentConfiguration(steps=1000, validate_every=200, batch_size=4)
        for images, expected_steps in ((1, 200), (2, 200), (3, 201), (7, 202)):
            configuration = training_configuration(budget, "shared", 7, images)
            self.assertEqual(configuration.epochs, 5)
            self.assertGreater(configuration.patience, configuration.epochs)
            self.assertEqual(configuration.family, SHARED)
            self.assertEqual(configuration.random_seed, 7)
            tiles = images * configuration.tiles_per_sample
            self.assertEqual(-(-tiles // budget.batch_size), expected_steps)

    def check_verdict_needs_reviewed_test_images_and_three_seeds(self):
        summary = [
            _summary_row("unet", UNET, 0.90), _summary_row("stardist", STARDIST, 0.80),
            _summary_row("shared", UNET, 0.895), _summary_row("shared", STARDIST, 0.80),
        ]
        reviewed = {"state": "reviewed_test"}
        verdict = adoption_verdict(summary, {"state": "no_test_split"}, (1, 2, 3))
        self.assertFalse(verdict["decided"])
        self.assertFalse(adoption_verdict(summary, reviewed, (1, 2))["decided"])
        self.assertFalse(adoption_verdict(summary[:2], reviewed, (1, 2, 3))["decided"])
        verdict = adoption_verdict(summary, reviewed, (1, 2, 3))
        self.assertTrue(verdict["decided"])
        self.assertTrue(verdict["shared_network_holds_up"])
        self.assertEqual(len(verdict["comparisons"]), 4)

    def check_verdict_reports_when_joint_training_falls_short(self):
        summary = [
            _summary_row("unet", UNET, 0.90), _summary_row("stardist", STARDIST, 0.80),
            _summary_row("shared", UNET, 0.85), _summary_row("shared", STARDIST, 0.85),
        ]
        verdict = adoption_verdict(summary, {"state": "reviewed_test"}, (1, 2, 3))
        self.assertTrue(verdict["decided"])
        self.assertFalse(verdict["shared_network_holds_up"])
        short = {(item["output"], item["shared_holds_up"]) for item in verdict["comparisons"]}
        self.assertEqual(short, {(UNET, False), (STARDIST, True)})

    def check_unreviewed_run_is_labelled_not_validation_and_resumes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "data" / "manifest.json"
            _export(manifest, "a", "train", False)
            _export(manifest, "b", "validation", False, offset=2)
            output = root / "run"
            lines = []
            report = run_training_experiment(manifest, output, QUICK, log=lines.append)
            self.assertEqual(report["evidence"]["state"], "no_test_split")
            self.assertIn("NOT VALIDATION", lines[0])
            self.assertIn("NOT VALIDATION", (output / "experiment.md").read_text(encoding="utf-8"))
            self.assertFalse(report["verdict"]["decided"])
            self.assertFalse(report["scientifically_validated"])
            self.assertEqual(report["optimizer_steps_by_arm"], {"unet": 2, "stardist": 2, "shared": 2})
            self.assertEqual(
                sorted((row["arm"], row["output"]) for row in report["summary"]),
                [("shared", STARDIST), ("shared", UNET), ("stardist", STARDIST), ("unet", UNET)],
            )
            stored = json.loads((output / "experiment.json").read_text(encoding="utf-8"))
            self.assertEqual(stored["evidence"], report["evidence"])
            run = next(item for item in stored["runs"] if item["arm"] == "shared")
            self.assertIn("unet_total", run["last_validation"])
            self.assertIn("stardist_total", run["last_validation"])
            with patch("seedvision.learning.training_experiment.train_from_manifest",
                       side_effect=AssertionError("retrained")), \
                 patch("seedvision.learning.training_experiment.evaluate_checkpoint",
                       side_effect=AssertionError("rescored")):
                resumed = run_training_experiment(manifest, output, QUICK, log=lines.append)
            self.assertEqual(resumed["summary"], report["summary"])
            with patch("seedvision.learning.training_experiment.train_from_manifest",
                       side_effect=RuntimeError("configuration changed")):
                with self.assertRaisesRegex(RuntimeError, "configuration changed"):
                    run_training_experiment(manifest, output, replace(QUICK, steps=4), log=lines.append)

    def check_reviewed_test_images_are_scored_with_the_validation_decoder(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "data" / "manifest.json"
            _export(manifest, "a", "train", False)
            _export(manifest, "b", "validation", True, offset=2)
            _export(manifest, "c", "test", True, offset=4)
            report = run_training_experiment(
                manifest, root / "run", replace(QUICK, arms=("stardist",)), log=lambda _text: None
            )
            evidence = report["evidence"]
            self.assertEqual(evidence["state"], "reviewed_test")
            self.assertEqual(evidence["scored_images"], ["c"])
            self.assertFalse(evidence["training_masks_reviewed"])
            run = report["runs"][0]
            self.assertEqual([image["sample_id"] for image in run["images"]], ["c"])
            self.assertIn("No recorded", run["membership_audit"])
            stored = json.loads(
                (root / "run" / "stardist-seed5" / "stardist-test" / "evaluation.json").read_text(encoding="utf-8")
            )
            self.assertEqual(stored["decoder_settings"], run["decoder_settings"])
            self.assertEqual(stored["decoder_selection_provenance"]["selection_protocol"], "validation")
            self.assertNotIn("NOT VALIDATION", (root / "run" / "experiment.md").read_text(encoding="utf-8"))
            self.assertIsNotNone(run["mean"]["f1_iou75"])
            manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
            for sample in manifest_payload["samples"]:
                if sample["identifier"] == "c":
                    sample["reviewed"] = False
            manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
            from seedvision.learning.data import LearningManifest

            self.assertEqual(describe_evidence(LearningManifest.load(manifest))["state"], "unreviewed_test")

    def check_launcher_forwards_and_runs_the_experiment(self):
        import seed_vision

        arguments = seed_vision.build_parser().parse_args([
            "--run-training-experiment", "manifest.json", "--experiment-output", "run",
            "--experiment-steps", "40", "--experiment-seeds", "1", "2", "--experiment-arms", "shared",
            "--allow-unreviewed-learning-data",
        ])
        forwarded = seed_vision._learning_arguments(arguments)
        reparsed = seed_vision.build_parser().parse_args(forwarded)
        for name in ("run_training_experiment", "experiment_output", "experiment_steps",
                     "experiment_seeds", "experiment_arms", "allow_unreviewed_learning_data"):
            self.assertEqual(getattr(reparsed, name), getattr(arguments, name), name)
        with patch("seedvision.learning.training_experiment.run_training_experiment") as run:
            self.assertEqual(seed_vision._run_learning_action(arguments), 0)
        configuration = run.call_args.args[2]
        self.assertEqual((configuration.steps, configuration.seeds, configuration.arms), (40, (1, 2), ("shared",)))
        self.assertTrue(configuration.allow_unreviewed)
        missing = seed_vision.build_parser().parse_args(["--run-training-experiment", "manifest.json"])
        with self.assertRaisesRegex(seed_vision.BootstrapError, "requires --experiment-output"):
            seed_vision._run_learning_action(missing)


if __name__ == "__main__":
    unittest.main()
