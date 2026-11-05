from __future__ import annotations

from tests.functional_units import functional_units

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
from seedvision.persistence.reference_regions import SeedInstanceAnnotation


@functional_units
class LearningWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        cls.application = QApplication.instance() or QApplication([])

    @staticmethod
    def _training_image(name, seeds=10, species="Soybean", **fields):
        from seedvision.ui.learning_workflow import TrainingImage

        return TrainingImage(key=name, path=Path(name + ".png"), seed_count=seeds, species=species, **fields)

    def check_test_default_splits_hold_out_about_a_fifth_per_species(self) -> None:
        from seedvision.ui.learning_workflow import default_training_splits

        image = self._training_image
        self.assertEqual(default_training_splits([]), {})
        self.assertEqual(default_training_splits([image("only")]), {"only": "train"})
        # The largest image of a species always trains.
        self.assertEqual(
            default_training_splits([image("small", 5), image("large", 50)]),
            {"large": "train", "small": "validation"},
        )
        counts = [31, 30, 23, 8, 6, 5, 3]
        splits = default_training_splits(
            [image("big", 657, "Lupinus mexicanus")]
            + [image(f"soy{count}", count) for count in counts]
        )
        self.assertEqual(splits["big"], "train")
        self.assertEqual(
            sorted(key for key, split in splits.items() if split == "validation"),
            ["soy3", "soy30"],
        )
        # A recorded group moves together and counts its images' seeds together.
        splits = default_training_splits(
            [image("a", 40), image("b1", 30, group="b"), image("b2", 30, group="b"), image("c", 5)]
        )
        self.assertEqual(splits, {"a": "validation", "b1": "train", "b2": "train", "c": "train"})
        # One image per species still has to supply a validation image.
        splits = default_training_splits(
            [image("big", 100, "Lupinus mexicanus"), image("soy", 20)]
        )
        self.assertEqual(splits, {"big": "train", "soy": "validation"})

    def check_test_training_dialog_keeps_groups_in_one_split_and_explains_refusals(self) -> None:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QMessageBox
        from seedvision.ui.learning_workflow import LearningTrainingDialog

        image = self._training_image
        images = (
            image("large", 50),
            image("small", 5, reviewed=True),
            image("third", 4, species=""),
            image("broken", 9, problem="Reconnect seeds 7."),
        )
        with TemporaryDirectory() as temporary:
            dialog = LearningTrainingDialog(
                project_root=Path(temporary),
                images=images,
                species_names=("Soybean", "Lupinus mexicanus"),
            )
            self.addCleanup(dialog.close)
            table = dialog.image_table
            splits = {item.key: item.split for item in dialog.selected_images()}
            self.assertEqual(set(splits), {"large", "small", "third"})
            self.assertEqual(splits["large"], "train")
            self.assertEqual(table.item(0, dialog.GROUP_COLUMN).text(), "large")
            self.assertEqual(table.item(1, dialog.SEEDS_COLUMN).text(), "5")

            # An image with a problem says why and cannot be used.
            self.assertEqual(table.item(3, dialog.NOTE_COLUMN).text(), "Reconnect seeds 7.")
            self.assertFalse(table.item(3, dialog.IMAGE_COLUMN).flags() & Qt.ItemFlag.ItemIsEnabled)
            self.assertFalse(table.cellWidget(3, dialog.SPECIES_COLUMN).isEnabled())

            # Joining a group takes that group's split; changing a split moves the group.
            table.item(2, dialog.GROUP_COLUMN).setText("large")
            self.assertEqual(dialog._split_combo(2).currentData(), "train")
            dialog._split_combo(0).setCurrentIndex(dialog._split_combo(0).findData("validation"))
            self.assertEqual(dialog._split_combo(2).currentData(), "validation")
            dialog._split_combo(0).setCurrentIndex(dialog._split_combo(0).findData("train"))
            self.assertEqual(dialog._split_combo(2).currentData(), "train")

            def refusal():
                with patch.object(QMessageBox, "warning") as warning:
                    dialog.accept()
                return warning.call_args.args[2] if warning.called else None

            self.assertIn("Choose a species for: third.png", refusal())
            table.cellWidget(2, dialog.SPECIES_COLUMN).setCurrentIndex(1)
            self.assertIn("Enter who checked", refusal())
            dialog.reviewer_edit.setText("Reviewer")
            dialog._split_combo(1).setCurrentIndex(dialog._split_combo(1).findData("train"))
            self.assertIn("one Validation image", refusal())
            dialog._split_combo(1).setCurrentIndex(dialog._split_combo(1).findData("validation"))
            table.item(1, dialog.GROUP_COLUMN).setText(" ")
            self.assertIn("Give a group to: small.png", refusal())
            table.item(1, dialog.GROUP_COLUMN).setText("small")
            self.assertIsNone(refusal())

            request = dialog.request()
            self.assertEqual(request.reviewer, "Reviewer")
            self.assertEqual(
                {item.key: (item.group, item.split, item.reviewed, item.species) for item in request.images},
                {
                    "large": ("large", "train", False, "Soybean"),
                    "small": ("small", "validation", True, "Soybean"),
                    "third": ("large", "train", False, "Soybean"),
                },
            )

    def check_test_training_dialog_names_same_stem_images_apart(self) -> None:
        from seedvision.ui.learning_workflow import LearningTrainingDialog, TrainingImage

        with TemporaryDirectory() as temporary:
            dialog = LearningTrainingDialog(
                project_root=Path(temporary),
                images=(
                    TrainingImage("a", Path("first/IMG 1.png"), 3, "Soybean"),
                    TrainingImage("b", Path("second/IMG 1.png"), 3, "Soybean"),
                ),
                species_names=("Soybean",),
            )
            self.addCleanup(dialog.close)
            self.assertEqual(
                [item.identifier for item in dialog.selected_images()], ["IMG_1", "IMG_1_2"]
            )

    def check_test_training_dialog_can_hold_an_image_out_as_test(self) -> None:
        from seedvision.ui.learning_workflow import LearningTrainingDialog, TrainingImage

        with TemporaryDirectory() as temporary:
            dialog = LearningTrainingDialog(
                project_root=Path(temporary),
                images=tuple(
                    TrainingImage(key, Path(f"{key}.png"), 3, "Soybean") for key in ("a", "b", "c")
                ),
                species_names=("Soybean",),
            )
            self.addCleanup(dialog.close)
            combo = dialog._split_combo(2)
            combo.setCurrentIndex(combo.findData("test"))
            self.assertEqual(
                sorted(item.split for item in dialog.selected_images()), ["test", "train", "validation"]
            )

    def check_test_training_dialog_constructs_new_and_refinement_requests(self) -> None:
        from seedvision.ui.learning_workflow import LearningTrainingDialog

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dialog = LearningTrainingDialog(project_root=root)
            dialog.family_combo.setCurrentIndex(
                dialog.family_combo.findData("stardist")
            )
            dialog.initial_row.edit.setText(str(root / "initial.pt"))
            dialog.epochs_spin.setValue(3)
            request = dialog.request()

            self.assertEqual(request.configuration.family, "stardist")
            self.assertEqual(request.configuration.epochs, 3)
            self.assertEqual(
                request.configuration.initial_checkpoint,
                str((root / "initial.pt").resolve()),
            )
            self.assertTrue(request.activate_after_training)
            # Each run writes its own dataset beside any earlier ones.
            self.assertEqual(request.manifest_path.name, "manifest.json")
            self.assertEqual(request.manifest_path.parent.parent, root / "learning-data")
            self.assertTrue(request.manifest_path.parent.name.startswith("stardist_seed_instances-"))
            dialog.close()

    def check_test_training_task_forwards_epoch_progress_and_completion(self) -> None:
        from seedvision.learning.training import TrainingConfiguration
        from seedvision.ui.learning_workflow import LearningTrainingRequest
        from seedvision.ui.main_window import _LearningTrainingTask

        request = LearningTrainingRequest(
            manifest_path=Path("manifest.json"),
            output_path=Path("checkpoint.pt"),
            configuration=TrainingConfiguration(
                family="unet_watershed", epochs=2, device="cpu"
            ),
            activate_after_training=False,
        )
        task = _LearningTrainingTask(request)
        progress: list[tuple[int, int, float, float]] = []
        completed: list[dict] = []
        failures: list[str] = []
        task.signals.progress.connect(lambda *values: progress.append(values))
        task.signals.completed.connect(completed.append)
        task.signals.failed.connect(failures.append)

        def fake_train(
            _manifest,
            _output,
            configuration,
            *,
            progress_callback,
            cancellation_requested,
        ):
            self.assertFalse(cancellation_requested())
            record = {
                "train": {"total": 0.4},
                "validation": {"total": 0.3},
            }
            progress_callback(1, configuration.epochs, record)
            return {"best_epoch": 1, "best_validation_loss": 0.3}

        with patch(
            "seedvision.learning.training.train_from_manifest",
            side_effect=fake_train,
        ):
            task.run()

        self.assertEqual(progress, [(1, 2, 0.4, 0.3)])
        self.assertEqual(completed[0]["best_epoch"], 1)
        self.assertEqual(failures, [])

    def check_test_analysis_export_derives_nonphysical_targets_from_instance_interiors(self) -> None:
        import cv2
        import torch

        from seedvision.learning.contracts import FeatureStackSpec
        from seedvision.learning.export import export_analysis_sample

        shape = (96, 96)
        labels = np.zeros(shape, np.uint16)
        cv2.circle(labels, (48, 48), 30, 1, -1)
        edge = np.zeros(shape, np.uint8)
        ridges = np.zeros(shape, np.uint8)
        cv2.line(edge, (28, 48), (68, 48), 220, 3)
        cv2.line(ridges, (28, 48), (68, 48), 255, 1)
        layers = SimpleNamespace(
            valid_mask=np.full(shape, 255, np.uint8),
            gpu_source=object(),
            gpu_valid=object(),
            edge_likelihood=edge,
            edge_ridges=ridges,
            foreground_noise_likelihood=np.zeros(shape, np.uint8),
            background_likelihood=np.zeros(shape, np.uint8),
            refined_background_likelihood=np.zeros(shape, np.uint8),
        )
        result = SimpleNamespace(
            calibration=SimpleNamespace(
                corrected_bgr=np.full((*shape, 3), 127, np.uint8)
            ),
            crop_offset=(0, 0),
            layers=layers,
            foreground_probability=np.zeros(shape, np.uint8),
            foreground_colour_probability=np.full(shape, 64, np.uint8),
            advanced=SimpleNamespace(
                rasters={
                    "sensor_noise": np.zeros(shape, np.uint8),
                    "flattened_grayscale": np.zeros(shape, np.uint8),
                    "shadow_likelihood": np.zeros(shape, np.uint8),
                    "highlight_likelihood": np.zeros(shape, np.uint8),
                }
            ),
            estimated_seed_diameter_px=64.0,
        )
        spec = FeatureStackSpec(
            nominal_seed_diameter_px=64.0,
            include_species_planes=False,
        )
        feature_stack = torch.zeros(
            (1, spec.input_channels, *shape), dtype=torch.float32
        )
        with TemporaryDirectory() as temporary, patch(
            "seedvision.learning.export.assemble_feature_stack",
            return_value=feature_stack,
        ):
            manifest = Path(temporary) / "manifest.json"
            sample = export_analysis_sample(
                result,
                labels,
                manifest,
                dataset_id="derived-boundaries",
                seed_annotations=tuple(SeedInstanceAnnotation(int(identifier),shape_reviewed=True,outline_visibility='complete') for identifier in np.unique(labels) if identifier),
                identifier="sample",
                species="soybean",
                group="capture-a",
                feature_spec=spec,
            )
            self.assertIsNotNone(sample.pattern_boundary)
            self.assertIsNotNone(sample.pattern_valid)
            pattern = cv2.imread(
                str(manifest.parent / sample.pattern_boundary),
                cv2.IMREAD_UNCHANGED,
            )
            pattern_valid = cv2.imread(
                str(manifest.parent / sample.pattern_valid),
                cv2.IMREAD_UNCHANGED,
            )
            self.assertGreater(int(np.count_nonzero(pattern)), 0)
            self.assertTrue(np.all(pattern[pattern_valid == 0] == 0))
            self.assertLess(
                int(np.count_nonzero(pattern)),
                int(np.count_nonzero(pattern_valid)),
            )

            def export_variant(identifier, annotations=(), masks=labels):
                return export_analysis_sample(
                    result,
                    masks,
                    manifest,
                    dataset_id="derived-boundaries",
                    seed_annotations=annotations,
                    identifier=identifier,
                    species="soybean",
                    group="capture-a",
                    feature_spec=spec,
                )

            # Unreviewed masks export; disconnected or excluded seeds do not.
            self.assertFalse(export_variant("unreviewed").reviewed)
            with self.assertRaisesRegex(ValueError, "shape exclusions"):
                export_variant("excluded", (SeedInstanceAnnotation(1, shape_exclusion_reason="damaged"),))
            disconnected = labels.copy()
            disconnected[5:8, 5:8] = 1
            with self.assertRaisesRegex(ValueError, "connected contour"):
                export_variant("disconnected", masks=disconnected)

    def check_test_reviewed_manifest_trains_and_refines_a_real_checkpoint(self) -> None:
        import torch

        from seedvision.learning.contracts import FeatureStackSpec
        from seedvision.learning.data import export_learning_sample
        from seedvision.learning.training import (
            TrainingConfiguration,
            train_from_manifest,
        )

        spec = FeatureStackSpec(include_species_planes=False)
        deterministic_before = torch.are_deterministic_algorithms_enabled()
        warn_only_before = torch.is_deterministic_algorithms_warn_only_enabled()
        features = np.zeros((spec.input_channels, 40, 40), dtype=np.float32)
        features[0, 8:32, 8:32] = 0.75
        labels = np.zeros((40, 40), dtype=np.uint16)
        labels[10:25, 9:20] = 1
        labels[14:30, 23:34] = 2
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            for identifier, group, split in (
                ("train", "capture-a", "train"),
                ("validation", "capture-b", "validation"),
            ):
                export_learning_sample(
                    manifest_path,
                    dataset_id="workflow-smoke",
                    feature_spec=spec,
                    identifier=identifier,
                    features=features + (np.float32(.01) if split == 'validation' else 0),
                    labels=labels,
                    image_bgr=np.zeros((40, 40, 3), dtype=np.uint8),
                    species="soybean",
                    group=group,
                    split=split,
                    reviewed=True,
                    annotation_author="Unit reviewer",
                )
            first = root / "unet-first.pt"
            progress: list[int] = []
            configuration = TrainingConfiguration(
                family="unet_watershed",
                epochs=1,
                batch_size=1,
                tile_size=32,
                tiles_per_sample=1,
                base_channels=8,
                depth=2,
                patience=1,
                device="cpu",
                mixed_precision=False,
            )
            report = train_from_manifest(
                manifest_path,
                first,
                configuration,
                progress_callback=lambda epoch, _maximum, _record: progress.append(epoch),
            )
            self.assertTrue(first.is_file())
            self.assertEqual(report["best_epoch"], 1)
            self.assertEqual(progress, [1])

            refined = root / "unet-refined.pt"
            refined_configuration = TrainingConfiguration(
                family="unet_watershed",
                epochs=1,
                batch_size=1,
                tile_size=32,
                tiles_per_sample=1,
                base_channels=8,
                depth=2,
                patience=1,
                device="cpu",
                mixed_precision=False,
                initial_checkpoint=str(first),
            )
            refined_report = train_from_manifest(
                manifest_path, refined, refined_configuration
            )
            self.assertTrue(refined.is_file())
            self.assertEqual(refined_report["best_epoch"], 1)
        self.assertEqual(
            torch.are_deterministic_algorithms_enabled(), deterministic_before
        )
        self.assertEqual(
            torch.is_deterministic_algorithms_warn_only_enabled(), warn_only_before
        )

    def test_workflow_lifecycle(self):
        """8 related LearningWorkflowTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_default_splits_hold_out_about_a_fifth_per_species)
        self.scenario(self.check_test_training_dialog_keeps_groups_in_one_split_and_explains_refusals)
        self.scenario(self.check_test_training_dialog_names_same_stem_images_apart)
        self.scenario(self.check_test_training_dialog_can_hold_an_image_out_as_test)
        self.scenario(self.check_test_training_dialog_constructs_new_and_refinement_requests)
        self.scenario(self.check_test_training_task_forwards_epoch_progress_and_completion)
        self.scenario(self.check_test_analysis_export_derives_nonphysical_targets_from_instance_interiors)
        self.scenario(self.check_test_reviewed_manifest_trains_and_refines_a_real_checkpoint)


if __name__ == "__main__":
    unittest.main()
