from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np


class LearningWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        cls.application = QApplication.instance() or QApplication([])

    def test_export_dialog_returns_review_and_split_metadata(self) -> None:
        from seedvision.ui.learning_workflow import LearningExportDialog

        with TemporaryDirectory() as temporary:
            dialog = LearningExportDialog(
                default_directory=Path(temporary),
                default_identifier="capture_01",
                default_group="lot-a/session-1",
                default_revision="2",
            )
            dialog.split_combo.setCurrentIndex(
                dialog.split_combo.findData("validation")
            )
            dialog.author_edit.setText("Reviewer")
            dialog.reviewed_checkbox.setChecked(True)
            options = dialog.options()

            self.assertEqual(options.manifest_path, Path(temporary) / "manifest.json")
            self.assertEqual(options.identifier, "capture_01")
            self.assertEqual(options.group, "lot-a/session-1")
            self.assertEqual(options.split, "validation")
            self.assertTrue(options.reviewed)
            self.assertEqual(options.annotation_author, "Reviewer")
            dialog.close()

    def test_training_dialog_constructs_new_and_refinement_requests(self) -> None:
        from seedvision.ui.learning_workflow import LearningTrainingDialog

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dialog = LearningTrainingDialog(project_root=root)
            dialog.manifest_row.edit.setText(str(root / "manifest.json"))
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
            dialog.close()

    def test_training_task_forwards_epoch_progress_and_completion(self) -> None:
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

    def test_reviewed_manifest_trains_and_refines_a_real_checkpoint(self) -> None:
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
                    features=features,
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


if __name__ == "__main__":
    unittest.main()
