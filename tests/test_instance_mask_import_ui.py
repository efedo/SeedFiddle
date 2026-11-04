from __future__ import annotations

from tests.review_fixtures import workspace_window
from tests.functional_units import functional_units

from tests.review_fixtures import dispose_window

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


@functional_units
class InstanceMaskImportUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtWidgets import QApplication
        except ImportError as error:
            raise unittest.SkipTest(f"PySide6 unavailable: {error}")
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        from PySide6.QtGui import QColor, QImage

        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.image_path = self.root / "images" / "capture.png"
        self.image_path.parent.mkdir(parents=True)
        image = QImage(80, 60, QImage.Format.Format_RGB888)
        image.fill(QColor("#8b927f"))
        self.assertTrue(image.save(str(self.image_path)))
        self.window = workspace_window(self.root)
        self.addCleanup(dispose_window, self.window)
        self.window.image_view.set_overlay_mode("none")
        self.result = SimpleNamespace(
            calibration=SimpleNamespace(
                corrected_bgr=np.full((60, 80, 3), 128, dtype=np.uint8),
                affine_matrix=np.eye(3, dtype=np.float64),
            ),
            layers=SimpleNamespace(
                edge_likelihood=np.zeros((60, 80), dtype=np.float32),
                undirected_edge_hue=np.zeros((60, 80), dtype=np.float32),
                directed_edge_hue=np.zeros((60, 80), dtype=np.float32),
                offset_x=0,
                offset_y=0,
            ),
            crop_offset=(0, 0),
            estimated_seed_diameter_px=20.0,
        )
        key = self.window._current_image_key()
        self.assertIsNotNone(key)
        assert key is not None
        self.key = key
        self.window._analyses[key] = self.result
        self.window.image_view._analysis_result = self.result
        self.window.image_view._render_analysis = lambda: None
        self.window._sync_background_controls()
        self.application.processEvents()

    @staticmethod
    def _imported(labels: np.ndarray, *, bundled: bool = True):
        from seedvision.persistence import ImportedInstanceMask

        return ImportedInstanceMask(
            labels=labels,
            source_path=Path("bundle.png" if bundled else "manual.npz"),
            coordinate_space=("source->corrected" if bundled else "corrected"),
            origin=(
                "bundled:seed-instance-references/masks/capture.png;reviewed=false"
                if bundled
                else "import:C:/manual.npz"
            ),
            reviewed=(False if bundled else None),
            external_reference_count=(1 if bundled else None),
        )

    def check_test_panel_import_is_draft_with_provenance_warning_and_undo(self) -> None:
        labels = np.zeros((60, 80), dtype=np.uint16)
        labels[4:9, 5:10] = 7
        labels[30:35, 45:50] = 7
        labels[12:22, 20:31] = 9

        self.window.instance_smart_fill_button.setChecked(True)
        self.window.show_instance_annotations_checkbox.setChecked(False)
        imported = self._imported(labels)
        with patch(
            "seedvision.ui.main_window.load_bundled_instance_mask",
            return_value=imported,
        ), patch(
            "seedvision.ui.main_window.QFileDialog.getOpenFileName"
        ) as chooser:
            self.window.load_instance_reference_button.click()

        chooser.assert_not_called()
        self.assertTrue(
            np.array_equal(self.window._draft_instance_annotations[self.key], labels)
        )
        self.assertEqual(
            self.window._draft_instance_annotation_origins[self.key], imported.origin
        )
        self.assertIn(self.key, self.window._instance_annotations_dirty)
        self.assertTrue(self.window.annotate_instances_action.isChecked())
        self.assertTrue(self.window.instance_smart_fill_button.isChecked())
        self.assertFalse(self.window.show_instance_annotations_checkbox.isChecked())
        self.assertEqual(
            self.window._instance_undo_histories[self.key].next_label,
            "imported seed labels",
        )
        self.assertIn(
            "1 seed ID has disconnected areas: 7",
            self.window.image_view.instance_continuity_warning_banner.text(),
        )
        self.assertIn("unreviewed", self.window.statusBar().currentMessage())
        self.assertIn("IDs were preserved exactly", self.window.statusBar().currentMessage())

        self.window._undo_instance_reference_edit()
        current = self.window._draft_instance_annotations.get(
            self.key, self.window._applied_instance_annotations.get(self.key)
        )
        self.assertTrue(current is None or not np.any(current))
        self.assertEqual(len(self.window._instance_undo_histories[self.key]),0)
        self.assertTrue(self.window._instance_undo_histories[self.key].can_redo)

    def check_test_bundle_import_is_explicit_and_preserves_applied_labels(self) -> None:
        from PySide6.QtWidgets import QMessageBox
        applied = np.zeros((60, 80), np.uint16)
        applied[5:15, 8:20] = 1
        bundled = np.zeros_like(applied)
        bundled[24:38, 31:47] = 9
        self.window._applied_instance_annotations[self.key] = applied
        self.window._set_instance_draft_state(self.key, applied, 'manual')
        with patch('seedvision.ui.main_window.load_bundled_instance_mask',
                   return_value=self._imported(bundled)) as loader, \
             patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Yes):
            self.window.load_instance_labels_action.trigger()
        loader.assert_called_once()
        np.testing.assert_array_equal(self.window._applied_instance_annotations[self.key], applied)
        np.testing.assert_array_equal(self.window._draft_instance_annotations[self.key], bundled)
        self.assertIn(self.key, self.window._instance_annotations_dirty)
        self.window._undo_instance_reference_edit()
        np.testing.assert_array_equal(self.window._draft_instance_annotations.get(
            self.key, self.window._applied_instance_annotations[self.key]), applied)

    def check_test_no_bundle_explains_explicit_file_import(self) -> None:
        from PySide6.QtWidgets import QMessageBox
        with patch('seedvision.ui.main_window.load_bundled_instance_mask', return_value=None), \
             patch('seedvision.ui.main_window.QFileDialog.getOpenFileName') as chooser, \
             patch.object(QMessageBox, 'information') as information:
            self.window.load_instance_labels_action.trigger()
        chooser.assert_not_called()
        self.assertIn('Import seed outlines from file', information.call_args.args[2])
        self.assertNotIn(self.key, self.window._instance_annotations_dirty)

    def check_test_explicit_file_chooser_bypasses_an_available_bundle(self) -> None:
        bundled_labels = np.zeros((60, 80), dtype=np.uint16)
        bundled_labels[2:10, 3:12] = 4
        manual_labels = np.zeros_like(bundled_labels)
        manual_labels[22:39, 31:52] = 27
        selected = self.root / "reviewed-correction.npz"
        np.savez_compressed(selected, labels=manual_labels)

        with patch(
            "seedvision.ui.main_window.load_bundled_instance_mask",
            return_value=self._imported(bundled_labels),
        ) as bundled_loader, patch(
            "seedvision.ui.main_window.QFileDialog.getOpenFileName",
            return_value=(str(selected), "Lossless categorical masks"),
        ):
            self.window.choose_instance_labels_action.trigger()

        bundled_loader.assert_not_called()
        self.assertTrue(
            np.array_equal(
                self.window._draft_instance_annotations[self.key], manual_labels
            )
        )
        self.assertEqual(
            self.window._draft_instance_annotation_origins[self.key],
            f"import:{selected.resolve()}",
        )
        self.assertIn(self.key, self.window._instance_annotations_dirty)
        self.assertEqual(
            self.window._instance_undo_histories[self.key].next_label,
            "imported seed labels",
        )

    def check_test_validation_failure_warns_without_mutating_draft_or_history(self) -> None:
        from seedvision.persistence import InstanceMaskImportError

        baseline = np.zeros((60, 80), dtype=np.uint16)
        baseline[10:20, 10:20] = 4
        self.window._set_instance_draft_state(self.key, baseline, "manual:existing")
        self.window.image_view.set_instance_annotations(
            baseline, copy=False, render=False
        )
        before = baseline.copy()
        with patch(
            "seedvision.ui.main_window.load_bundled_instance_mask",
            side_effect=InstanceMaskImportError("fingerprint or dimensions mismatch"),
        ), patch(
            "seedvision.ui.main_window.QMessageBox.critical"
        ) as warning, patch(
            "seedvision.ui.main_window.QFileDialog.getOpenFileName"
        ) as chooser:
            self.window._load_instance_reference_mask()

        warning.assert_called_once()
        chooser.assert_not_called()
        self.assertTrue(
            np.array_equal(self.window._draft_instance_annotations[self.key], before)
        )
        self.assertEqual(
            self.window._draft_instance_annotation_origins[self.key],
            "manual:existing",
        )
        self.assertNotIn(self.key, self.window._instance_undo_histories)

    def check_test_replacement_cancel_leaves_existing_draft_untouched(self) -> None:
        from PySide6.QtWidgets import QMessageBox

        existing = np.zeros((60, 80), dtype=np.uint16)
        existing[2:7, 2:7] = 1
        replacement = np.zeros_like(existing)
        replacement[30:40, 30:40] = 2
        self.window._set_instance_draft_state(self.key, existing, "manual:existing")
        with patch.object(
            QMessageBox,
            "question",
            return_value=QMessageBox.StandardButton.No,
        ):
            changed = self.window._install_imported_instance_mask(
                self._imported(replacement)
            )
        self.assertFalse(changed)
        self.assertTrue(
            np.array_equal(self.window._draft_instance_annotations[self.key], existing)
        )
        self.assertNotIn(self.key, self.window._instance_undo_histories)

    def _with_cellpose(self, crop: np.ndarray, *, offset=(10, 5)) -> None:
        count = int(crop.max())
        self.result.crop_offset = offset
        self.result.source_sha256 = "c" * 64
        self.result.cellpose_sam_instances = SimpleNamespace(
            labels=crop,
            centres_xy=np.zeros((count, 2), dtype=np.float32),
            checkpoint_id="C:/models/cpsam_v2#sha256=" + "d" * 64,
        )
        self.window._sync_background_controls()

    def check_test_cellpose_seeds_copy_into_an_undoable_outline_draft(self) -> None:
        from PySide6.QtWidgets import QMenu, QMessageBox

        action = self.window.copy_cellpose_outlines_action
        self.assertIn("Seed outlines", [menu.title() for menu in action.associatedObjects()
                                         if isinstance(menu, QMenu)])
        self.assertFalse(action.isEnabled())
        with patch.object(QMessageBox, "information") as information:
            self.window._copy_cellpose_instances_to_outlines()
        information.assert_called_once()
        self.assertNotIn(self.key, self.window._draft_instance_annotations)

        crop = np.zeros((20, 30), dtype=np.int32)
        crop[2:8, 3:9] = 1
        crop[10:18, 15:25] = 2
        self._with_cellpose(crop)
        self.assertTrue(action.isEnabled())
        action.trigger()

        expected = np.zeros((60, 80), dtype=np.uint16)
        expected[5:25, 10:40] = crop
        np.testing.assert_array_equal(self.window._draft_instance_annotations[self.key], expected)
        origin = self.window._draft_instance_annotation_origins[self.key]
        self.assertEqual(origin, "cellpose-sam;weights-sha256=" + "d" * 64)
        self.assertNotIn("models", origin)
        self.assertEqual(self.window._instance_generation_source(self.key), "c" * 64)
        self.assertTrue(self.window.annotate_instances_action.isChecked())
        self.assertEqual(self.window._instance_undo_histories[self.key].next_label,
                         "Cellpose-SAM seed outlines")
        message = self.window.statusBar().currentMessage()
        self.assertIn("Copied 2 Cellpose-SAM seed(s)", message)
        self.assertIn("Apply the draft", message)

        self.window._undo_instance_reference_edit()
        current = self.window._draft_instance_annotations.get(
            self.key, self.window._applied_instance_annotations.get(self.key))
        self.assertTrue(current is None or not np.any(current))

    def check_test_cellpose_seeds_add_to_or_replace_existing_outlines(self) -> None:
        existing = np.zeros((60, 80), dtype=np.uint16)
        existing[5:13, 11:19] = 3
        self.window._set_instance_draft_state(self.key, existing, "manual:existing")
        crop = np.zeros((20, 30), dtype=np.int32)
        crop[1:7, 2:8] = 1  # inside the existing outline once placed
        crop[10:18, 15:25] = 2
        self._with_cellpose(crop)
        predicted = np.zeros((60, 80), dtype=np.uint16)
        predicted[5:25, 10:40] = crop

        with patch.object(self.window, "_ask_cellpose_outline_mode", return_value=None) as ask:
            self.window._copy_cellpose_instances_to_outlines()
        ask.assert_called_once_with(1)
        np.testing.assert_array_equal(self.window._draft_instance_annotations[self.key], existing)
        self.assertNotIn(self.key, self.window._instance_undo_histories)

        with patch.object(self.window, "_ask_cellpose_outline_mode", return_value="add"):
            self.window._copy_cellpose_instances_to_outlines()
        merged = self.window._draft_instance_annotations[self.key]
        np.testing.assert_array_equal(merged[existing > 0], existing[existing > 0])
        np.testing.assert_array_equal(merged == 4, predicted == 2)
        self.assertEqual(set(np.unique(merged)), {0, 3, 4})
        self.assertIn("Added 1 Cellpose-SAM seed(s)", self.window.statusBar().currentMessage())

        with patch.object(self.window, "_ask_cellpose_outline_mode", return_value="add"):
            self.window._copy_cellpose_instances_to_outlines()
        np.testing.assert_array_equal(self.window._draft_instance_annotations[self.key], merged)
        self.assertIn("already outlined", self.window.statusBar().currentMessage())

        with patch.object(self.window, "_ask_cellpose_outline_mode", return_value="replace"):
            self.window._copy_cellpose_instances_to_outlines()
        np.testing.assert_array_equal(self.window._draft_instance_annotations[self.key], predicted)
        self.window._undo_instance_reference_edit()
        np.testing.assert_array_equal(self.window._draft_instance_annotations[self.key], merged)

    def _training_images(self):
        with patch('seedvision.ui.training_dataset.run_persistence', side_effect=lambda window, call, title: call()):
            return self.window._training_images()

    def check_test_training_lists_images_and_explains_unusable_ones(self) -> None:
        from seedvision.persistence.reference_regions import ReferenceRegionBundle, SeedInstanceAnnotation

        # No outlines anywhere: nothing to train on.
        self.assertEqual(self._training_images(), ())
        labels = np.zeros((60, 80), np.uint16)
        labels[10:20, 10:20] = 4
        labels[30:40, 30:40] = 8
        self.window._applied_instance_annotations[self.key] = labels
        self.window._applied_seed_annotations[self.key] = {
            4: SeedInstanceAnnotation(4, shape_reviewed=True, outline_visibility='complete'),
            8: SeedInstanceAnnotation(8, shape_exclusion_reason='damaged')}
        image, = self._training_images()
        self.assertEqual((image.key, image.seed_count), (self.key, 2))
        self.assertIn('8', image.problem)
        self.assertFalse(image.reviewed)
        # Seed 8 now has no review record at all, which does not block training.
        del self.window._applied_seed_annotations[self.key][8]
        image, = self._training_images()
        self.assertIsNone(image.problem)
        self.assertFalse(image.reviewed)
        self.window._applied_seed_annotations[self.key][8] = SeedInstanceAnnotation(
            8, shape_reviewed=True, outline_visibility='complete')
        image, = self._training_images()
        self.assertTrue(image.reviewed)
        self.window._instance_annotations_dirty.add(self.key)
        image, = self._training_images()
        self.assertIn('draft', image.problem)
        self.window._instance_annotations_dirty.clear()
        # A disconnected seed is named, because unlabelled pieces would train as background.
        broken = labels.copy()
        broken[50:55, 50:55] = 4
        self.window._applied_instance_annotations[self.key] = broken
        image, = self._training_images()
        self.assertIn('4', image.problem)

        # Outlines saved beside an image not yet opened are read without installing them.
        del self.window._applied_instance_annotations[self.key]
        del self.window._applied_seed_annotations[self.key]
        self.window._reference_region_autoload_attempted.discard(self.key)
        bundle = SimpleNamespace(annotated_seeds=labels, seed_annotations=(), annotation_species='soybean')
        with patch.object(self.window._reference_region_store, 'load_if_present', return_value=bundle):
            image, = self._training_images()
        self.assertEqual((image.seed_count, image.species), (2, 'Soybean'))
        self.assertNotIn(self.key, self.window._applied_instance_annotations)

    def _training_request(self, *entries, **fields):
        from seedvision.learning.training import TrainingConfiguration
        from seedvision.ui.learning_workflow import LearningTrainingRequest
        return LearningTrainingRequest(
            self.root / 'learning-data' / 'run' / 'manifest.json', self.root / 'model.pt',
            TrainingConfiguration('unet_watershed'), False, images=tuple(entries), **fields)

    def _training_entry(self, identifier, reviewed=False, split='train', key=None, path=None):
        from seedvision.ui.learning_workflow import TrainingDatasetImage
        return TrainingDatasetImage(key or self.key, path or self.image_path, identifier, 'Soybean',
                                    identifier, split, reviewed)

    def check_test_training_asks_before_using_unreviewed_samples(self) -> None:
        from PySide6.QtWidgets import QDialog, QMessageBox

        def start(request, answer=QMessageBox.StandardButton.Cancel):
            dialog = SimpleNamespace(exec=lambda: QDialog.DialogCode.Accepted, request=lambda: request)
            with patch.object(self.window, '_training_images', return_value=(object(),)), \
                 patch('seedvision.ui.main_window.LearningTrainingDialog', return_value=dialog), \
                 patch.object(self.window, '_start_training_dataset_build') as build, \
                 patch.object(QMessageBox, 'question', return_value=answer) as question:
                self.window._start_learning_training()
            return question, build

        request = self._training_request(self._training_entry('kept', True), self._training_entry('checked'))
        question, build = start(request)
        self.assertIn('checked', question.call_args.args[2])
        self.assertNotIn('kept', question.call_args.args[2])
        build.assert_not_called()

        question, build = start(request, QMessageBox.StandardButton.Yes)
        question.assert_called_once()
        self.assertTrue(build.call_args.args[0].configuration.allow_unreviewed)

        question, build = start(self._training_request(self._training_entry('kept', True)))
        question.assert_not_called()
        self.assertFalse(build.call_args.args[0].configuration.allow_unreviewed)

        # No outlines at all says so instead of opening an empty table.
        with patch.object(self.window, '_training_images', return_value=()), \
             patch('seedvision.ui.main_window.LearningTrainingDialog') as dialog, \
             patch.object(QMessageBox, 'information') as information:
            self.window._start_learning_training()
        dialog.assert_not_called()
        self.assertIn('seed outlines', information.call_args.args[2])

    def _build_training_dataset(self, request, analyze, *, exporter=None, audit=None):
        """Run a build with the analysis worker replaced; return what it called."""
        from PySide6.QtWidgets import QMessageBox

        calls = SimpleNamespace(exporter=None, task=None, critical=None, warning=None)

        class Started(Exception):
            pass

        with patch('seedvision.ui.training_dataset.run_persistence', side_effect=lambda window, call, title: call()), \
             patch('seedvision.learning.export.export_analysis_sample',
                   **({'side_effect': exporter} if exporter else {})) as export_call, \
             patch('seedvision.learning.data.audit_manifest',
                   return_value=audit or {'errors': [], 'splits': {'train': 1, 'validation': 1}}), \
             patch('seedvision.ui.main_window._LearningTrainingTask', side_effect=Started) as task, \
             patch.object(self.window, '_analyze_current_image', side_effect=analyze) as run, \
             patch.object(QMessageBox, 'critical') as critical, \
             patch.object(QMessageBox, 'warning') as warning:
            try:
                self.window._start_training_dataset_build(request)
            except Started:
                pass
            calls.exporter, calls.task, calls.critical, calls.warning, calls.run = export_call, task, critical, warning, run
        return calls

    def _training_labels(self, key=None):
        labels = np.zeros((60, 80), np.uint16)
        labels[10:20, 10:20] = 4
        self.window._applied_instance_annotations[key or self.key] = labels
        return labels

    def check_test_finished_training_replaces_the_training_status(self) -> None:
        from PySide6.QtWidgets import QMessageBox

        request = self._training_request(self._training_entry('capture'))
        self.window._learning_training_task = SimpleNamespace(request=request)
        self.window.statusBar().showMessage('Training unet_watershed from manifest.json…')
        with patch.object(self.window, '_pipeline_editable', return_value=True), \
             patch.object(self.window, '_analyze_current_image'), \
             patch.object(QMessageBox, 'information') as information:
            self.window._learning_training_completed({'best_epoch': 1, 'best_validation_loss': 0.5})
        information.assert_called_once()
        self.assertIsNone(self.window._learning_training_task)
        self.assertEqual(self.window.statusBar().currentMessage(),
                         f'Training finished; saved {request.output_path.name}.')

    def check_test_training_build_reuses_an_analysis_and_saves_each_image_then_trains(self) -> None:
        labels = self._training_labels()
        self.window._analysis_scopes.pop(self.key, None)
        request = self._training_request(self._training_entry('capture', True), reviewer='Reviewer')
        calls = self._build_training_dataset(request, lambda **kwargs: self.fail('analysis was current'))
        calls.run.assert_not_called()
        calls.exporter.assert_called_once()
        self.assertIs(calls.exporter.call_args.args[0], self.result)
        np.testing.assert_array_equal(calls.exporter.call_args.args[1], labels)
        arguments = calls.exporter.call_args.kwargs
        self.assertEqual(arguments['identifier'], 'capture')
        self.assertEqual(arguments['dataset_id'], 'run')
        self.assertEqual(arguments['split'], 'train')
        self.assertEqual(arguments['annotation_author'], 'Reviewer')
        self.assertTrue(arguments['reviewed'])
        self.assertTrue(arguments['species_reviewed'])
        calls.task.assert_called_once_with(request)
        self.assertIsNone(self.window._training_dataset_build)

    def check_test_training_build_analyses_with_only_the_features_prerequisites(self) -> None:
        self._training_labels()
        self.window._analyses.pop(self.key)
        scopes = []

        def analyze(*, enabled_node_scope):
            # A completed run installs its result and moves the build on.
            scopes.append(enabled_node_scope)
            self.window._analyses[self.key] = self.result
            self.window._analysis_scopes[self.key] = enabled_node_scope
            self.window._active_tasks[self.key] = object()
            def complete():
                self.window._active_tasks.pop(self.key)
                self.window._advance_training_dataset_build()
            completions.append(complete)

        completions = []
        request = self._training_request(self._training_entry('capture'))
        calls = self._build_training_dataset(request, analyze)
        calls.exporter.assert_not_called()
        self.assertIsNotNone(self.window._training_dataset_build)
        self.assertTrue(self.window._stoppable_work_is_active())
        self.assertIn('deskew_colour', scopes[0])
        self.assertNotIn('unet_instances', scopes[0])
        self.assertNotIn('procedural_instances', scopes[0])
        with patch('seedvision.ui.training_dataset.run_persistence', side_effect=lambda window, call, title: call()), \
             patch('seedvision.learning.export.export_analysis_sample') as exporter, \
             patch('seedvision.learning.data.audit_manifest', return_value={'errors': []}), \
             patch('seedvision.ui.main_window._LearningTrainingTask') as task, \
             patch.object(self.window._thread_pool, 'start'):
            task.return_value.signals = SimpleNamespace(**{name: SimpleNamespace(connect=lambda *_: None)
                for name in ('progress', 'completed', 'failed', 'cancelled')})
            completions.pop()()
        exporter.assert_called_once()
        task.assert_called_once_with(request)
        self.window._finish_learning_training_ui()
        self.window._learning_training_task = None

    def check_test_training_build_stops_on_failure_stop_and_bad_audit(self) -> None:
        self._training_labels()
        entry = self._training_entry('capture')

        # Saving fails: the message names the image and no model trains.
        calls = self._build_training_dataset(self._training_request(entry), lambda **kwargs: None,
                                             exporter=ValueError('Disconnected seed.'))
        self.assertIn('capture.png', calls.critical.call_args.args[2])
        self.assertIn('Disconnected seed.', calls.critical.call_args.args[2])
        calls.task.assert_not_called()
        self.assertIsNone(self.window._training_dataset_build)

        # The audit refuses the saved dataset: no model trains.
        calls = self._build_training_dataset(self._training_request(entry), lambda **kwargs: None,
                                             audit={'errors': ['Group crosses splits.']})
        self.assertIn('Group crosses splits.', calls.warning.call_args.args[2])
        calls.task.assert_not_called()

        # Analysis never produces a result: give up instead of looping forever.
        self.window._analyses.pop(self.key)
        def analyze_without_result(**kwargs):
            self.assertLess(calls_seen.append(1) or len(calls_seen), 4)
            self.window._active_tasks[self.key] = object()
            self.window._active_tasks.pop(self.key)
        calls_seen = []
        calls = self._build_training_dataset(self._training_request(entry), analyze_without_result)
        self.assertIn('Analysis could not start', calls.critical.call_args.args[2])
        calls.exporter.assert_not_called()

        # Stop during analysis ends the build with nothing trained.
        task = SimpleNamespace(cancel=lambda: None)
        def analyze_then_wait(**kwargs):
            self.window._active_tasks[self.key] = task
        calls = self._build_training_dataset(self._training_request(entry), analyze_then_wait)
        self.window._stop_current_work()
        self.assertTrue(self.window._training_dataset_build.stopped)
        self.window._active_tasks.pop(self.key)
        self.window._advance_training_dataset_build()
        self.assertIsNone(self.window._training_dataset_build)
        self.assertIn('no model was trained', self.window.statusBar().currentMessage())
        calls.task.assert_not_called()

        # Failure during analysis ends it too; the analysis failure already showed its message.
        self.window._stop_requested = False
        calls = self._build_training_dataset(self._training_request(entry), analyze_then_wait)
        self.window._training_dataset_build.failed = True
        self.window._active_tasks.pop(self.key)
        self.window._advance_training_dataset_build()
        self.assertIsNone(self.window._training_dataset_build)

    def check_test_seed_label_export_uses_applied_16_bit_labels(self) -> None:
        import cv2
        from PySide6.QtWidgets import QFileDialog
        applied = np.zeros((60, 80), np.uint16)
        applied[10:20, 10:20] = 31777
        draft = applied.copy()
        draft[30:40, 30:40] = 42
        self.window._applied_instance_annotations[self.key] = applied
        self.window._draft_instance_annotations[self.key] = draft
        invalid = self.root / 'lossy.jpg'
        from PySide6.QtWidgets import QMessageBox
        with patch.object(QFileDialog, 'getSaveFileName', return_value=(str(invalid), '')), \
             patch.object(QMessageBox, 'critical') as warning:
            self.window._save_instance_labels()
        self.assertFalse(invalid.exists())
        self.assertIn('lossless 16-bit', warning.call_args.args[2])
        destination = self.root / 'export.png'
        with patch.object(QFileDialog, 'getSaveFileName', return_value=(str(destination), '')):
            self.window._save_instance_labels()
        loaded = cv2.imread(str(destination), cv2.IMREAD_UNCHANGED)
        self.assertEqual(loaded.dtype, np.uint16)
        np.testing.assert_array_equal(loaded, applied)
        self.window._unsaved_reference_sidecars.add(self.key)
        self.window._sync_background_controls()
        self.assertIn('disk save failed', self.window.instance_annotation_confirmation_label.text())

    def test_workflow_lifecycle(self):
        """13 related InstanceMaskImportUiTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_panel_import_is_draft_with_provenance_warning_and_undo)
        self.scenario(self.check_test_bundle_import_is_explicit_and_preserves_applied_labels)
        self.scenario(self.check_test_no_bundle_explains_explicit_file_import)
        self.scenario(self.check_test_explicit_file_chooser_bypasses_an_available_bundle)
        self.scenario(self.check_test_training_lists_images_and_explains_unusable_ones)
        self.scenario(self.check_test_training_asks_before_using_unreviewed_samples)
        self.scenario(self.check_test_training_build_reuses_an_analysis_and_saves_each_image_then_trains)
        self.scenario(self.check_test_training_build_analyses_with_only_the_features_prerequisites)
        self.scenario(self.check_test_training_build_stops_on_failure_stop_and_bad_audit)
        self.scenario(self.check_test_finished_training_replaces_the_training_status)
        self.scenario(self.check_test_seed_label_export_uses_applied_16_bit_labels)
        self.scenario(self.check_test_cellpose_seeds_copy_into_an_undoable_outline_draft)
        self.scenario(self.check_test_cellpose_seeds_add_to_or_replace_existing_outlines)

    def test_validation_and_recovery(self):
        """2 related InstanceMaskImportUiTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_validation_failure_warns_without_mutating_draft_or_history)
        self.scenario(self.check_test_replacement_cancel_leaves_existing_draft_untouched)


if __name__ == "__main__":
    unittest.main()
