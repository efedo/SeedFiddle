"""Native label producers bind to the displayed corrected source, including history."""
from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from seedvision.persistence.instance_masks import ImportedInstanceMask
from seedvision.persistence.reference_regions import ReferenceRegionBundle, file_sha256
from seedvision.ui.annotation_canvas import AnnotationEditContext, displayed_canvas_binding
from tests.review_fixtures import dispose_window


class AnnotationGenerationBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        from seedvision.calibration import image as calibration
        from seedvision.calibration.geometry import DishDetectionSettings
        from seedvision.segmentation.baseline import analyze_path
        from tests.test_neutral_calibration import fixture, ruler
        from tests.test_project_analysis_ui import ProjectAnalysisUiTests

        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'images' / 'a.png'
        self.source.parent.mkdir()
        image, card = fixture()
        self.assertTrue(cv2.imwrite(str(self.source), image))
        with patch.object(calibration, 'detect_colour_card', return_value=card), patch.object(
            calibration, 'detect_ruler', side_effect=ruler
        ):
            self.result = analyze_path(self.source,
                calibration_settings=calibration.CalibrationSettings(apply_perspective_correction=False),
                dish_settings=DishDetectionSettings(expected_outer_diameter_mm=0), enabled_nodes=frozenset())
        self.digest = file_sha256(self.source)
        self.assertEqual(self.result.source_sha256, self.digest)
        self.window = ProjectAnalysisUiTests._window(self.root)
        self.addCleanup(dispose_window, self.window)
        self.window.resize(960, 700)
        self.window.show()
        self.window._analyze_current_image = lambda **kwargs: None
        self._open(self.result)

    def _open(self, result):
        from seedvision.ui.main_window import _path_identity
        view = self.window.image_view
        view.set_overlay_mode('none')
        view.load_image(result.image_path)
        self.key = _path_identity(result.image_path)
        self.window._image_paths[self.key] = result.image_path
        self.window._analyses[self.key] = result
        self.assertTrue(self.window._prepare_reference_frame(self.key, result))
        view.show_analysis(result, render=False)
        view.prepare_analysis_coordinates()
        self.window._sync_reference_masks_to_view(self.key)
        self.window.annotate_instances_action.setChecked(True)
        self.window._sync_background_controls()
        self.application.processEvents()
        view.fit_image()

    def labels(self, *, second=False):
        labels = np.zeros(self.result.calibration.corrected_bgr.shape[:2], np.uint16)
        labels[20:29, 30:39] = 1
        if second:
            labels[35:44, 55:64] = 2
        return labels

    def deliver(self, labels, context=None):
        view = self.window.image_view
        view.instance_annotations_generated.emit(labels, context or view.annotation_edit_context())

    def generation(self):
        return self.window._instance_generation_source(self.key)

    def current_labels(self):
        return self.window._instance_reference_state(self.key)[0]

    def test_actual_mouse_brush_binds_source_and_retains_legacy_signal(self):
        from PySide6.QtCore import QPointF, Qt
        from PySide6.QtTest import QTest
        view = self.window.image_view
        generated, legacy = [], []
        view.instance_annotations_generated.connect(lambda labels, context: generated.append(context))
        view.instance_annotations_edited.connect(lambda labels: legacy.append(np.array(labels)))
        point = view.mapFromScene(QPointF(35, 25))
        QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=point)
        self.application.processEvents()
        self.assertTrue(np.any(self.current_labels()))
        self.assertEqual(self.generation(), self.digest)
        self.assertEqual(len(generated), 1)
        self.assertEqual(len(legacy), 1)
        self.assertEqual(generated[0].binding.source_sha256, self.digest)
        self.assertEqual(generated[0].binding.corrected_image_sha256,
            sha256(self.result.calibration.corrected_bgr.tobytes()).hexdigest())
        with self.assertRaises(FrozenInstanceError):
            generated[0].epoch = -1
        self.assertEqual(self.result.calibration.gpu_corrected_bgr.device.type, 'cuda')

    def test_first_edit_known_edits_clear_and_fresh_drawing(self):
        self.deliver(self.labels())
        self.assertEqual(self.generation(), self.digest)
        self.deliver(self.labels(second=True))
        self.assertEqual(self.generation(), self.digest)
        self.window._clear_all_instance_annotations()
        self.assertIsNone(self.generation())
        self.deliver(self.labels())
        self.assertEqual(self.generation(), self.digest)

    def test_legacy_partial_and_trait_edits_do_not_upgrade_history(self):
        self.window._instance_annotations_edited(self.labels())
        self.assertIsNone(self.generation())
        self.deliver(self.labels(second=True))
        self.assertIsNone(self.generation())
        self.window.seed_conditions_reviewed_checkbox.setChecked(True)
        self.assertIsNone(self.generation())
        self.assertTrue(self.window._draft_seed_annotations[self.key][1].conditions_reviewed)

    def test_undo_redo_restore_known_and_explicit_unknown_atomically(self):
        self.window._instance_annotations_edited(self.labels())
        self.window._clear_all_instance_annotations()
        self.deliver(self.labels(second=True))
        self.assertEqual(self.generation(), self.digest)
        self.window._undo_instance_reference_edit()
        self.assertFalse(np.any(self.current_labels()))
        self.assertIsNone(self.generation())
        self.window._undo_instance_reference_edit()
        np.testing.assert_array_equal(self.current_labels(), self.labels())
        self.assertIsNone(self.generation())
        self.window._undo_instance_reference_edit(redo=True)
        self.window._undo_instance_reference_edit(redo=True)
        np.testing.assert_array_equal(self.current_labels(), self.labels(second=True))
        self.assertEqual(self.generation(), self.digest)

    def test_revert_restores_saved_source_and_discards_pending_event(self):
        saved = self.labels()
        self.window._install_reference_region_bundle(self.key, ReferenceRegionBundle(saved.shape,
            annotated_seeds=saved, source_to_corrected=self.result.calibration.affine_matrix,
            archive_source_sha256=self.digest, annotation_snapshot_source_sha256=self.digest))
        context = self.window.image_view.annotation_edit_context()
        self.window._instance_annotations_edited(self.labels(second=True))
        self.assertIsNone(self.generation())
        self.window._revert_instance_annotations()
        self.assertEqual(self.generation(), self.digest)
        self.deliver(self.labels(second=True), context)
        np.testing.assert_array_equal(self.current_labels(), saved)
        self.assertEqual(self.generation(), self.digest)

    def test_queued_old_result_and_empty_events_are_rejected(self):
        from PySide6.QtCore import Qt
        view = self.window.image_view
        self.deliver(self.labels())
        before = self.current_labels().copy()
        view.instance_annotations_generated.disconnect(self.window._instance_annotations_generated)
        view.instance_annotations_generated.connect(self.window._instance_annotations_generated,
            Qt.ConnectionType.QueuedConnection)
        old = view.annotation_edit_context()
        self.deliver(np.zeros_like(before), old)
        self.deliver(self.labels(second=True), old)
        newer = replace(self.result)
        self.window._analyses[self.key] = newer
        view.show_analysis(newer, render=False)
        view.prepare_analysis_coordinates()
        self.application.processEvents()
        np.testing.assert_array_equal(self.current_labels(), before)
        self.assertEqual(self.generation(), self.digest)

    def test_same_result_source_round_trip_and_history_replacement_invalidate_events(self):
        view = self.window.image_view
        self.deliver(self.labels())
        for replace_history in (lambda: self.window._undo_instance_reference_edit(),
            self.window._clear_all_instance_annotations,
            lambda: (view._restore_source_image(), view.prepare_analysis_coordinates())):
            self.deliver(self.labels())
            context = view.annotation_edit_context()
            replace_history()
            expected = self.current_labels().copy()
            self.deliver(self.labels(second=True), context)
            np.testing.assert_array_equal(self.current_labels(), expected)

    def test_unknown_source_queued_event_cannot_cross_image_switch(self):
        self._open(replace(self.result, source_sha256=None))
        view = self.window.image_view
        old = view.annotation_edit_context()
        self.assertIsNone(old.binding)
        second = self.source.with_name('b.png')
        second.write_bytes(self.source.read_bytes())
        self._open(replace(self.result, image_path=second))
        self.deliver(self.labels(), old)
        self.assertFalse(np.any(self.current_labels()))
        self.assertIsNone(self.generation())

    def test_two_queued_initial_edits_preserve_order_and_source(self):
        from PySide6.QtCore import Qt
        view = self.window.image_view
        view.instance_annotations_generated.disconnect(self.window._instance_annotations_generated)
        view.instance_annotations_generated.connect(self.window._instance_annotations_generated,
            Qt.ConnectionType.QueuedConnection)
        context = view.annotation_edit_context()
        self.deliver(self.labels(), context)
        self.deliver(self.labels(second=True), context)
        self.application.processEvents()
        np.testing.assert_array_equal(self.current_labels(), self.labels(second=True))
        self.assertEqual(self.generation(), self.digest)
        self.assertEqual(len(self.window._instance_undo_histories[self.key]), 2)

    def test_same_shape_changed_transform_and_controller_result_do_not_bind(self):
        view = self.window.image_view
        context = view.annotation_edit_context()
        altered = np.array(self.result.calibration.affine_matrix, copy=True)
        altered[0, 2] += 1
        self.window._reference_transforms[self.key] = altered
        self.deliver(self.labels(), context)
        self.assertFalse(np.any(self.current_labels()))
        self.window._reference_transforms[self.key] = self.result.calibration.affine_matrix
        self.window._analyses[self.key] = replace(self.result)
        self.deliver(self.labels(), context)
        self.assertFalse(np.any(self.current_labels()))

    def test_raw_diagnostic_stale_or_copied_buffers_cannot_establish_proof(self):
        view = self.window.image_view
        self.assertIsNone(displayed_canvas_binding(self.result,
            self.result.calibration.corrected_bgr.copy(), 0))
        for mode in ('source', 'gamut', 'prototype_collage', 'stale'):
            with self.subTest(mode=mode):
                view._displayed_base = mode
                self.assertIsNone(view.annotation_canvas_binding())
        view.prepare_analysis_coordinates()
        self.assertIsNotNone(view.annotation_canvas_binding())

    def test_external_pipeline_origin_is_unknown_and_supported_saved_identity_restores(self):
        imported = ImportedInstanceMask(self.labels(), self.source, 'corrected', 'pipeline:historical')
        self.assertTrue(self.window._install_imported_instance_mask(imported))
        self.assertIsNone(self.generation())
        self.window._revert_instance_annotations()
        self.assertTrue(self.window._install_imported_instance_mask(imported, generation_source=self.digest))
        self.assertEqual(self.generation(), self.digest)

    def test_apply_autosave_preserves_source_before_analysis_invalidation(self):
        self.deliver(self.labels())
        self.window._apply_instance_annotations()
        loaded = self.window._reference_region_store.load_if_present(self.source)
        np.testing.assert_array_equal(loaded.annotated_seeds, self.labels())
        self.assertEqual(loaded.annotation_snapshot_source_sha256, self.digest)
        self.assertEqual(self.window._reference_source_identities[self.key][1], self.digest)
        self.assertNotIn(self.key, self.window._draft_instance_generation_sources)
        self.assertNotIn(self.key, self.window._analyses)

    def test_bulk_two_image_drafts_keep_known_and_unknown_source_overrides(self):
        self.deliver(self.labels())
        first_key = self.key
        second = self.source.with_name('b.png')
        second.write_bytes(self.source.read_bytes())
        self._open(replace(self.result, image_path=second))
        self.window._reference_source_identities[self.key] = (self.digest, self.digest)
        self.window._instance_annotations_edited(self.labels(second=True))
        self.assertIsNone(self.generation())
        self.assertEqual(self.window._draft_instance_generation_sources[self.key], None)
        self.assertTrue(self.window._apply_and_save_all_project_drafts({first_key, self.key}, recompute=False))
        first = self.window._reference_region_store.load_if_present(self.source)
        other = self.window._reference_region_store.load_if_present(second)
        self.assertEqual(first.annotation_snapshot_source_sha256, self.digest)
        self.assertIsNone(other.annotation_snapshot_source_sha256)
        np.testing.assert_array_equal(other.annotated_seeds, self.labels(second=True))
        self.window._clear_project_session_state()
        self.assertEqual(self.window._draft_instance_generation_sources, {})

    def test_identical_unbound_notification_does_not_erase_known_source(self):
        self.deliver(self.labels())
        self.window._instance_annotations_edited(self.current_labels().copy())
        self.assertEqual(self.generation(), self.digest)

    def test_known_trait_edits_preserve_outline_source_without_repainting(self):
        self.deliver(self.labels())
        before = self.current_labels().copy()
        self.window.seed_conditions_reviewed_checkbox.setChecked(True)
        np.testing.assert_array_equal(self.current_labels(), before)
        self.assertEqual(self.generation(), self.digest)

    def test_wrong_label_shape_cannot_acquire_corrected_canvas_identity(self):
        self.deliver(np.ones((10, 10), np.uint16))
        self.assertFalse(np.any(self.current_labels()))
        self.assertIsNone(self.generation())


if __name__ == '__main__':
    unittest.main()
