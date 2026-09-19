import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest
import numpy as np
import cv2


class ReviewWorkflowTests(unittest.TestCase):
    def test_window_fits_1080p_at_two_hundred_percent_logical_size(self):
        from seedvision.ui.main_window import MainWindow
        with TemporaryDirectory() as directory:
            window = MainWindow(Path(directory))
            window.resize(960,540)
            window.show()
            self.app.processEvents()
            self.assertLessEqual(window.width(),960)
            self.assertLessEqual(window.height(),540)
            self.assertTrue(window.image_view.isVisible())
            window.close()

    def test_results_filter_navigation_review_history_and_method_comparison(self):
        from types import SimpleNamespace
        from PySide6.QtWidgets import QDialog,QTableWidget,QComboBox,QPushButton,QMessageBox
        from seedvision.ui.main_window import MainWindow
        with TemporaryDirectory() as directory:
            window = MainWindow(Path(directory))
            labels = np.zeros((16,16),np.int32)
            labels[1:4,1:4]=1; labels[6:9,6:9]=2
            window._analyses[None] = SimpleNamespace(image_path=None,crop_offset=(0,0),warnings=(),
                calibration=SimpleNamespace(corrected_bgr=np.zeros((16,16,3),np.uint8),affine_matrix=np.eye(3)),
                layers=SimpleNamespace(foreground_colour_profile=object()),
                procedural_instances=SimpleNamespace(labels=labels,instance_confidences=np.array([.8,.7]),source_shape=(16,16)))
            def exercise(dialog):
                table = dialog.findChild(QTableWidget)
                table.cellWidget(0,1).setCurrentText('accepted')
                selector = next(item for item in dialog.findChildren(QComboBox) if item.accessibleName()=='Filter seeds by review decision')
                selector.setCurrentText('unreviewed')
                self.assertTrue(table.isRowHidden(0))
                buttons = dialog.findChildren(QPushButton)
                next(item for item in buttons if item.text().startswith('Next matching')).click()
                self.assertEqual(table.currentRow(),1)
                next(item for item in buttons if item.text().startswith('Compare completed')).click()
                return QDialog.DialogCode.Accepted
            with patch.object(QDialog,'exec',new=exercise),patch.object(QMessageBox,'information') as information:
                window._review_results()
            self.assertIn('2 proposed instances',information.call_args.args[2])
            self.assertEqual(len(list((Path(directory)/'result-reviews'/'history').rglob('*.json'))),1)
            window.close()

    def test_batch_project_does_not_require_an_optional_species_library(self):
        import shutil
        from seedvision.persistence import ProjectAnalysisStore,ProjectImageSpec,analysis_settings_profile_from_graph
        from seedvision.pipeline import build_default_pipeline
        from seedvision.export.project_runner import run_project
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'config').mkdir()
            shutil.copyfile(Path(__file__).parents[1]/'config'/'traits.json',root/'config'/'traits.json')
            source = root/'image.png'
            cv2.imwrite(str(source),np.zeros((16,16,3),np.uint8))
            master = ProjectAnalysisStore(root).capture_and_save(
                analysis_settings=analysis_settings_profile_from_graph(build_default_pipeline()),
                images=(ProjectImageSpec(source,(16,16)),),destination=root/'project.seedfiddle-project.json')
            with patch('seedvision.export.project_runner.analyze_path') as analyze, \
                 patch('seedvision.export.project_runner.result_report',return_value={'state':'unavailable'}), \
                 patch('seedvision.export.project_runner.export_report',return_value=root/'result'):
                self.assertEqual(run_project(root,master,root/'results'),[root/'result'])
            self.assertIsNone(analyze.call_args.kwargs['species_library'])
            self.assertIn('layer_settings',analyze.call_args.kwargs)

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_loose_close_cancel_preserves_draft_and_discard_is_explicit(self):
        from PySide6.QtWidgets import QMessageBox
        from PySide6.QtGui import QCloseEvent
        from seedvision.ui.main_window import MainWindow
        with TemporaryDirectory() as directory:
            window = MainWindow(Path(directory))
            key = 'unsaved-test-image'
            labels = np.ones((8,8),np.uint16)
            window._draft_instance_annotations[key] = labels
            window._instance_annotations_dirty.add(key)
            with patch.object(QMessageBox,'warning',return_value=QMessageBox.StandardButton.Cancel):
                event = QCloseEvent()
                window.closeEvent(event)
            self.assertFalse(event.isAccepted())
            self.assertIs(window._draft_instance_annotations[key],labels)
            with patch.object(QMessageBox,'warning',return_value=QMessageBox.StandardButton.Discard):
                event = QCloseEvent()
                window.closeEvent(event)
            self.assertTrue(event.isAccepted())
            self.assertNotIn(key,window._instance_annotations_dirty)

    def test_failed_autosave_blocks_close(self):
        from PySide6.QtGui import QCloseEvent
        from seedvision.ui.main_window import MainWindow
        with TemporaryDirectory() as directory:
            window = MainWindow(Path(directory))
            with patch.object(window,'_retry_unsaved_project_sidecars',return_value=False):
                event = QCloseEvent()
                window.closeEvent(event)
            self.assertFalse(event.isAccepted())
            window.close()

    def test_annotation_redo_and_byte_budget(self):
        from seedvision.ui.reference_history import RasterUndoHistory
        before = np.zeros((32,32),np.uint16)
        after = before.copy();after[4:9,7:12] = 3
        history = RasterUndoHistory()
        history.record('paint','instances',(before,),(after,),metadata='before')
        undo = history.undo((after,),metadata='after')
        np.testing.assert_array_equal(undo.rasters[0],before)
        redo = history.redo(undo.rasters,metadata='before')
        np.testing.assert_array_equal(redo.rasters[0],after)
        self.assertEqual(redo.metadata,'after')
        bounded = RasterUndoHistory(maximum_bytes=1)
        bounded.record('paint','instances',(before,),(after,))
        self.assertLessEqual(bounded.stored_bytes,1)

    def test_snapshot_preserves_old_annotations_and_restores_relocated_project(self):
        from seedvision.persistence.snapshot import create_snapshot, restore_snapshot
        from seedvision.persistence import (ReferenceRegionStore,ReferenceRegionBundle,ProjectAnalysisStore,
            ProjectImageSpec,analysis_settings_profile_from_graph)
        from seedvision.pipeline import build_default_pipeline
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image = root/'images'/'one.png';image.parent.mkdir()
            cv2.imwrite(str(image),np.zeros((16,16,3),np.uint8))
            mask = np.zeros((16,16),bool);mask[4:8,4:8]=True
            references = ReferenceRegionStore(root)
            references.save(image,ReferenceRegionBundle(shape=mask.shape,foreground=mask,source_to_corrected=np.eye(3)))
            store = ProjectAnalysisStore(root)
            master = store.capture_and_save(analysis_settings=analysis_settings_profile_from_graph(build_default_pipeline()),
                images=(ProjectImageSpec(image,(16,16)),),destination=root/'project.seedfiddle-project.json')
            snapshot = create_snapshot(root,master,root/'snapshots')
            references.save(image,ReferenceRegionBundle(shape=mask.shape,source_to_corrected=np.eye(3)))
            restored = restore_snapshot(snapshot,root)
            loaded = store.load(restored)
            self.assertFalse(loaded.issues)
            entry = loaded.images[0]
            old = references.load_project_archive(entry.path,entry.sidecar('reference_regions').path,None)
            np.testing.assert_array_equal(old.foreground,mask)

    def test_result_decisions_are_bound_to_prediction_revision(self):
        from types import SimpleNamespace
        from seedvision.export.results import result_report
        labels = np.zeros((16,16),np.int32);labels[4:8,4:8] = 1
        instances = SimpleNamespace(labels=labels,instance_confidences=np.array([.8]),source_shape=(16,16))
        analysis = SimpleNamespace(image_path=None,crop_offset=(10,20),warnings=(),
            calibration=SimpleNamespace(corrected_bgr=np.zeros((40,40,3),np.uint8),affine_matrix=np.eye(3)),
            layers=SimpleNamespace(foreground_colour_profile=object()),procedural_instances=instances)
        report = result_report(analysis)
        review = {'result_revision':report['result_revision'],'decisions':{'1':'accepted'}}
        self.assertEqual(result_report(analysis,review=review)['accepted_count'],1)
        labels[8,8]=1
        self.assertEqual(result_report(analysis,review=review)['accepted_count'],0)

    def test_validation_helpers_do_not_certify_science(self):
        from seedvision.learning.validation_tools import score_reliability,assess_metric_geometry
        self.assertFalse(score_reliability([.1,.9],[0,1])['scientifically_validated'])
        report = assess_metric_geometry([{'location':str(i),'direction':direction,'reference_mm':10,'measured_mm':10.1}
            for i in range(3) for direction in ('horizontal','vertical')],maximum_relative_error=.02)
        self.assertTrue(report['within_predeclared_bound'])
        self.assertFalse(report['scientifically_validated'])
