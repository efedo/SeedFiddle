"""Actual async compact plots, model-first saves and full modal lifecycle."""
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PySide6.QtCore import QSaveFile, QTimer
from PySide6.QtWidgets import QApplication, QDialog

from seedvision.benchmarking import naive_bayes_training_directory as directory
from seedvision.benchmarking import native_naive_bayes_training as training
from seedvision.benchmarking import native_naive_bayes_recipe as factory
from seedvision.export import naive_bayes_plots as plots
from seedvision.ui import native_naive_bayes_training_dialog as ui
from seedvision.ui import native_naive_bayes_dialog as parent_ui
from tests.test_native_naive_bayes_training import sample_bytes, constant_value_samples
from tests import test_native_naive_bayes_training_dialog as sample_workflow
from tests.test_plantcv_naive_bayes_sampling import coordinate_pair


class DenseSentinel:
    def __getattribute__(self,name):
        raise AssertionError('Plot save traversed dense training products')


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required for actual training UI')
class NativeBayesPlotDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.samples = self.root/'samples.tsv'
        self.samples.write_bytes(sample_bytes(7))
        completion = training.train_bayes_sample_file(self.samples)
        self.completion = replace(completion,samples=DenseSentinel(),fit=DenseSentinel())
        self.dialogs = []

    tearDown = sample_workflow.NativeBayesTrainingWorkflowTests.tearDown
    wait = sample_workflow.NativeBayesTrainingWorkflowTests.wait

    def dialog(self,**kwargs):
        dialog = ui.NativeNaiveBayesTrainingDialog(**kwargs)
        dialog.samples_edit.setText(str(self.samples))
        dialog._run_path = str(self.samples.resolve())
        dialog._run_revision = dialog._revision
        dialog._completion = self.completion
        dialog.save_button.setEnabled(True)
        self.dialogs.append(dialog)
        return dialog

    def save(self,dialog,path):
        with patch.object(ui.QFileDialog,'getSaveFileName',return_value=(str(path),'tsv')):
            dialog._save_model()

    def test_default_off_and_real_async_plots_preserve_tsv_identity_without_dense_access(self):
        plain = self.dialog()
        self.assertFalse(plain.save_plots_checkbox.isChecked())
        self.save(plain,self.root/'plain.tsv')
        self.assertEqual(plain.result(),QDialog.DialogCode.Accepted)
        self.assertFalse(any((self.root/name).exists() for name in plots.FILENAMES))
        self.assertEqual(plain.saved_plot_paths,())
        dialog = self.dialog()
        dialog.save_plots_checkbox.setChecked(True)
        self.samples.unlink()
        with (patch.object(torch.Tensor,'cpu',side_effect=AssertionError('dense CPU transfer')),
                patch.object(torch.Tensor,'numpy',side_effect=AssertionError('dense NumPy transfer'))):
            self.save(dialog,self.root/'with-plots.tsv')
            self.assertIsNotNone(dialog._worker)
            self.assertFalse(dialog.save_plots_checkbox.isEnabled())
            self.wait(dialog)
        self.assertEqual(dialog.result(),QDialog.DialogCode.Accepted)
        self.assertEqual((self.root/'with-plots.tsv').read_bytes(),self.completion.tsv_bytes)
        self.assertEqual(dialog.saved_model_source['source_sha256'],plain.saved_model_source['source_sha256'])
        self.assertEqual(factory.load_selected_model(dialog.saved_model_source),self.completion.model)
        self.assertEqual(tuple(Path(p).name for p in dialog.saved_plot_paths),plots.FILENAMES)
        metadata = json.loads(dialog.rendered_plot_metadata)
        for record in metadata['plots']:
            self.assertEqual((self.root/record['filename']).stat().st_size,record['bytes'])
        self.assertEqual(dialog.plot_save_errors,())

    def test_actual_constant_completion_plots_preserve_peak_and_training_record(self):
        self.samples.write_bytes(constant_value_samples())
        completion = training.train_bayes_sample_file(self.samples)
        self.assertGreater(completion.fit.channels[0][2].mean,255)
        self.completion = replace(completion,samples=DenseSentinel(),fit=DenseSentinel())
        dialog = self.dialog()
        dialog.save_plots_checkbox.setChecked(True)
        self.samples.unlink()
        with (patch.object(torch.Tensor,'cpu',side_effect=AssertionError('dense CPU transfer')),
                patch.object(torch.Tensor,'numpy',side_effect=AssertionError('dense NumPy transfer'))):
            self.save(dialog,self.root/'constant-with-plots.tsv')
            self.wait(dialog)
        self.assertEqual(dialog.result(),QDialog.DialogCode.Accepted)
        self.assertEqual((self.root/'constant-with-plots.tsv').read_bytes(),completion.tsv_bytes)
        selected = dialog.saved_model_source
        self.assertEqual(factory.load_selected_model(selected),completion.model)
        metadata = json.loads(selected[factory.TRAINING_FIELD]['json'])
        self.assertEqual(metadata['channel_fits'][0][2]['constant_value'],255)
        self.assertGreater(metadata['channel_fits'][0][2]['mean'],255)
        self.assertEqual(tuple(Path(p).name for p in dialog.saved_plot_paths),plots.FILENAMES)
        self.assertTrue(all(Path(p).read_bytes().startswith(b'<?xml') for p in dialog.saved_plot_paths))

    def test_actual_directory_completion_uses_same_compact_plot_save_after_source_removal(self):
        image_root,mask_root = self.root/'images',self.root/'masks'
        image_root.mkdir(); mask_root.mkdir()
        image = np.random.default_rng(24).integers(0,256,(31,37,3),dtype=np.uint8)
        mask = np.zeros((31,37),dtype=np.uint8); mask.reshape(-1)[:555] = 255
        self.assertTrue(cv2.imwrite(str(image_root/'seed.png'),image))
        self.assertTrue(cv2.imwrite(str(mask_root/'seed.png'),mask))
        completion = directory.train_bayes_directories(image_root,mask_root,rng=np.random.RandomState(17))
        dialog = self.dialog()
        dialog.input_mode.setCurrentIndex(1)
        dialog.image_directory_edit.setText(str(image_root))
        dialog.mask_directory_edit.setText(str(mask_root))
        dialog._run_mode = 'directories'
        dialog._run_path = (str(image_root.resolve()),str(mask_root.resolve()))
        dialog._run_revision = dialog._revision
        dialog._completion = replace(completion,samples=DenseSentinel(),fit=DenseSentinel(),selection=DenseSentinel())
        for root in (image_root,mask_root):
            (root/'seed.png').unlink(); root.rmdir()
        dialog.save_plots_checkbox.setChecked(True)
        self.save(dialog,self.root/'binary.tsv'); self.wait(dialog)
        self.assertEqual(dialog.result(),QDialog.DialogCode.Accepted)
        self.assertEqual(factory.load_selected_model(dialog.saved_model_source),completion.model)
        self.assertEqual(len(dialog.saved_plot_paths),3)

    def test_different_mask_dimensions_plot_save_keeps_v2_record_without_dense_or_sources(self):
        image_root,mask_root = self.root/'coordinate-images',self.root/'coordinate-masks'
        image_root.mkdir(); mask_root.mkdir()
        image,mask = coordinate_pair((20,19),(31,37))
        self.assertTrue(cv2.imwrite(str(image_root/'seed.png'),image))
        self.assertTrue(cv2.imwrite(str(mask_root/'seed.png'),mask))
        completion = directory.train_bayes_directories(image_root,mask_root,rng=np.random.RandomState(17))
        dialog = self.dialog()
        dialog.input_mode.setCurrentIndex(1)
        dialog.image_directory_edit.setText(str(image_root))
        dialog.mask_directory_edit.setText(str(mask_root))
        dialog._run_mode = 'directories'
        dialog._run_path = (str(image_root.resolve()),str(mask_root.resolve()))
        dialog._run_revision = dialog._revision
        dialog._completion = replace(completion,samples=DenseSentinel(),fit=DenseSentinel(),selection=DenseSentinel())
        for root in (image_root,mask_root):
            (root/'seed.png').unlink(); root.rmdir()
        dialog.save_plots_checkbox.setChecked(True)
        with (patch.object(torch.Tensor,'cpu',side_effect=AssertionError('dense CPU transfer')),
                patch.object(torch.Tensor,'numpy',side_effect=AssertionError('dense NumPy transfer'))):
            self.save(dialog,self.root/'coordinate-plots.tsv'); self.wait(dialog)
        self.assertEqual(dialog.result(),QDialog.DialogCode.Accepted)
        self.assertEqual(factory.load_selected_model(dialog.saved_model_source),completion.model)
        record = json.loads(dialog.saved_model_source[factory.TRAINING_FIELD]['json'])
        self.assertEqual(record['parser_id'],'plantcv-4.11.3-ordered-binary-directories-v2')
        self.assertEqual(record['occurrences'][0]['mask_shape'],[31,37])
        self.assertEqual(record['occurrences'][0]['image_shape'],[20,19,3])
        self.assertEqual(len(dialog.saved_plot_paths),3)

    def test_render_errors_and_invalid_bundle_write_nothing_and_allow_model_only_retry(self):
        bundle = plots.render_naive_bayes_plots(self.completion.model)
        for operation in [lambda *a,**k:(_ for _ in ()).throw(plots.BayesPlotError('runtime unavailable')),
                lambda *a,**k:replace(bundle,model_sha256='a'*64)]:
            dialog = self.dialog(plot_operation=operation)
            dialog.save_plots_checkbox.setChecked(True)
            target = self.root/'failed-render.tsv'
            self.save(dialog,target); self.wait(dialog)
            self.assertFalse(target.exists())
            self.assertIn('no files saved',dialog.status_label.text())
            self.assertIsNone(dialog.saved_model_source)
            self.assertTrue(dialog.save_button.isEnabled())
            dialog.save_plots_checkbox.setChecked(False)
            self.save(dialog,target)
            self.assertIsNotNone(dialog.saved_model_source)
            target.unlink()
        self.assertFalse(any((self.root/name).exists() for name in plots.FILENAMES))

    def test_cancel_changed_input_accept_reject_and_close_drain_plot_worker_without_files(self):
        bundle = plots.render_naive_bayes_plots(self.completion.model)
        for action in ['stop','change','accept','reject','close']:
            entered,release = threading.Event(),threading.Event()
            def operation(*args,**kwargs):
                entered.set(); release.wait(5)
                return bundle
            dialog = self.dialog(plot_operation=operation)
            dialog.save_plots_checkbox.setChecked(True)
            target = self.root/(action+'.tsv')
            self.save(dialog,target)
            self.assertTrue(entered.wait(2))
            if action == 'stop': dialog._stop_training()
            elif action == 'change': dialog.samples_edit.setText(str(self.root/'changed.tsv'))
            else: getattr(dialog,action)()
            self.assertIsNotNone(dialog._worker)
            release.set(); self.wait(dialog)
            self.assertFalse(target.exists())
            self.assertIsNone(dialog.saved_model_source)
            self.assertFalse(any((self.root/name).exists() for name in plots.FILENAMES))

    def test_tsv_commit_and_reload_failure_prevent_all_svg_publication(self):
        class FailedCommit:
            def __init__(self,path): self.target = QSaveFile(path)
            def __getattr__(self,name): return getattr(self.target,name)
            def commit(self): return False
        target = self.root/'model.tsv'; target.write_bytes(b'previous model')
        for failure in ['commit','reload']:
            dialog = self.dialog(); dialog.save_plots_checkbox.setChecked(True)
            with (patch.object(ui.QMessageBox,'warning'),
                    patch.object(ui,'QSaveFile',FailedCommit if failure=='commit' else QSaveFile),
                    patch.object(ui,'model_source_for_trained_model',side_effect=ValueError('reload failed')) if
                        failure=='reload' else patch.object(ui,'model_source_for_trained_model',wraps=ui.model_source_for_trained_model)):
                self.save(dialog,target); self.wait(dialog)
            self.assertIsNone(dialog.saved_model_source)
            self.assertFalse(any((self.root/name).exists() for name in plots.FILENAMES))
            if failure=='commit': self.assertEqual(target.read_bytes(),b'previous model')

    def test_individual_svg_commit_failure_preserves_old_file_and_reports_verified_model(self):
        previous = self.root/'saturation_pdf.svg'; previous.write_bytes(b'previous plot')
        class FailedOneCommit:
            def __init__(self,path):
                self.target = QSaveFile(path)
                self.fail = Path(path).name == 'saturation_pdf.svg'
            def __getattr__(self,name): return getattr(self.target,name)
            def commit(self): return False if self.fail else self.target.commit()
        dialog = self.dialog(); dialog.save_plots_checkbox.setChecked(True)
        with patch.object(ui,'QSaveFile',FailedOneCommit),patch.object(ui.QMessageBox,'warning') as warning:
            self.save(dialog,self.root/'partial.tsv'); self.wait(dialog)
        self.assertEqual(dialog.result(),QDialog.DialogCode.Accepted)
        self.assertEqual(factory.load_selected_model(dialog.saved_model_source),self.completion.model)
        self.assertEqual(previous.read_bytes(),b'previous plot')
        self.assertEqual(tuple(Path(p).name for p in dialog.saved_plot_paths),('hue_pdf.svg','value_pdf.svg'))
        self.assertEqual(dialog.plot_save_errors[0][0],'saturation_pdf.svg')
        self.assertIn('verified TSV model is selected',warning.call_args.args[2])

    def test_reserved_output_and_file_dialog_reentrancy_cannot_replace_model_with_svg(self):
        dialog = self.dialog(); dialog.save_plots_checkbox.setChecked(True)
        for name in plots.FILENAMES:
            with patch.object(ui.QMessageBox,'warning'):
                self.save(dialog,self.root/name)
            self.assertIsNone(dialog._worker)
            self.assertFalse((self.root/name).exists())
        target = self.root/'stale.tsv'
        def choose(*args):
            dialog.samples_edit.setText(str(self.root/'changed.tsv'))
            return str(target),'tsv'
        with patch.object(ui.QFileDialog,'getSaveFileName',side_effect=choose):
            dialog._save_model()
        self.assertFalse(target.exists())
        self.assertIsNone(dialog._worker)

    def test_parent_close_during_svg_failure_report_finishes_publication_then_rejects(self):
        parent = parent_ui.NativeNaiveBayesDialog(factory.build_native_naive_bayes_recipe())
        self.dialogs.append(parent)
        child = self.dialog()
        child.setParent(parent)
        child.save_plots_checkbox.setChecked(True)
        original_recipe = parent.recipe
        target = self.root/'closing.tsv'
        def report(*args):
            self.assertTrue(child._publishing)
            parent.reject()
            self.assertTrue(child._close_requested)
            self.assertTrue(child._publishing)
        with (patch.object(parent_ui,'NativeNaiveBayesTrainingDialog',return_value=child),
                patch.object(ui.QFileDialog,'getSaveFileName',return_value=(str(target),'tsv')),
                patch.object(ui.NativeNaiveBayesTrainingDialog,'_write_plot',side_effect=OSError('disk failure')),
                patch.object(ui.QMessageBox,'warning',side_effect=report)):
            QTimer.singleShot(0,child._save_model)
            parent._train_model()
        self.assertFalse(child._publishing)
        self.assertEqual(child.result(),QDialog.DialogCode.Rejected)
        self.assertIsNone(parent._training_dialog)
        self.assertIsNone(child._completion)
        self.assertEqual(parent.recipe,original_recipe)
        self.assertEqual(target.read_bytes(),self.completion.tsv_bytes)


if __name__ == '__main__':
    unittest.main()
