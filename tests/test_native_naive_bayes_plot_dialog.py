"""Actual async compact plots, model-first saves and full modal lifecycle."""

from tests.catalogue_fixtures import CatalogueTestCase
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
class NativeBayesPlotDialogTests(CatalogueTestCase):
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



if __name__ == '__main__':
    unittest.main()
