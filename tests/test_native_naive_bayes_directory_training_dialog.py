"""Directory training, strict portable provenance, and actual Qt/project saves."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import threading
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtCore import QSaveFile, QTimer
from PySide6.QtWidgets import QApplication, QDialog

from seedvision.benchmarking import naive_bayes_training_directory as directory
from seedvision.benchmarking import native_naive_bayes_recipe as factory
from seedvision.benchmarking import native_recipe as runner
from seedvision.export import native_results as outputs
from seedvision.export.project_native_graphs import export_project_native_graphs
from seedvision.model_io import naive_bayes_provenance as provenance
from seedvision.persistence import reference_models as assets
from seedvision.persistence.project_analysis import ProjectAnalysisStore, InvalidProjectAnalysis
from seedvision.segmentation.baseline import PipelineAnalysisCache
from seedvision.ui import native_naive_bayes_training_dialog as ui
from seedvision.ui import native_naive_bayes_dialog as picker
from tests import test_native_naive_bayes_training_dialog as sample_workflow
from tests.test_plantcv_naive_bayes import _image
from tests.test_plantcv_naive_bayes_sampling import coordinate_pair, pair

bind_recipe = sample_workflow.bind_recipe


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')


def selection_digest(value):
    selection = value['selection']
    selection['selection_sha256'] = sha256(canonical({key:item for key,item in selection.items()
        if key != 'selection_sha256'})).hexdigest()


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class NativeBayesDirectoryTrainingWorkflowTests(unittest.TestCase):
    wait = sample_workflow.NativeBayesTrainingWorkflowTests.wait
    tearDown = sample_workflow.NativeBayesTrainingWorkflowTests.tearDown
    make_project = sample_workflow.NativeBayesTrainingWorkflowTests.make_project

    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        import tempfile
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images, self.masks = self.root/'images-é-种子', self.root/'masks-α'
        self.images.mkdir()
        self.masks.mkdir()
        for index, name in enumerate(('z/same.png', 'a/same.png')):
            image, mask = pair(seed=101+index)
            path = self.images/name
            path.parent.mkdir()
            self.write_image(path, image)
            self.write_image(self.masks/'same.png', mask)
        self.completion = directory.train_bayes_directories(self.images, self.masks,
            rng=np.random.RandomState(17))
        self.model_path = self.root/'trained-directory.tsv'
        self.model_path.write_bytes(self.completion.tsv_bytes)
        self.selected = factory.model_source_for_trained_model(self.model_path,
            self.completion.tsv_bytes, self.completion.provenance_json)
        self.recipe = bind_recipe(self.selected)
        self.image_path = self.root/'seeds.png'
        self.write_image(self.image_path, _image())
        self.dialogs = []

    def write_image(self, path, image):
        ok, raw = cv2.imencode('.png', image)
        self.assertTrue(ok)
        path.write_bytes(raw.tobytes())

    def dialog(self, **kwargs):
        result = ui.NativeNaiveBayesTrainingDialog(**kwargs)
        self.dialogs.append(result)
        result.input_mode.setCurrentIndex(1)
        result.image_directory_edit.setText(str(self.images))
        result.mask_directory_edit.setText(str(self.masks))
        return result

    def prepared_dialog(self):
        result = self.dialog()
        result._run_mode = 'directories'
        result._run_path = (str(self.images.resolve()), str(self.masks.resolve()))
        result._completion = self.completion
        result.save_button.setEnabled(True)
        return result





















if __name__ == '__main__':
    unittest.main()
