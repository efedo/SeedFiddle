"""Real directory training, CUDA row caching and asynchronous desktop lifecycle."""

from tests.catalogue_fixtures import CatalogueTestCase

from tests.functional_units import functional_units
from dataclasses import replace
import os
from pathlib import Path
import random
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication, QDialog

from seedvision.benchmarking import native_kmeans_training as native
from seedvision.benchmarking import native_kmeans_recipe as factory
from seedvision.benchmarking import native_recipe as runner
from seedvision.benchmarking import kmeans_model_artifact as artifacts
from seedvision.cuda import plantcv_kmeans_model as models
from seedvision.ui import native_kmeans_training_dialog as ui
from seedvision.ui import native_kmeans_dialog as picker


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
@functional_units
class NativeKMeansTrainingTests(CatalogueTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.folder = self.root/'training'
        self.folder.mkdir()
        for name in ('a.png', 'b.png'):
            self.assertTrue(cv2.imwrite(str(self.folder/name), np.zeros((17, 18, 3), np.uint8)))
        self.request = native.KMeansTrainingRequest(str(self.folder), patch_size=2, sampling=5, n_init=1)




    def check_test_bad_parameters_and_retained_rows_obey_effective_bounds(self):
        for changes in (dict(patch_size=True), dict(sampling=True), dict(sampling=0),
                dict(sampling=float('nan')), dict(sampling=1.), dict(sampling=1.1), dict(seed=-1), dict(n_init=0)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                native.train_kmeans_directory(replace(self.request, **changes))
        first = native.train_kmeans_directory(self.request)
        images = native._directory.materialize_kmeans_training_directory(first.cache.manifest)
        changed = replace(self.request, patch_size=3)
        plan = native._training.preflight_kmeans_image_fit(images.images, changed.clusters, source_n_threads=None,
            patch_size=changed.patch_size, sampling=changed.sampling, n_init=changed.n_init)
        identity = native._row_calculation_identity()
        with patch.object(native, '_row_calculation_identity', return_value=identity), \
                patch.object(native._training, 'MAX_CUDA_LIVE_BYTES', plan.peak_live_bytes):
            with self.assertRaisesRegex(ValueError, 'retained row cache'):
                native.train_kmeans_directory(changed, cached_rows=first.cache)

    def check_test_calculation_identity_rejects_changed_loaded_producer_contract(self):
        identity = native._row_calculation_identity()
        self.assertEqual(len(identity), 64)
        with patch.object(native._training, 'PATCH_SEED', 7):
            with self.assertRaises(ValueError):
                native._row_calculation_identity()

    def check_test_provenance_projection_never_inspects_or_transfers_dense_fit_products(self):
        completed = native.train_kmeans_directory(self.request)
        provenance = artifacts.parse_kmeans_artifact(completed.artifact_payload).provenance
        class DenseSentinel:
            def __getattribute__(self, name):
                raise AssertionError('provenance traversed dense CUDA data')
        scalar = provenance['fit_result']
        fitted = SimpleNamespace(centres=DenseSentinel(), counts=DenseSentinel(),
            labels=DenseSentinel(), row_distances=DenseSentinel(),
            plan=native._fit.KMeansFitPlan(**provenance['fit_plan']),
            inertia=native._fit._intervals.F64Interval(**scalar['inertia']),
            scaled_tolerance=native._fit._intervals.F64Interval(**scalar['scaled_tolerance']),
            n_steps=scalar['n_steps'], n_iter=scalar['n_iter'], source_n_threads=None,
            source_reduction_profile=scalar['source_reduction_profile'],
            random_identity=scalar['random_identity'])
        result = native._training_provenance(self.request, completed.cache.manifest,
            completed.cache.key[-1], fitted, completed.export.model, completed.cache.rows.features.device)
        self.assertEqual(result['fit_result'], scalar)
        self.assertEqual(artifacts.parse_kmeans_artifact(
            artifacts.encode_kmeans_artifact(completed.export.model, result)).provenance, provenance)

    def test_workflow_lifecycle(self):
        """2 related NativeKMeansTrainingTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_bad_parameters_and_retained_rows_obey_effective_bounds)
        self.scenario(self.check_test_provenance_projection_never_inspects_or_transfers_dense_fit_products)

    def test_validation_and_recovery(self):
        """1 related NativeKMeansTrainingTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_calculation_identity_rejects_changed_loaded_producer_contract)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class NativeKMeansTrainingDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.folder = self.root/'training'
        self.folder.mkdir()
        self.assertTrue(cv2.imwrite(str(self.folder/'a.png'), np.zeros((17, 18, 3), np.uint8)))
        self.completion = native.train_kmeans_directory(native.KMeansTrainingRequest(str(self.folder),
            patch_size=2, sampling=5, clusters=1, n_init=1))
        self.dialog = ui.NativeKMeansTrainingDialog()
        self.dialog.directory_edit.setText(str(self.folder))

    def tearDown(self):
        self.dialog.close()
        self.wait()
        self.application.processEvents()
        self.temp.cleanup()

    def wait(self):
        deadline = time.monotonic()+15
        while self.dialog._worker is not None and time.monotonic() < deadline:
            self.application.processEvents()
            time.sleep(.005)
        self.assertIsNone(self.dialog._worker, 'training worker did not drain')

    def configure(self):
        self.dialog.patch_size.setValue(2)
        self.dialog.sampling_mode.setCurrentIndex(1)
        self.dialog.sampling_count.setValue(5)
        self.dialog.clusters.setValue(1)
        self.dialog.n_init.setValue(1)
