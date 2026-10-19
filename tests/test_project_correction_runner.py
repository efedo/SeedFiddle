"""Pinned correction execution, origin archives and actual batch companions."""

from tests.catalogue_fixtures import CatalogueTestCase
from copy import deepcopy
from dataclasses import replace
import gc
from hashlib import sha256
import inspect
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
import weakref

import cv2
import numpy as np
import torch

from seedvision.benchmarking import project_correction as adapter
from seedvision.benchmarking.native_correction_inputs import run_native_correction_files
from seedvision.benchmarking.native_correction_recipe import STAGE_ID, RECIPE_ID
from seedvision.benchmarking.native_recipe import NativeRecipeCancelled
from seedvision.export import native_results as exports
from seedvision.export import project_native_graphs as graphs
from seedvision.persistence import correction_inputs as assets
from seedvision.persistence.project_analysis import ProjectImageSpec, ProjectAnalysisStore
from seedvision.segmentation.baseline import PipelineAnalysisCache
from tests import test_project_correction_inputs as fixtures
from tests.test_native_result_export import _attachment, _output


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class ProjectCorrectionRunnerTests(CatalogueTestCase):
    setUp = fixtures.ProjectCorrectionInputTests.setUp
    tearDown = fixtures.ProjectCorrectionInputTests.tearDown
    make_binding = fixtures.ProjectCorrectionInputTests.make_binding
    project = fixtures.ProjectCorrectionInputTests.project

    def run_binding(self, document, **kwargs):
        return adapter.run_project_correction(self.recipe, self.workspace, document.correction_bindings[0],
                                             document.images[0], self.source, **kwargs)

    def test_actual_transport_run_preserves_origins_and_verified_producer_identity(self):
        _, document = self.project()
        result, prepared = self.run_binding(document)
        binding = document.correction_bindings[0]
        self.assertEqual(result.input_origins.binding, binding)
        self.assertEqual(result.input_origins.project_source_path, document.images[0].source.path)
        self.assertEqual(result.source_sha256, document.images[0].source.sha256)
        self.assertEqual(result.coordinate_frame.shape, document.images[0].source_shape)
        for asset in binding.inputs:
            actual = result.input_sources[asset.role]
            self.assertEqual(actual['path'], str((self.workspace/asset.path).resolve()))
            self.assertNotEqual(actual['path'], asset.origin_path)
            self.assertEqual(actual['sha256'], asset.source_sha256)
            self.assertEqual(tuple(actual['frame']['shape']), asset.frame.shape)
            self.assertIn('path='+actual['path'], result.preparation[asset.role])
            self.assertEqual(prepared.tensors[asset.role].device.type, 'cuda')
        self.assertEqual({name: result.runtime[name] for name in assets.PROJECT_RUNTIME_KEYS}, adapter._implementation())
        self.assertEqual(result.input_origins, assets.CorrectionRunOrigins.from_dict(result.input_origins.to_dict()))









    def test_legacy_schema3_local_correction_identity_and_arrays_remain_readable(self):
        from seedvision.benchmarking.checkpoints import load_checkpoint, publish_checkpoint
        local, _ = run_native_correction_files(self.recipe, self.paths)
        path = exports.export_native_recipe_result(self.root/'v4', self.recipe, local)
        report = json.loads(path.read_bytes())
        self.assertIsNone(report.pop('correction_input_origins'))
        report.pop('result_revision')
        groups = report.pop('groups')
        report['version'] = 3
        for node in report['nodes']:
            # Calculation metadata belongs to schema5, not the legacy wire format.
            node.pop('calculation')
        self.assertNotIn('correction_input_origins', report['checkpoint_identity']['image_semantics'])
        binding = sha256(exports._json_bytes(report)).hexdigest()
        destination = self.root/'legacy3'
        destination.mkdir()
        saved = []
        for index, group in enumerate(groups):
            checkpoint = load_checkpoint(path.parent/'checkpoints', group['checkpoint_sha256'])
            reference = publish_checkpoint(destination/'checkpoints', checkpoint.identity, checkpoint.arrays,
                execution=checkpoint.execution, measurement=checkpoint.measurement, raster_roles=checkpoint.raster_roles,
                metadata={'format': exports.FORMAT, 'version': 3, 'run_binding_sha256': binding,
                          'group_index': index, 'arrays': group['arrays']}, allow_cpu_transfer=True)
            saved.append({'checkpoint_sha256': reference.checkpoint_sha256, 'arrays': group['arrays']})
        report['groups'] = saved
        report['result_revision'] = sha256(exports._json_bytes(report)).hexdigest()
        (destination/'native-results.json').write_bytes(exports._json_bytes(report))
        restored, arrays = exports.load_native_recipe_result(destination)
        self.assertEqual(restored['version'], 3)
        self.assertNotIn('correction_input_origins', restored)
        self.assertEqual(restored['checkpoint_identity'], exports._identity(self.recipe, local).to_dict())
        np.testing.assert_array_equal(arrays[_output(restored, STAGE_ID, 'corrected_image')['name']],
                                      local.node(STAGE_ID).outputs['corrected_image'].cpu())


    def duplicate_project(self):
        other = self.workspace/'same-rgb.png'
        shutil.copyfile(self.source, other)
        images = self.store.capture(analysis_settings=self.profile,
            images=(ProjectImageSpec(self.source, (17, 25)), ProjectImageSpec(other, (17, 25))),
            native_recipes=(self.recipe,)).images
        paths = dict(self.paths)
        paths['source_rgb'] = other
        mask = cv2.imread(str(paths['source_mask']), cv2.IMREAD_UNCHANGED)
        mask[0, 0] = 10
        paths['source_mask'] = self.root/'distinct-mask.png'
        self.assertTrue(cv2.imwrite(str(paths['source_mask']), mask))
        second = self.make_binding(images[1].identifier, paths)
        master, document = self.project((self.binding, second), images)
        return master, document, (self.source, other)
