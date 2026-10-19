"""Actual native product persistence and saved-project batch bindings."""

from tests.catalogue_fixtures import CatalogueTestCase
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time
import unittest
import weakref
from unittest.mock import patch

import cv2
import numpy as np
import torch

from seedvision.benchmarking import native_kmeans_recipe as kmeans
from seedvision.benchmarking.native_recipe import (
    NativeRecipeCancelled, build_native_seed_recipe, run_native_recipe_path,
)
from seedvision.benchmarking.checkpoints import CheckpointLimits
from seedvision.benchmarking.recipes import initial_stage_run
from seedvision.export import native_results as exports
from seedvision.export.project_native_graphs import export_project_native_graphs
from seedvision.model_io import plantcv_kmeans_model as models
from seedvision.persistence.reference_graphs import ReferenceGraphAttachment
from seedvision.segmentation.baseline import PipelineAnalysisCache
from tests.test_native_kmeans_recipe import _parameter
from tests.test_native_recipe import fixture, selected_recipe


def _attachment(recipe):
    return ReferenceGraphAttachment(recipe, tuple(initial_stage_run(recipe, stage.stage_id)
                                                for stage in recipe.stages))


def _output(report, stage, port):
    return dict(next(node for node in report['nodes'] if node['stage_id'] == stage)['outputs'])[port]


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class NativeResultExportTests(CatalogueTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root/'seeds.png'
        image = np.zeros((13, 17, 3), np.uint8)
        image[:, 8:] = 180
        self.assertTrue(cv2.imwrite(str(self.source), image))
        self.model_path = self.root/'centres.json'
        self.recipe = self.make_recipe(2)

    def tearDown(self):
        self.temp.cleanup()

    def make_recipe(self, count):
        self.model_path.write_text(json.dumps({'schema': models.SCHEMA,
            'preprocessing': models.PREPROCESSING, 'dtype': 'float64',
            'patch_size': 1, 'channels': 1,
            'centres': [[float(value)] for value in (range(256) if count == 256 else (0, 180))]}),
            encoding='utf-8')
        return _parameter(kmeans.build_native_kmeans_recipe(), kmeans.MODEL_STAGE_ID,
                          'model_source', kmeans.model_source_for_path(self.model_path))

    def run_recipe(self, recipe=None):
        return run_native_recipe_path(recipe or self.recipe, self.source)














    def test_project_graph_cancel_or_changed_source_never_publishes(self):
        attachments = (_attachment(self.recipe),)
        with self.assertRaises(ValueError):
            export_project_native_graphs(attachments, self.source, self.root/'wrong-source',
                                          expected_source_sha256='0'*64)
        with self.assertRaises(NativeRecipeCancelled):
            export_project_native_graphs(attachments, self.source, self.root/'cancelled-graphs',
                expected_source_sha256=sha256(self.source.read_bytes()).hexdigest(), cancel=lambda: True)
        self.assertFalse((self.root/'cancelled-graphs').exists())
        self.assertFalse(list(self.root.glob('.native-graphs-*')))





class NativeMetadataTests(CatalogueTestCase):
    def test_ordered_mappings_tuples_and_nonfinite_values_are_unambiguous(self):
        from seedvision.benchmarking.checkpoints import CoordinateFrame
        collector = exports._Collector(CoordinateFrame('metadata', (2, 3), 'top-left', 'xy', 'pixel'),
                                       CheckpointLimits(), None)
        encoded = collector.encode({'kind': ('first', [float('nan'), float('inf'), -float('inf')]),
                                    'second': {'value': 2., 'available': False}})
        self.assertEqual([name for name, _ in encoded['items']], ['kind', 'second'])
        values = encoded['items'][0][1]['items'][1]['items']
        self.assertEqual([value['value'] for value in values], ['nan', 'positive_infinity', 'negative_infinity'])
        self.assertEqual(json.loads(exports._json_bytes(encoded)), encoded)
        self.assertEqual(collector.arrays, {})

    def test_metadata_depth_cycles_cpu_tensors_and_visit_caps_reject(self):
        from seedvision.benchmarking.checkpoints import CoordinateFrame
        def collector():
            return exports._Collector(CoordinateFrame('metadata', (2, 3), 'top-left', 'xy', 'pixel'),
                                      CheckpointLimits(), None)
        cycle = []
        cycle.append(cycle)
        for value in (cycle, torch.zeros((2, 3)), np.zeros((2, 3), np.uint8), 2**257):
            with self.assertRaises(exports.NativeResultExportError):
                collector().encode(value)
        with patch.object(exports, 'MAX_METADATA_ITEMS', 2):
            with self.assertRaises(exports.NativeResultExportError):
                collector().encode([1, 2, 3])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_ambiguous_layout_and_unversioned_tensors_are_explicit(self):
        from seedvision.benchmarking.checkpoints import CoordinateFrame
        collector = exports._Collector(CoordinateFrame('square', (3, 3), 'top-left', 'xy', 'pixel'),
                                       CheckpointLimits(), None)
        binding = collector.array(torch.zeros((3, 3, 3), dtype=torch.uint8, device='cuda'))
        self.assertNotIn(binding['name'], collector.roles)
        with torch.inference_mode():
            value = torch.zeros((3, 3), dtype=torch.uint8, device='cuda')
        with self.assertRaises(exports.NativeResultExportError):
            collector.array(value)



@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class NativeDesktopExportTests(unittest.TestCase):
    setUp = NativeResultExportTests.setUp
    tearDown = NativeResultExportTests.tearDown
    make_recipe = NativeResultExportTests.make_recipe
    run_recipe = NativeResultExportTests.run_recipe
