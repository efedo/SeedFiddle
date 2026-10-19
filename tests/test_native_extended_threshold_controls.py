"""Pinned control witnesses and desktop routing for the four working thresholds."""

from tests.catalogue_fixtures import CatalogueTestCase

from tests.functional_units import functional_units
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from seedvision.benchmarking.native_extended_threshold_recipes import build_native_extended_threshold_recipes
from seedvision.benchmarking.recipe_catalogue import build_recipe_catalogue
from seedvision.benchmarking.recipes import RecipeSpec
from seedvision.benchmarking.native_recipe import run_native_recipe
from seedvision.benchmarking.checkpoints import CoordinateFrame
from tests.test_native_extended_threshold_recipe import _fixture, _reference_mask, _with_parameters, SOURCE, PREPARATION


def _fixtures():
    rng = np.random.default_rng(20261008)
    noise = rng.integers(0, 256, (31, 37), dtype=np.uint8)
    y, x = np.indices((31, 37))
    checker = (((x//3+y//4) % 2)*220+17).astype(np.uint8)
    edge = np.where(x < 18, 38, 206).astype(np.uint8)
    return noise, checker, edge


def _native_mask(operation, image, case):
    recipe = _with_parameters(build_native_extended_threshold_recipes()[
        'plantcv_threshold_'+operation], **case)
    is_rgb = operation == 'dual_channels'
    gray = np.zeros(image.shape[:2], dtype=np.uint8) if is_rgb else image
    rgb = np.ascontiguousarray(image[..., ::-1]) if is_rgb else np.repeat(image[..., None], 3, axis=-1)
    result = run_native_recipe(recipe,
        torch.as_tensor(np.ascontiguousarray(gray), device='cuda'),
        torch.as_tensor(np.ascontiguousarray(rgb), device='cuda'),
        source_sha256=SOURCE, coordinate_frame=CoordinateFrame(
            'control.witness', gray.shape, 'top-left', 'x-right,y-down', 'pixel'),
        preparation=PREPARATION)
    node = result.node(recipe.stages[1].stage_id)
    if node.status != 'executed':
        raise AssertionError(node.reason)
    return node.outputs['mask'].cpu().numpy()


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for threshold control witnesses')
@functional_units
class ExtendedThresholdControlTests(CatalogueTestCase):
    def witness(self, operation, base, variant, fixtures):
        for image in fixtures:
            native = [_native_mask(operation, image, case) for case in (base, variant)]
            if np.array_equal(native[0], native[1]):
                continue
            for actual, case in zip(native, (base, variant)):
                expected = _reference_mask(image, operation, case)
                np.testing.assert_array_equal(actual, expected,
                    err_msg=f'{operation} control {base!r} -> {variant!r}')
            return
        self.fail(f'No fixture makes {operation} control live: {base!r} -> {variant!r}')

    def check_test_every_gaussian_control_changes_a_pinned_graph_mask(self):
        base = {'ksize': 3, 'offset': 0, 'object_type': 'light'}
        for parameter, value in (('ksize', 9), ('offset', 24), ('object_type', 'dark')):
            with self.subTest(parameter=parameter):
                self.witness('gaussian', base, {**base, parameter:value}, _fixtures())

    def check_test_every_dual_channel_control_changes_a_pinned_graph_mask(self):
        rng = np.random.default_rng(502)
        image = rng.integers(0, 256, (31, 37, 3), dtype=np.uint8)
        base = {'x_channel':'B', 'y_channel':'G',
            'points':[[25,225],[225,25]], 'above':True}
        for parameter, value in (('x_channel','R'), ('y_channel','R'),
                ('points',[[45,190],[210,70]]), ('above',False)):
            with self.subTest(parameter=parameter):
                self.witness('dual_channels', base, {**base,parameter:value}, (image,))

    def check_test_every_texture_control_changes_a_pinned_graph_mask(self):
        # Two-level fixtures keep declared precision withholding separate from
        # whether an exposed parameter reaches the actual graph calculation.
        base = {'ksize':3, 'offset':1, 'threshold':90,
            'texture_method':'dissimilarity', 'borders':'nearest'}
        for parameter, value in (('ksize',7), ('offset',2), ('threshold',190),
                ('texture_method','homogeneity'), ('borders','wrap')):
            with self.subTest(parameter=parameter):
                selected = {**base, 'threshold':40} if parameter == 'borders' else base
                self.witness('texture', selected, {**selected,parameter:value}, _fixtures()[1:])

    def check_test_triangle_polarity_changes_a_pinned_graph_mask(self):
        self.witness('triangle', {'object_type':'light'}, {'object_type':'dark'}, (_fixture()[2],))

    def test_additional_numerical_regressions(self):
        self.scenario(self.check_test_every_gaussian_control_changes_a_pinned_graph_mask)
        self.scenario(self.check_test_every_dual_channel_control_changes_a_pinned_graph_mask)
        self.scenario(self.check_test_every_texture_control_changes_a_pinned_graph_mask)
        self.scenario(self.check_test_triangle_polarity_changes_a_pinned_graph_mask)






class ExtendedThresholdDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        from PySide6.QtWidgets import QApplication
        cls.application = QApplication.instance() or QApplication([])

    def wait_until(self, predicate, timeout=30):
        deadline = time.monotonic()+timeout
        while not predicate() and time.monotonic() < deadline:
            self.application.processEvents()
            time.sleep(.01)
        self.application.processEvents()
        self.assertTrue(predicate(), 'The native threshold worker did not drain.')
