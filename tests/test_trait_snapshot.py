"""Portable trait evidence and actual relocated CUDA/desktop/batch consumers."""

from tests.functional_units import functional_units
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import inspect
import io
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from seedvision.persistence import (
    ProjectAnalysisStore, ProjectImageSpec, ReferenceRegionStore, ReferenceRegionBundle,
    analysis_settings_profile_from_graph, apply_analysis_settings_profile,
)
from seedvision.persistence.reference_regions import SeedInstanceAnnotation
from seedvision.persistence.snapshot import create_snapshot, restore_snapshot
from seedvision.persistence import trait_model_assets as assets
from seedvision.pipeline import build_default_pipeline
import tests.test_trait_classifier as core_fixture


@unittest.skipUnless(torch.cuda.is_available(), 'Frozen trait fixture and native inference require CUDA')
@functional_units
class TraitSnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        core_fixture.TraitClassifierTests.setUpClass()
        cls.model = core_fixture.TraitClassifierTests.model

    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root = self.base / 'original'
        self.root.mkdir()
        self.model_path = self.root / 'selected.json'
        self.raw = self.model.to_json().encode('utf-8')
        self.model_path.write_bytes(self.raw)
        self.source = self.root / 'capture.png'
        self.image = np.zeros((24, 24, 3), np.uint8)
        self.image[4:8, 4:8, 2] = 30
        self.image[14:18, 14:18, 2] = 225
        self.assertTrue(cv2.imwrite(str(self.source), self.image))
        self.configure_root(self.root)

    def configure_root(self, root):
        (root / 'config').mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(__file__).parents[1] / 'config' / 'traits.json', root / 'config' / 'traits.json')

    def project(self, *, enabled=True, selection=None):
        graph = build_default_pipeline()
        graph.set_parameter('deskew_colour', 'apply_perspective_correction', False)
        graph.set_parameter('layout_detection', 'expected_outer_diameter_mm', 0.)
        graph.set_parameter('trait_classifier', 'trait_model_path', str(self.model_path) if selection is None else selection)
        graph.set_parameter('trait_classifier', 'trait_abstention_threshold', .61)
        if enabled:
            graph.set_enabled('trait_classifier', True)
        return ProjectAnalysisStore(self.root).capture_and_save(
            analysis_settings=analysis_settings_profile_from_graph(graph),
            images=(ProjectImageSpec(self.source, self.image.shape[:2]),),
            destination=self.root / 'test.seedfiddle-project.json')

    def snapshot(self, **kwargs):
        master = self.project(**kwargs)
        return master, create_snapshot(self.root, master, self.base / 'snapshots')

    def index(self, snapshot):
        return json.loads((snapshot / 'snapshot-index.json').read_text(encoding='utf-8'))

    def selected(self, root, master):
        loaded = ProjectAnalysisStore(root).load(master)
        self.assertFalse(loaded.issues)
        return assets.selected_trait_model(loaded.document.analysis_settings)


    def check_test_none_and_legacy_without_selected_model_preserve_absence(self):
        master, snapshot = self.snapshot(selection='None')
        self.assertNotIn('trait_model_assets', self.index(snapshot))
        working = restore_snapshot(snapshot, self.base / 'restored')
        self.assertIsNone(self.selected(self.base / 'restored', working))
        self.assertIsNone(self.selected(self.root, master))

    def check_test_missing_invalid_or_oversized_selection_cannot_publish_snapshot(self):
        for content in (None, b'{}', b'x' * (assets.MAX_TRAIT_MODEL_BYTES + 1)):
            with self.subTest(kind=None if content is None else len(content)):
                master = self.project()
                if content is None:
                    self.model_path.unlink(missing_ok=True)
                else:
                    self.model_path.write_bytes(content)
                with self.assertRaises((ValueError, OSError)):
                    create_snapshot(self.root, master, self.base / 'failed')
                self.assertFalse((self.base / 'failed').exists())
                self.model_path.write_bytes(self.raw)

    def check_test_exact_byte_limit_and_growth_reads_are_bounded(self):
        padded = self.raw + b' ' * (assets.MAX_TRAIT_MODEL_BYTES - len(self.raw))
        self.model_path.write_bytes(padded)
        raw, digest = assets.trait_model_bytes(self.model_path)
        self.assertEqual(len(raw), assets.MAX_TRAIT_MODEL_BYTES)
        self.assertEqual(digest, self.model.model_sha256)
        for expected in (None, assets.TraitModelAsset('objects/' + sha256(self.raw).hexdigest() + '.trait.json',
                sha256(self.raw).hexdigest(), len(self.raw), self.model.model_sha256)):
            sizes = []
            class Growing(io.BytesIO):
                def read(inner, size=-1):
                    sizes.append(size)
                    return super().read(size)
            with patch.object(Path, 'open', return_value=Growing(padded + b'x')):
                with self.assertRaises(ValueError):
                    assets.trait_model_bytes(self.model_path, expected)
            self.assertEqual(sizes, [(expected.source_bytes if expected else assets.MAX_TRAIT_MODEL_BYTES) + 1])

    def check_test_changed_source_or_failed_copy_never_publishes_completion(self):
        from seedvision.persistence import snapshot as snapshots
        master = self.project()
        original = assets.copy_trait_model
        def replace_source(source, target, asset, raw=None):
            self.model_path.write_bytes(self.raw[:-1] + b' ')
            return original(source, target, asset, raw)
        with patch.object(snapshots, 'copy_trait_model', side_effect=replace_source):
            with self.assertRaises(ValueError):
                create_snapshot(self.root, master, self.base / 'partial')
        folders = list((self.base / 'partial').iterdir())
        self.assertEqual(len(folders), 1)
        self.assertFalse((folders[0] / 'snapshot-index.json').exists())
        copied = next((folders[0] / 'objects').glob('*.trait.json'))
        self.assertEqual(copied.read_bytes(), self.raw)
        self.model_path.write_bytes(self.raw)
        with patch.object(snapshots, 'copy_trait_model', side_effect=OSError('interrupted copy')):
            with self.assertRaises(OSError):
                create_snapshot(self.root, master, self.base / 'partial')
        self.assertTrue(all(not (p / 'snapshot-index.json').exists() for p in (self.base / 'partial').iterdir()))

    def check_test_incomplete_duplicate_or_inconsistent_inventory_rejects_before_restore(self):
        _, snapshot = self.snapshot()
        original = self.index(snapshot)
        for mutate in (
            lambda x: x.pop('trait_model_assets'),
            lambda x: x['trait_model_assets'].append(deepcopy(x['trait_model_assets'][0])),
            lambda x: x['trait_model_assets'][0].update(path='../wrong.trait.json'),
            lambda x: x['trait_model_assets'][0].update(source_bytes=True),
            lambda x: x['trait_model_assets'][0].update(source_sha256='f' * 64),
            lambda x: x['trait_model_assets'][0].update(model_sha256='f' * 64),
            lambda x: x['files'].pop(x['trait_model_assets'][0]['path']),
        ):
            index = deepcopy(original)
            mutate(index)
            (snapshot / 'snapshot-index.json').write_text(json.dumps(index), encoding='utf-8')
            with self.assertRaises(ValueError):
                restore_snapshot(snapshot, self.base / 'rejected')
            self.assertFalse((self.base / 'rejected').exists())
        (snapshot / 'snapshot-index.json').write_text(json.dumps(original), encoding='utf-8')

    def check_test_corrupt_object_and_restore_copy_failure_do_not_publish_working_project(self):
        from seedvision.persistence import snapshot as snapshots
        _, snapshot = self.snapshot()
        index = self.index(snapshot)
        asset_path = snapshot / index['trait_model_assets'][0]['path']
        asset_path.write_bytes(self.raw[:-1] + b' ')
        with self.assertRaises(ValueError):
            restore_snapshot(snapshot, self.base / 'rejected')
        self.assertFalse((self.base / 'rejected').exists())
        asset_path.write_bytes(self.raw)
        with patch.object(snapshots, 'copy_trait_model', side_effect=OSError('interrupted restore')):
            with self.assertRaises(OSError):
                restore_snapshot(snapshot, self.base / 'partial-restore')
        self.assertFalse(list((self.base / 'partial-restore').rglob('working.seedfiddle-project.json')))
        self.assertFalse(list((self.base / 'partial-restore').rglob('snapshot-index.json')))


    def test_workflow_lifecycle(self):
        """1 related TraitSnapshotTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_none_and_legacy_without_selected_model_preserve_absence)

    def test_validation_and_recovery(self):
        """5 related TraitSnapshotTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_missing_invalid_or_oversized_selection_cannot_publish_snapshot)
        self.scenario(self.check_test_exact_byte_limit_and_growth_reads_are_bounded)
        self.scenario(self.check_test_changed_source_or_failed_copy_never_publishes_completion)
        self.scenario(self.check_test_incomplete_duplicate_or_inconsistent_inventory_rejects_before_restore)
        self.scenario(self.check_test_corrupt_object_and_restore_copy_failure_do_not_publish_working_project)


if __name__ == '__main__':
    unittest.main()
