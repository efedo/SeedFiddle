"""Portable trait evidence and actual relocated CUDA/desktop/batch consumers."""
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

    def test_enabled_and_disabled_selections_copy_exact_bytes_and_relocate(self):
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                master, snapshot = self.snapshot(enabled=enabled, selection='selected.json')
                before = master.read_bytes()
                index = self.index(snapshot)
                record = assets.TraitModelAsset(**index['trait_model_assets'][0])
                self.assertEqual((snapshot / record.path).read_bytes(), self.raw)
                self.assertEqual(record.source_bytes, len(self.raw))
                self.assertEqual(record.source_sha256, sha256(self.raw).hexdigest())
                self.assertEqual(record.model_sha256, self.model.model_sha256)
                self.assertEqual(self.selected(snapshot, snapshot / index['master']), record.path)
                restored_root = self.base / ('restored-' + str(enabled))
                working = restore_snapshot(snapshot, restored_root)
                selected = self.selected(restored_root, working)
                self.assertTrue((restored_root / selected).is_file())
                self.assertEqual((restored_root / selected).read_bytes(), self.raw)
                graph = build_default_pipeline()
                loaded = ProjectAnalysisStore(restored_root).load(working)
                apply_analysis_settings_profile(graph, loaded.document.analysis_settings)
                self.assertEqual(graph.node('trait_classifier').enabled, enabled)
                self.assertEqual(master.read_bytes(), before)

    def test_none_and_legacy_without_selected_model_preserve_absence(self):
        master, snapshot = self.snapshot(selection='None')
        self.assertNotIn('trait_model_assets', self.index(snapshot))
        working = restore_snapshot(snapshot, self.base / 'restored')
        self.assertIsNone(self.selected(self.base / 'restored', working))
        self.assertIsNone(self.selected(self.root, master))

    def test_missing_invalid_or_oversized_selection_cannot_publish_snapshot(self):
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

    def test_exact_byte_limit_and_growth_reads_are_bounded(self):
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

    def test_changed_source_or_failed_copy_never_publishes_completion(self):
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

    def test_incomplete_duplicate_or_inconsistent_inventory_rejects_before_restore(self):
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

    def test_corrupt_object_and_restore_copy_failure_do_not_publish_working_project(self):
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

    def test_actual_relocated_batch_and_native_worker_preserve_predictions_and_reviewed_traits(self):
        from seedvision.calibration import image as calibration
        from seedvision.export.project_runner import run_project
        from seedvision.export.results import result_report
        from seedvision.segmentation.baseline import analyze_path, PipelineAnalysisCache
        from seedvision.segmentation.procedural import ProceduralInstanceResult
        from tests.test_neutral_calibration import fixture, ruler
        from PySide6.QtWidgets import QApplication
        from seedvision.ui.main_window import _AnalysisTask
        app = QApplication.instance() or QApplication([])
        image, card = fixture()
        self.image = image
        self.assertTrue(cv2.imwrite(str(self.source), image))
        labels = np.zeros(image.shape[:2], np.uint16)
        labels[28:32, 30:34] = 1
        labels[28:32, 60:64] = 2
        annotations = (
            SeedInstanceAnnotation(1, coat_pattern='white', conditions_reviewed=True,
                shape_reviewed=True, outline_visibility='complete', pose='flat', hilum_point=(33., 29.),
                hilum_direction=(1., 0.)),
            SeedInstanceAnnotation(2),
        )
        ReferenceRegionStore(self.root).save(self.source, ReferenceRegionBundle(
            image.shape[:2], foreground=labels > 0, annotated_seeds=labels,
            seed_annotations=annotations, source_to_corrected=np.eye(3)))
        master = self.project()
        captured = []
        def decode(valid, *args, **kwargs):
            h, w = valid.shape[-2:]
            output = np.zeros((h, w), np.int32)
            output[h//2-2:h//2+2, w//3-2:w//3+2] = 1
            output[h//2-2:h//2+2, 2*w//3-2:2*w//3+2] = 2
            zeros = np.zeros((h, w), np.float32)
            return ProceduralInstanceResult(labels=output, centres_xy=np.zeros((2, 2), np.float32),
                marker_scores=np.ones(2, np.float32), instance_confidences=np.ones(2, np.float32),
                occupancy_likelihood=zeros, occupancy_mask=zeros.astype(bool), boundary_cost=zeros,
                centre_likelihood=zeros, source_shape=(h, w), working_scale=1.)
        def native(path, **kwargs):
            captured.append((path, kwargs))
            return analyze_path(path, **kwargs)
        with (patch.object(calibration, 'detect_colour_card', return_value=card),
              patch.object(calibration, 'detect_ruler', side_effect=ruler),
              patch('seedvision.segmentation.baseline.procedural_seed_instances', side_effect=decode),
              patch('seedvision.export.project_runner.analyze_path', side_effect=native)):
            before_path = run_project(self.root, master, self.base / 'before')[0]
            before = json.loads((before_path / 'results.json').read_text(encoding='utf-8'))
            snapshot = create_snapshot(self.root, master, self.base / 'snapshots')
            self.assertTrue(self.root.resolve().is_relative_to(self.base.resolve()))
            shutil.rmtree(self.root)
            restored_root = self.base / 'restored'
            self.configure_root(restored_root)
            working = restore_snapshot(snapshot, restored_root)
            after_path = run_project(restored_root, working, self.base / 'after')[0]
            after = json.loads((after_path / 'results.json').read_text(encoding='utf-8'))
            self.assertEqual(before['trait_classifier']['predictions'], after['trait_classifier']['predictions'])
            for key in ('model_sha256', 'model_artifact_sha256', 'frozen_model', 'abstention_threshold'):
                self.assertEqual(before['trait_classifier'][key], after['trait_classifier'][key])
            path, kwargs = captured[-1]
            required = {name: kwargs.get(name) for name, p in inspect.signature(_AnalysisTask).parameters.items()
                        if p.default is inspect.Parameter.empty}
            required.update(path=path, background_reference_points=(), foreground_reference_points=(),
                physical_edge_reference_mask=None, non_edge_reference_mask=None, pipeline_revision=1,
                node_cache=PipelineAnalysisCache(), dirty_nodes=frozenset(), trait_settings=kwargs['trait_settings'],
                reference_transform=kwargs['reference_transform'],
                reference_archive_source_sha256=kwargs['reference_archive_source_sha256'])
            task = _AnalysisTask(**required)
            completed, failed = [], []
            task.signals.completed.connect(lambda result, revision: completed.append(result))
            task.signals.failed.connect(lambda *args: failed.append(args))
            task.run()
            self.assertEqual(failed, [])
            self.assertEqual(len(completed), 1)
            self.assertEqual(result_report(completed[0])['trait_classifier']['predictions'],
                after['trait_classifier']['predictions'])
            confidence = [row['confidence'] for row in after['trait_classifier']['predictions']]
            self.assertNotEqual(confidence[0], confidence[1])
            threshold = sum(confidence) / 2
            loaded = ProjectAnalysisStore(restored_root).load(working)
            graph = build_default_pipeline()
            apply_analysis_settings_profile(graph, loaded.document.analysis_settings)
            graph.set_parameter('trait_classifier', 'trait_abstention_threshold', threshold)
            strict_master = ProjectAnalysisStore(restored_root).save(replace(loaded.document,
                analysis_settings=analysis_settings_profile_from_graph(graph)),
                restored_root / 'threshold.seedfiddle-project.json')
            strict_path = run_project(restored_root, strict_master, self.base / 'strict')[0]
            strict = json.loads((strict_path / 'results.json').read_text(encoding='utf-8'))
            strict_rows = strict['trait_classifier']['predictions']
            self.assertEqual([row['abstained'] for row in strict_rows], [c < threshold for c in confidence])
            self.assertEqual(sum(row['abstained'] for row in strict_rows), 1)
            self.assertEqual([row['probabilities'] for row in strict_rows],
                [row['probabilities'] for row in after['trait_classifier']['predictions']])
        self.assertFalse(self.model_path.exists())
        loaded = ProjectAnalysisStore(restored_root).load(working)
        sidecar = loaded.images[0].sidecar('reference_regions')
        bundle = ReferenceRegionStore(restored_root).load_project_archive(path, sidecar.path, None)
        self.assertEqual(bundle.seed_annotations, annotations)
        for report in (before, after):
            self.assertEqual([row['annotation']['coat_pattern'] for row in report['reviewed_measurements']['observations']], ['white', None])
            self.assertEqual(report['reviewed_measurements']['observations'][0]['annotation']['hilum_point'], [33., 29.])
            self.assertFalse(report['scientifically_validated'])
        # Independent NumPy softmax from actual compact CUDA feature rows.
        result = completed[0]
        binding = result.trait_classifier.report
        features = task.node_cache.values['trait_classifier.features'].values
        payload = self.model.payload
        standardized = (np.asarray(features.tolist()) - payload['standardization']['mean']) / payload['standardization']['scale']
        logits = (standardized @ np.asarray(payload['weights']).T + payload['bias']) / payload['temperature_calibration']['temperature']
        scores = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities = scores / scores.sum(axis=1, keepdims=True)
        identifiers = [row['identifier'] for row in payload['vocabulary']['traits']]
        np.testing.assert_allclose([[row['probabilities'][name] for name in identifiers]
            for row in binding['predictions']], probabilities, rtol=1e-12, atol=1e-14)
        self.assertTrue(features.is_cuda)


if __name__ == '__main__':
    unittest.main()
