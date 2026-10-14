"""Real native CUDA analysis, desktop/batch identity and exact review portability."""
from dataclasses import replace
from hashlib import sha256
import inspect
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

import tests.test_trait_classifier as classifier_fixture
import tests.test_neutral_calibration as calibration_fixture
from seedvision.persistence import (ProjectAnalysisStore, ProjectImageSpec,
    ReferenceRegionStore, ReferenceRegionBundle, analysis_settings_profile_from_graph,
    apply_analysis_settings_profile)
from seedvision.persistence.reference_regions import SeedInstanceAnnotation
from seedvision.pipeline import build_default_pipeline
from seedvision.reference_library.contracts import BiologicalContext


@unittest.skipUnless(torch.cuda.is_available(), 'Native parity requires CUDA')
class AuthoritativeParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])
        classifier_fixture.TraitClassifierTests.setUpClass()
        cls.model = classifier_fixture.TraitClassifierTests.model

    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / 'config').mkdir()
        shutil.copyfile(Path(__file__).parents[1] / 'config' / 'traits.json',
                        self.root / 'config' / 'traits.json')
        self.image, self.card = calibration_fixture.fixture()
        self.card = replace(self.card, angle_degrees=7.)
        self.source = self.root / 'capture.png'
        self.assertTrue(cv2.imwrite(str(self.source), self.image))
        self.model_path = self.root / 'frozen.json'
        self.model_path.write_text(self.model.to_json(), encoding='utf-8')
        self.context = BiologicalContext('soybean', seed_lot_id='lot-a')
        self.graph = build_default_pipeline()
        self.graph.set_parameter('deskew_colour', 'apply_perspective_correction', False)
        self.graph.set_parameter('layout_detection', 'expected_outer_diameter_mm', 0.)
        self.graph.set_parameter('layout_detection', 'perimeter_background_buffer_cm', 0.)
        self.graph.set_parameter('layout_detection', 'perimeter_background_band_thickness_cm', .05)
        self.graph.set_parameter('trait_classifier', 'trait_model_path', str(self.model_path))
        self.graph.set_enabled('trait_classifier', True)
        self.labels = np.zeros(self.image.shape[:2], np.uint16)
        self.labels[28:32, 30:34] = 1
        self.labels[28:32, 60:64] = 2
        self.annotations = (SeedInstanceAnnotation(1, coat_pattern='white',
            conditions_reviewed=True, shape_reviewed=True, outline_visibility='complete', pose='flat',
            hilum_point=(33., 29.), hilum_direction=(1., 0.)), SeedInstanceAnnotation(2))
        # The archive declares a translated reference frame; this is not the current transform.
        self.reference_transform = np.array([[1., 0., 2.], [0., 1., 1.], [0., 0., 1.]])
        self.save_project()
        from seedvision.ui.main_window import MainWindow
        self.window = MainWindow(self.root)
        self.addCleanup(self.window.close)
        self.key = str(self.source.resolve()).casefold()
        self.window._current_image_key = lambda: self.key
        self.install_profile()
        from seedvision.calibration import image as calibration
        from seedvision.calibration.geometry import DishCircle
        self.patches = [patch.object(calibration, 'detect_colour_card', return_value=self.card),
            patch.object(calibration, 'detect_ruler', side_effect=calibration_fixture.ruler),
            patch('seedvision.segmentation.baseline.detect_dish', return_value=DishCircle(48, 32, 24, .9)),
            patch('seedvision.segmentation.baseline.procedural_seed_instances', side_effect=self.decode)]
        for mocked in self.patches:
            mocked.start()
            self.addCleanup(mocked.stop)

    def save_project(self, *, layout_x=0., species='Soybean'):
        from seedvision.persistence.project_analysis import ProjectUiState
        ReferenceRegionStore(self.root).save(self.source, ReferenceRegionBundle(
            self.image.shape[:2], foreground=self.labels > 0, annotated_seeds=self.labels,
            seed_annotations=self.annotations, source_to_corrected=self.reference_transform,
            annotation_species='soybean'))
        self.master = ProjectAnalysisStore(self.root).capture_and_save(
            analysis_settings=analysis_settings_profile_from_graph(self.graph),
            images=(ProjectImageSpec(self.source, self.image.shape[:2], biological_context=self.context),),
            species=species, ui_state=ProjectUiState(node_positions={'deskew_colour': (layout_x, 0.)}),
            destination=self.root / 'parity.seedfiddle-project.json')

    def install_profile(self):
        loaded = ProjectAnalysisStore(self.root).load(self.master)
        apply_analysis_settings_profile(self.window.pipeline, loaded.document.analysis_settings)
        self.window._cache_dirty_nodes.clear()
        self.window._current_project_path = self.master
        self.window._project_reference_sidecar_paths[self.key] = loaded.images[0].sidecar('reference_regions').path

    @staticmethod
    def decode(valid, *args, **kwargs):
        from seedvision.segmentation.procedural import ProceduralInstanceResult
        h, w = valid.shape[-2:]
        labels = np.zeros((h, w), np.int32)
        labels[h//2-2:h//2+2, w//3-2:w//3+2] = 1
        labels[h//2-2:h//2+2, 2*w//3-2:2*w//3+2] = 2
        zeros = np.zeros((h, w), np.float32)
        return ProceduralInstanceResult(labels=labels, centres_xy=np.zeros((2, 2), np.float32),
            marker_scores=np.ones(2, np.float32), instance_confidences=np.array([.8, .9], np.float32),
            occupancy_likelihood=zeros, occupancy_mask=zeros.astype(bool), boundary_cost=zeros,
            centre_likelihood=zeros, source_shape=(h, w), working_scale=1.)

    def paired(self):
        from seedvision.export.project_runner import run_project
        from seedvision.segmentation.baseline import analyze_path, PipelineAnalysisCache
        from seedvision.ui.main_window import _AnalysisTask
        from PySide6.QtCore import QThreadPool
        captured = []
        def native(path, **kwargs):
            captured.append((path, kwargs))
            return analyze_path(path, **kwargs)
        with patch('seedvision.export.project_runner.analyze_path', side_effect=native):
            destination = run_project(self.root, self.master, self.root / 'reports')[0]
        batch = json.loads((destination / 'results.json').read_text(encoding='utf-8'))
        path, kwargs = captured[-1]
        parameters = inspect.signature(_AnalysisTask).parameters
        task_kwargs = {name: kwargs[name] for name in parameters if name in kwargs}
        task_kwargs.update(path=path, background_reference_points=(), foreground_reference_points=(),
            physical_edge_reference_mask=None, non_edge_reference_mask=None,
            manual_seed_centres=None,
            pipeline_revision=self.window.pipeline.revision, node_cache=PipelineAnalysisCache(),
            dirty_nodes=frozenset())
        task = _AnalysisTask(**task_kwargs)
        task.setAutoDelete(False)
        completed, failed = [], []
        task.signals.completed.connect(lambda result, revision: completed.append(result))
        task.signals.failed.connect(lambda *args: failed.append(args))
        pool = QThreadPool()
        pool.start(task)
        self.assertTrue(pool.waitForDone(30000), 'Native worker did not finish')
        self.app.processEvents()
        self.assertEqual(failed, [])
        self.assertEqual(len(completed), 1)
        self.result = completed[0]
        self.task = task
        self.window._analyses[self.key] = self.result
        desktop = self.window._current_report(self.result)
        self.assertEqual(desktop['identity'], batch['identity'])
        self.assertEqual(desktop['result_revision'], batch['result_revision'])
        self.assertEqual(desktop['instances'], batch['instances'])
        self.assertEqual(desktop['review_lookup'], batch['review_lookup'])
        self.assertEqual(desktop['execution_provenance']['route'], 'desktop')
        self.assertEqual(batch['execution_provenance']['route'], 'batch')
        return desktop, batch

    def save_desktop_review(self):
        from PySide6.QtWidgets import QDialog, QPushButton, QTableWidget
        def exercise(dialog):
            table = dialog.findChild(QTableWidget)
            save = next(button for button in dialog.findChildren(QPushButton)
                        if button.text().startswith('Save review'))
            table.cellWidget(0, 1).setCurrentText('accepted')
            table.cellWidget(1, 1).setCurrentText('excluded')
            save.click()
            return QDialog.DialogCode.Accepted
        with patch.object(QDialog, 'exec', new=exercise):
            self.window._review_results()

    def test_actual_worker_and_batch_share_crop_calibration_lineage_and_prediction_identity(self):
        desktop, batch = self.paired()
        # Radius24 + 0.05cm * 4px/mm * 10mm/cm =26px; crop starts at(22,6).
        self.assertEqual(desktop['identity']['crop_offset'], [22, 6])
        affine = np.asarray(desktop['identity']['source_to_corrected'])
        self.assertFalse(np.allclose(affine, np.eye(3)))
        point = np.linalg.inv(self.reference_transform) @ [33., 29., 1.]
        expected = affine @ point
        observation = desktop['reviewed_measurements']['observations'][0]
        np.testing.assert_allclose(observation['annotation']['hilum_point'], expected[:2] / expected[2])
        self.assertEqual(self.result.pixels_per_mm, 4.)
        self.assertAlmostEqual(observation['scaled_bundle']['projected_area'],
                               observation['pixel_bundle']['projected_area'] / 16.)
        self.assertAlmostEqual(observation['scaled_bundle']['maximum_span'],
                               observation['pixel_bundle']['maximum_span'] / 4.)
        lineage = desktop['reviewed_measurements']['lineage']
        self.assertEqual(lineage['reference_archive_source_identity_state'], 'verified')
        self.assertEqual(lineage['reference_annotation_snapshot_source_identity_state'], 'not_recorded')
        self.assertTrue(self.task.node_cache.values['trait_classifier.features'].values.is_cuda)
        rows = desktop['trait_classifier']['predictions']
        for row in rows:
            self.assertAlmostEqual(sum(row['probabilities'].values()), 1.)
        self.assertNotEqual(rows[0]['probabilities'], rows[1]['probabilities'])
        self.assertFalse(batch['scientifically_validated'])
        self.window._project_biological_context = BiologicalContext('soybean', seed_lot_id='unconsumed-ui-value')
        self.assertEqual(self.window._current_report(self.result)['identity']['analysis_inputs'],
                         desktop['identity']['analysis_inputs'])

    def test_desktop_saved_review_reaches_batch_and_batch_revision_reaches_desktop(self):
        before, _ = self.paired()
        self.save_desktop_review()
        path = self.window._review_path(before)
        old_bytes = path.read_bytes()
        desktop, batch = self.paired()
        self.assertEqual(batch['review_lookup'], 'matched')
        self.assertEqual([row['review'] for row in batch['instances']], ['accepted', 'excluded'])
        self.assertEqual(before['trait_classifier'], batch['trait_classifier'])
        # An external consumer uses exactly the batch identity, without changing analysis.
        payload = json.loads(old_bytes)
        payload['result_revision'] = batch['result_revision']
        payload['decisions'] = {'1': 'excluded', '2': 'accepted'}
        path.write_text(json.dumps(payload), encoding='utf-8')
        fresh = self.window._current_report(self.result)
        self.assertEqual([row['review'] for row in fresh['instances']], ['excluded', 'accepted'])
        self.assertEqual(fresh['result_revision'], before['result_revision'])
        self.assertTrue(list((path.parent / 'history' / before['result_revision']).glob('*.json')))

    def test_layout_only_save_preserves_review_but_annotations_context_and_model_bytes_invalidate(self):
        before, _ = self.paired()
        self.save_desktop_review()
        path = self.window._review_path(before)
        historical = path.read_bytes()
        original_project_sha = sha256(self.master.read_bytes()).hexdigest()
        self.save_project(layout_x=250.)
        title, _ = self.paired()
        self.assertNotEqual(title['execution_provenance']['project_sha256'], original_project_sha)
        self.assertEqual(title['result_revision'], before['result_revision'])
        self.assertEqual(title['accepted_count'], 1)
        self.annotations = (replace(self.annotations[0], hilum_point=(32., 29.)), self.annotations[1])
        self.save_project()
        annotation, _ = self.paired()
        self.assertNotEqual(annotation['result_revision'], title['result_revision'])
        self.assertEqual(annotation['accepted_count'], 0)
        self.context = BiologicalContext('soybean', seed_lot_id='lot-b')
        self.save_project()
        context, _ = self.paired()
        self.assertNotEqual(context['result_revision'], annotation['result_revision'])
        self.model_path.write_bytes(self.model_path.read_bytes() + b'\n')
        model, _ = self.paired()
        self.assertNotEqual(model['result_revision'], context['result_revision'])
        self.assertEqual(model['accepted_count'], 0)
        self.assertEqual(path.read_bytes(), historical)
        self.window._cache_dirty_nodes[self.key] = {'project'}
        with self.assertRaisesRegex(ValueError, 'Recompute'):
            self.window._current_report(self.result)

    def test_missing_model_and_incompatible_review_are_explicit_on_both_routes(self):
        self.graph.set_parameter('trait_classifier', 'trait_model_path', 'missing.json')
        self.save_project()
        self.install_profile()
        desktop, _ = self.paired()
        self.assertTrue(all(row['trait_prediction']['abstention_reason'] == 'model_file_missing'
                            for row in desktop['instances']))
        path = self.window._review_path(desktop)
        path.parent.mkdir(exist_ok=True)
        raw = json.dumps({'result_revision': 'f' * 64, 'decisions': {'1': 'accepted'}}).encode()
        path.write_bytes(raw)
        desktop, batch = self.paired()
        self.assertEqual(batch['review_lookup'], 'incompatible')
        self.assertEqual(desktop['accepted_count'], 0)
        self.assertEqual(path.read_bytes(), raw)


class ReviewLookupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_per_image_contexts_survive_desktop_load_and_save_without_current_image_leakage(self):
        from seedvision.ui.main_window import MainWindow
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'config').mkdir()
            shutil.copyfile(Path(__file__).parents[1] / 'config' / 'traits.json',
                            root / 'config' / 'traits.json')
            paths = (root / 'first.png', root / 'second.png')
            for path in paths:
                self.assertTrue(cv2.imwrite(str(path), np.zeros((24, 24, 3), np.uint8)))
            contexts = (BiologicalContext('soybean', seed_lot_id='a'),
                        BiologicalContext('soybean', seed_lot_id='b'))
            master = ProjectAnalysisStore(root).capture_and_save(
                analysis_settings=analysis_settings_profile_from_graph(build_default_pipeline()),
                images=tuple(ProjectImageSpec(path, (24, 24), biological_context=context)
                             for path, context in zip(paths, contexts)), species='Soybean',
                biological_context=BiologicalContext('soybean', seed_lot_id='project-default'),
                destination=root / 'context.seedfiddle-project.json')
            window = MainWindow(root)
            try:
                with patch.object(window, '_confirm_project_replacement', return_value=True), \
                     patch.object(window, '_open_path'):
                    self.assertTrue(window._open_project(master))
                for path, context in zip(paths, contexts):
                    window._current_image_key = lambda path=path: str(path.resolve()).casefold()
                    self.assertEqual(window._effective_biological_context(), context)
                self.assertEqual(tuple(spec.biological_context for spec in window._project_image_specs()), contexts)
            finally:
                window.close()

    def test_missing_project_species_loads_unknown_and_malformed_capture_cannot_use_legacy_fallback(self):
        from seedvision.ui.main_window import MainWindow
        from seedvision.export.results import result_report
        import tests.test_result_prediction_revision as result_fixture
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'source.png'
            self.assertTrue(cv2.imwrite(str(path), np.zeros((24, 24, 3), np.uint8)))
            master = ProjectAnalysisStore(root).capture_and_save(
                analysis_settings=analysis_settings_profile_from_graph(build_default_pipeline()),
                images=(ProjectImageSpec(path, (24, 24)),),
                destination=root / 'unknown.seedfiddle-project.json')
            window = MainWindow(root)
            try:
                with patch.object(window, '_confirm_project_replacement', return_value=True), \
                     patch.object(window, '_open_path'):
                    self.assertTrue(window._open_project(master))
                self.assertEqual(window.species_combo.currentText(), 'unknown')
                self.assertEqual(window._effective_biological_context(), BiologicalContext('unknown'))
                analysis = result_fixture._analysis()
                analysis.analysis_inputs_json = '[]'
                with self.assertRaisesRegex(ValueError, 'unsupported contract'):
                    result_report(analysis)
            finally:
                window.close()

    def test_invalid_duplicate_oversized_and_stale_reviews_remain_distinct_and_unchanged(self):
        from seedvision.export.review import lookup_result_review, MAX_REVIEW_BYTES
        with TemporaryDirectory() as directory:
            root = Path(directory)
            revision = 'a' * 64
            self.assertEqual(lookup_result_review(root, revision), (None, 'not_recorded'))
            path = root / 'result-reviews' / (revision + '.json')
            path.parent.mkdir()
            for raw in (b'{', b'[]', b'x' * (MAX_REVIEW_BYTES + 1),
                        ('{"result_revision":"' + revision + '","decisions":{"1":"accepted","1":"excluded"}}').encode(),
                        json.dumps({'result_revision': revision, 'decisions': {'1': []}}).encode()):
                path.write_bytes(raw)
                self.assertEqual(lookup_result_review(root, revision), (None, 'malformed'))
                self.assertEqual(path.read_bytes(), raw)
            self.assertEqual(lookup_result_review(root, '../elsewhere'), (None, 'unavailable'))


if __name__ == '__main__':
    unittest.main()
