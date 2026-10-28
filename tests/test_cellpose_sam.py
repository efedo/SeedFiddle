"""Optional Cellpose-SAM branch: honest settings, flow decoding and engine caching."""

from tests.functional_units import functional_units
from dataclasses import replace
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from seedvision.learning import cellpose_sam
from seedvision.learning.cellpose_sam import CellposeSamSettings, decode_cellpose_sam, resolve_weights
from seedvision.learning.contracts import LearnedInstanceResult, PretrainedModelFamily

INSTALLED = cellpose_sam.cellpose_installed()


def _discs(shape=(160, 200), centres=((50, 50), (50, 130), (115, 90)), radius=22):
    labels = np.zeros(shape, np.int32)
    for identifier, (row, column) in enumerate(centres, start=1):
        cv2.circle(labels, (column, row), radius, identifier, -1)
    return labels


def _flows(labels):
    from cellpose import dynamics

    flows, _ = dynamics.masks_to_flows_gpu(labels, device=torch.device('cpu'))
    logits = np.where(labels > 0, 6.0, -6.0).astype(np.float32)
    # Cellpose networks are trained to predict five times the unit flow.
    return {'flows': np.asarray(5.0 * flows, np.float32), 'cell_probability_logits': logits,
            'diameter_px': None, 'device': 'cpu'}


@functional_units
class CellposeSamTests(unittest.TestCase):
    def check_test_settings_reject_values_no_calculation_could_use(self):
        for change in (dict(pretrained_model=' '), dict(flow_threshold=-0.1), dict(flow_threshold=3.5),
                       dict(cellprob_threshold=6.5), dict(minimum_instance_area_fraction=0.0),
                       dict(minimum_instance_area_fraction=2.0)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                CellposeSamSettings(**change)
        settings = CellposeSamSettings()
        self.assertEqual(settings.inference_signature, ('cpsam_v2', False))
        self.assertNotEqual(replace(settings, flow_threshold=0.8).decoder_signature, settings.decoder_signature)

    def check_test_weights_resolve_paths_project_files_then_cellpose_cache(self):
        with TemporaryDirectory() as directory:
            root, cache = Path(directory) / 'project', Path(directory) / 'cache'
            root.mkdir()
            (root / 'tuned.pt').write_bytes(b'x')
            with patch.dict(os.environ, {'CELLPOSE_LOCAL_MODELS_PATH': str(cache)}):
                self.assertEqual(resolve_weights('cpsam_v2', root), (cache / 'cpsam_v2').resolve())
                self.assertEqual(resolve_weights('tuned.pt', root), (root / 'tuned.pt').resolve())
                self.assertEqual(resolve_weights(str(cache / 'a.pt'), root), (cache / 'a.pt').resolve())

    @unittest.skipUnless(INSTALLED, 'optional cellpose package is not installed')
    def check_test_decoder_follows_flows_and_drops_fragments_outside_the_dish(self):
        labels = _discs()
        outputs = _flows(labels)
        result = decode_cellpose_sam(outputs, settings=CellposeSamSettings(), seed_diameter_px=44.0,
                                     checkpoint_id='weights#sha256=x')
        self.assertIs(result.family, PretrainedModelFamily.CELLPOSE_SAM)
        self.assertEqual(int(result.labels.max()), 3)
        np.testing.assert_allclose(sorted(map(tuple, result.centres_xy.round())),
                                   [(50., 50.), (90., 115.), (130., 50.)], atol=1.0)
        self.assertTrue(np.all(result.instance_confidences > 0.99))
        self.assertEqual(result.rasters['cell_probability'].dtype, np.uint8)
        dish = np.ones(labels.shape, np.uint8)
        dish[:, 120:] = 0
        clipped = decode_cellpose_sam(outputs, settings=CellposeSamSettings(), seed_diameter_px=44.0,
                                      valid_mask=dish)
        # The right-hand disc is centred outside the dish; the others stay whole.
        self.assertEqual(int(clipped.labels.max()), 2)
        self.assertFalse(np.any(clipped.labels[:, 120:]))
        strict = decode_cellpose_sam(outputs, settings=CellposeSamSettings(minimum_instance_area_fraction=1.0),
                                     seed_diameter_px=44.0)
        # pi * 22^2 is about 0.79 of 44^2, so a full-diameter minimum removes every disc.
        self.assertEqual(int(strict.labels.max()), 0)
        unconfident = decode_cellpose_sam(outputs, settings=CellposeSamSettings(cellprob_threshold=6.0),
                                          seed_diameter_px=44.0)
        self.assertEqual(int(unconfident.labels.max()), 0)

    def check_test_engine_reuses_inference_for_decoder_edits_and_reports_missing_dependency(self):
        from seedvision import calibration
        from seedvision.calibration.geometry import DishDetectionSettings
        from seedvision.export import backend_outcomes
        from seedvision.segmentation.baseline import PipelineAnalysisCache, analyze_path
        from tests.test_neutral_calibration import fixture, ruler

        with TemporaryDirectory() as directory, patch.dict(os.environ, {'CELLPOSE_LOCAL_MODELS_PATH': directory}):
            source = Path(directory) / 'a.png'
            image, card = fixture()
            self.assertTrue(cv2.imwrite(str(source), image))
            weights = Path(directory) / 'cpsam_v2'
            weights.write_bytes(b'stand-in weights')
            calls = {'predict': 0, 'decode': []}

            def predict(settings, **kwargs):
                calls['predict'] += 1
                path = resolve_weights(settings.pretrained_model)
                return {'flows': None}, f'{path}#sha256={cellpose_sam.weights_sha256(path)}'

            def decode(outputs, *, settings, valid_mask, checkpoint_id, **kwargs):
                calls['decode'].append(settings.flow_threshold)
                return LearnedInstanceResult(PretrainedModelFamily.CELLPOSE_SAM, np.zeros(valid_mask.shape, np.int32),
                                             np.zeros((0, 2), np.float32), np.zeros(0, np.float32),
                                             checkpoint_id=checkpoint_id)

            cache = PipelineAnalysisCache()

            def run(settings, *, installed=True, dirty=frozenset()):
                with patch.object(calibration, 'detect_colour_card', return_value=card), \
                        patch.object(calibration, 'detect_ruler', side_effect=ruler), \
                        patch.object(cellpose_sam, 'cellpose_installed', return_value=installed), \
                        patch.object(cellpose_sam, 'predict_cellpose_sam', side_effect=predict), \
                        patch.object(cellpose_sam, 'decode_cellpose_sam', side_effect=decode):
                    return analyze_path(
                        source, calibration_settings=calibration.CalibrationSettings(apply_perspective_correction=False),
                        dish_settings=DishDetectionSettings(expected_outer_diameter_mm=0),
                        enabled_nodes=frozenset({'cellpose_sam_instances'}), node_cache=cache,
                        dirty_nodes=dirty, learning_root=Path(directory), cellpose_settings=settings)

            run(CellposeSamSettings())
            run(CellposeSamSettings(flow_threshold=0.8), dirty=frozenset({'cellpose_sam_instances'}))
            self.assertEqual(calls, {'predict': 1, 'decode': [0.4, 0.8]})
            run(CellposeSamSettings(rescale_to_seed_diameter=True), dirty=frozenset({'cellpose_sam_instances'}))
            self.assertEqual(calls['predict'], 2)
            missing = run(CellposeSamSettings(), installed=False, dirty=frozenset({'cellpose_sam_instances'}))
            self.assertEqual(calls['predict'], 2)
            self.assertIsNone(missing.cellpose_sam_instances)
            record = json.loads(missing.backend_outcomes_json)['outcomes']['cellpose_sam_instances']
            self.assertEqual((record['state'], record['reason_code']), ('unavailable', 'dependency_missing'))
            self.assertEqual(record['backend_id'], backend_outcomes.BACKENDS['cellpose_sam_instances'])

    def check_test_native_worker_forwards_cellpose_settings(self):
        import inspect
        from types import SimpleNamespace
        from seedvision.ui.main_window import _AnalysisTask

        arguments = {name: None for name, parameter in inspect.signature(_AnalysisTask).parameters.items()
                     if parameter.default is inspect.Parameter.empty}
        settings = CellposeSamSettings(flow_threshold=0.9)
        with TemporaryDirectory() as directory:
            arguments.update(path=Path(directory) / 'capture.png', learning_root=Path(directory), pipeline_revision=1,
                species='unknown', seed_trait_species='', background_colour_enabled=True,
                background_reference_points=(), foreground_reference_points=(), seed_instance_traits=(),
                seed_trait_coat_patterns=(), seed_trait_conditions=(),
                enabled_nodes=frozenset({'cellpose_sam_instances'}), dirty_nodes=frozenset(),
                cellpose_settings=settings)
            task = _AnalysisTask(**arguments)
            failures = []
            task.signals.failed.connect(lambda *args: failures.append(args))
            with patch('seedvision.segmentation.baseline.analyze_path', return_value=SimpleNamespace()) as analyzer:
                task.run()
        self.assertEqual(failures, [])
        self.assertIs(analyzer.call_args.kwargs['cellpose_settings'], settings)

    def test_branch_contract(self):
        """5 related CellposeSamTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_settings_reject_values_no_calculation_could_use)
        self.scenario(self.check_test_weights_resolve_paths_project_files_then_cellpose_cache)
        self.scenario(self.check_test_decoder_follows_flows_and_drops_fragments_outside_the_dish)
        self.scenario(self.check_test_engine_reuses_inference_for_decoder_edits_and_reports_missing_dependency)
        self.scenario(self.check_test_native_worker_forwards_cellpose_settings)


@unittest.skipUnless(INSTALLED and torch.cuda.is_available()
                     and resolve_weights(cellpose_sam.DEFAULT_MODEL).is_file(),
                     'Cellpose-SAM, CUDA or its cached cpsam_v2 weights are unavailable')
class CellposeSamModelTests(unittest.TestCase):
    def test_pretrained_model_separates_dark_seeds_on_a_light_dish(self):
        canvas = np.full((240, 320, 3), 225, np.uint8)
        labels = _discs((240, 320), ((70, 70), (70, 160), (70, 250), (170, 115), (170, 205)), radius=26)
        canvas[labels > 0] = (40, 70, 110)
        source = torch.from_numpy(canvas).to('cuda').permute(2, 0, 1)[None].float()
        settings = CellposeSamSettings()
        outputs, producer = cellpose_sam.predict_cellpose_sam(
            settings, root=Path.cwd(), source_bgr=source, seed_diameter_px=52.0)
        self.assertIn('#sha256=', producer)
        result = decode_cellpose_sam(outputs, settings=settings, seed_diameter_px=52.0, checkpoint_id=producer)
        self.assertEqual(int(result.labels.max()), 5)


if __name__ == '__main__':
    unittest.main()
