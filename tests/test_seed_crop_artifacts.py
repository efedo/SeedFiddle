
from tests.functional_units import functional_units
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np


def _analysis(source=None):
    import torch
    from dataclasses import MISSING, fields
    from seedvision.calibration.image import ImageCalibration
    from seedvision.segmentation.baseline import BaselineAnalysis
    from seedvision.segmentation.procedural import ProceduralInstanceResult
    corrected = np.arange(19 * 23 * 3, dtype=np.uint8).reshape(19, 23, 3)
    image = torch.from_numpy(corrected.copy()).to('cuda:0').permute(2, 0, 1)[None].float()
    labels = np.zeros((6, 7), np.int32)
    labels[1:3, 1:4] = 1
    labels[1:3, 4:6] = 2
    instances = ProceduralInstanceResult(labels=labels, centres_xy=np.zeros((2, 2), np.float32),
        marker_scores=np.ones(2), instance_confidences=np.array([.8, .7]),
        occupancy_likelihood=np.ones(labels.shape), occupancy_mask=labels > 0,
        boundary_cost=np.zeros(labels.shape), centre_likelihood=np.ones(labels.shape),
        source_shape=(9, 11), working_scale=6 / 9)
    calibration = ImageCalibration(corrected, None, None, np.array([[1., 0., 2.], [0., 1., -1.], [0., 0., 1.]]),
        0., False, 0., (1., 1., 1.), None, 0., None, 10., (), gpu_corrected_bgr=image)
    required = {field.name: None for field in fields(BaselineAnalysis)
                if field.default is MISSING and field.default_factory is MISSING}
    required.update(image_path=source, calibration=calibration, crop_offset=(3, 4), warnings=(),
        layers=SimpleNamespace(foreground_colour_profile=object(), valid_mask=np.ones((9, 11), bool), offset_x=3, offset_y=4),
        procedural_instances=instances, proposals=(), approximate=True)
    return BaselineAnalysis(**required)


@functional_units
class SeedCropArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        if not torch.cuda.is_available():
            raise unittest.SkipTest('CUDA is required for explicit seed crop serialization.')

    def check_test_exported_pixels_masks_coordinates_and_hashes_are_bound_to_result(self):
        from hashlib import sha256
        from seedvision.export.results import result_report
        from seedvision.export.crop_artifacts import export_seed_crop_artifacts
        analysis = _analysis()
        report = result_report(analysis)
        report['instances'][0]['review'] = 'accepted'
        with TemporaryDirectory() as directory:
            destination = export_seed_crop_artifacts(directory, analysis, report, seed_ids=(1, 99), padding_px=1)
            payload = json.loads((destination / 'seed-crops.json').read_text())
            self.assertEqual(payload['result_revision'], report['result_revision'])
            self.assertFalse(payload['scientifically_validated'])
            available, missing = payload['records']
            self.assertEqual(available['review'], 'accepted')
            self.assertEqual(missing['status'], 'missing')
            self.assertEqual(missing['artifacts'], {})
            x0, y0, x1, y1 = available['corrected_bbox']
            image = cv2.imread(str(destination / available['artifacts']['image']['path']))
            mask = cv2.imread(str(destination / available['artifacts']['mask']['path']), cv2.IMREAD_GRAYSCALE)
            np.testing.assert_array_equal(image, analysis.calibration.corrected_bgr[y0:y1, x0:x1])
            self.assertEqual(mask.shape, image.shape[:2])
            self.assertEqual(set(np.unique(mask)), {0, 255})
            self.assertIn(2, available['foreign_seed_ids'])
            translation = np.eye(3)
            translation[:2, 2] = (-x0, -y0)
            np.testing.assert_array_equal(available['source_to_crop'], translation @ analysis.calibration.affine_matrix)
            for artifact in available['artifacts'].values():
                self.assertEqual(sha256((destination / artifact['path']).read_bytes()).hexdigest(), artifact['sha256'])

    def check_test_padding_changes_crop_and_each_export_preserves_previous_artifacts(self):
        from seedvision.export.results import result_report
        from seedvision.export.crop_artifacts import export_seed_crop_artifacts
        analysis = _analysis()
        report = result_report(analysis)
        with TemporaryDirectory() as directory:
            one = export_seed_crop_artifacts(directory, analysis, report, seed_ids=(1,), padding_px=0)
            before = (one / 'seed-1-image.png').read_bytes()
            two = export_seed_crop_artifacts(directory, analysis, report, seed_ids=(1,), padding_px=3)
            self.assertNotEqual(one, two)
            self.assertEqual(before, (one / 'seed-1-image.png').read_bytes())
            self.assertNotEqual(cv2.imread(str(one / 'seed-1-image.png')).shape,
                                cv2.imread(str(two / 'seed-1-image.png')).shape)

    def check_test_changed_labels_and_source_are_rejected_before_output(self):
        from seedvision.export.results import result_report
        from seedvision.export.crop_artifacts import export_seed_crop_artifacts
        analysis = _analysis()
        report = result_report(analysis)
        analysis.procedural_instances.labels[4, 4] = 1
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'Result changed'):
                export_seed_crop_artifacts(directory, analysis, report, seed_ids=(1,), padding_px=0)
            self.assertEqual(list(Path(directory).iterdir()), [])
        with TemporaryDirectory() as directory:
            source = Path(directory) / 'source.png'
            cv2.imwrite(str(source), np.zeros((19, 23, 3), np.uint8))
            analysis = _analysis(source)
            report = result_report(analysis)
            cv2.imwrite(str(source), np.full((19, 23, 3), 100, np.uint8))
            with self.assertRaisesRegex(ValueError, 'Result changed'):
                export_seed_crop_artifacts(Path(directory) / 'output', analysis, report, seed_ids=(1,), padding_px=0)
            self.assertFalse((Path(directory) / 'output').exists())

    def check_test_cancellation_and_encoding_failure_never_publish_complete_revision(self):
        from seedvision.export.results import result_report
        from seedvision.export.crop_artifacts import export_seed_crop_artifacts, CropExportCancelled
        analysis = _analysis()
        report = result_report(analysis)
        cancelled = Event()
        cancelled.set()
        with TemporaryDirectory() as directory:
            with self.assertRaises(CropExportCancelled):
                export_seed_crop_artifacts(directory, analysis, report, seed_ids=(1,), padding_px=0, cancel=cancelled)
            self.assertEqual(list(Path(directory).iterdir()), [])
            with patch('seedvision.export.crop_artifacts.cv2.imwrite', return_value=False):
                with self.assertRaises(OSError):
                    export_seed_crop_artifacts(directory, analysis, report, seed_ids=(1,), padding_px=0)
            self.assertFalse(list(Path(directory).glob('seed-crops-*')))
            self.assertFalse(list(Path(directory).rglob('seed-crops.json')))

    def check_test_file_backed_corrected_pixels_crop_offset_and_scale_change_revision(self):
        from dataclasses import replace
        from seedvision.export.results import result_report
        from seedvision.export.crop_artifacts import export_seed_crop_artifacts
        with TemporaryDirectory() as directory:
            source = Path(directory) / 'source.png'
            cv2.imwrite(str(source), np.zeros((19, 23, 3), np.uint8))
            for mutation in ('pixels', 'offset', 'scale'):
                analysis = _analysis(source)
                report = result_report(analysis)
                if mutation == 'pixels':
                    analysis.calibration.corrected_bgr[5, 5] = (123, 42, 15)
                elif mutation == 'offset':
                    analysis = replace(analysis, crop_offset=(4, 4))
                else:
                    analysis = replace(analysis, procedural_instances=replace(
                        analysis.procedural_instances, source_shape=(10, 11)))
                with self.subTest(mutation=mutation):
                    self.assertNotEqual(result_report(analysis)['result_revision'], report['result_revision'])
                    output = Path(directory) / mutation
                    with self.assertRaisesRegex(ValueError, 'Result changed'):
                        export_seed_crop_artifacts(output, analysis, report, seed_ids=(1,), padding_px=0)
                    self.assertFalse(output.exists())

    def check_test_gpu_crop_drift_is_rejected_before_publication(self):
        from seedvision.export.results import result_report
        from seedvision.export.crop_artifacts import export_seed_crop_artifacts
        analysis = _analysis()
        report = result_report(analysis)
        analysis.calibration.gpu_corrected_bgr[:, :, 4:10, 3:11] = 42
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'CUDA crop pixels differ'):
                export_seed_crop_artifacts(directory, analysis, report, seed_ids=(1,), padding_px=1)
            self.assertFalse(list(Path(directory).glob('seed-crops-*')))
            self.assertFalse(list(Path(directory).rglob('seed-crops.json')))

    def check_test_result_mutation_during_encoding_prevents_publication(self):
        from seedvision.export.results import result_report
        from seedvision.export.crop_artifacts import export_seed_crop_artifacts
        analysis = _analysis()
        report = result_report(analysis)
        imwrite = cv2.imwrite
        def changed_while_writing(path, raster):
            analysis.procedural_instances.labels[4, 4] = 1
            return imwrite(path, raster)
        with TemporaryDirectory() as directory:
            with patch('seedvision.export.crop_artifacts.cv2.imwrite', side_effect=changed_while_writing):
                with self.assertRaisesRegex(ValueError, 'Result changed during'):
                    export_seed_crop_artifacts(directory, analysis, report, seed_ids=(1,), padding_px=0)
            self.assertFalse(list(Path(directory).glob('seed-crops-*')))


    def check_test_window_stop_reaches_export_worker_cancellation_event(self):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        from PySide6.QtWidgets import QApplication
        from seedvision.ui.main_window import MainWindow
        from seedvision.ui.results import _CropExportTask
        from seedvision.export.results import result_report
        from seedvision.export.crop_artifacts import CropExportCancelled
        app = QApplication.instance() or QApplication([])
        started = Event()
        received = []
        def exporting(*args, cancel, **kwargs):
            started.set()
            if cancel.wait(10):
                received.append(True)
                raise CropExportCancelled('Stopped at the export boundary.')
            raise AssertionError('The window did not forward Stop to crop export.')
        with TemporaryDirectory() as directory:
            window = MainWindow(Path(directory))
            analysis = _analysis()
            task = _CropExportTask(directory, analysis, result_report(analysis), (1,), 0)
            window._result_export_task = task
            with patch('seedvision.export.crop_artifacts.export_seed_crop_artifacts', side_effect=exporting):
                window._thread_pool.start(task)
                self.assertTrue(started.wait(5))
                window._stop_current_work()
                self.assertTrue(window._thread_pool.waitForDone(10000))
                app.processEvents()
            self.assertEqual(received, [True])
            window._result_export_task = None
            window.close()


    def check_test_batch_cli_surfaces_partial_crop_manifest(self):
        from contextlib import redirect_stderr, redirect_stdout
        import io
        from scripts.analyze_pilot import main
        from seedvision.export.results import result_report
        from seedvision.export.crop_artifacts import export_seed_crop_artifacts
        analysis = _analysis()
        with TemporaryDirectory() as directory:
            output = Path(directory) / 'report'
            manifest = export_seed_crop_artifacts(output, analysis, result_report(analysis),
                                                  seed_ids=(1, 99), padding_px=0)
            stderr, stdout = io.StringIO(), io.StringIO()
            with patch('sys.argv', ['analyze_pilot.py', '--project', 'unused', '--seed-crop-padding-px', '0']), \
                 patch('scripts.analyze_pilot.run_project', return_value=[output]), \
                 redirect_stderr(stderr), redirect_stdout(stdout):
                self.assertEqual(main(), 0)
            self.assertIn(str(output), stdout.getvalue())
            self.assertIn('1 crops exported; 1 requested seeds have no crop', stderr.getvalue())
            self.assertIn(str(manifest / 'seed-crops.json'), stderr.getvalue())


    def test_workflow_lifecycle(self):
        """4 related SeedCropArtifactTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_exported_pixels_masks_coordinates_and_hashes_are_bound_to_result)
        self.scenario(self.check_test_padding_changes_crop_and_each_export_preserves_previous_artifacts)
        self.scenario(self.check_test_file_backed_corrected_pixels_crop_offset_and_scale_change_revision)
        self.scenario(self.check_test_batch_cli_surfaces_partial_crop_manifest)

    def test_validation_and_recovery(self):
        """5 related SeedCropArtifactTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_changed_labels_and_source_are_rejected_before_output)
        self.scenario(self.check_test_cancellation_and_encoding_failure_never_publish_complete_revision)
        self.scenario(self.check_test_gpu_crop_drift_is_rejected_before_publication)
        self.scenario(self.check_test_result_mutation_during_encoding_prevents_publication)
        self.scenario(self.check_test_window_stop_reaches_export_worker_cancellation_event)


if __name__ == '__main__':
    unittest.main()
