"""Real mean/Otsu desktop sessions; stored-result checks are not a new oracle."""
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import cv2
import numpy as np
from PySide6.QtWidgets import QApplication

from seedvision.benchmarking.checkpoints import _thaw_json
from seedvision.benchmarking.reference_runner import PlantCVReferenceRunner
from seedvision.ui import calibrated_public_capture_dialog as ui
from tests.test_calibrated_public_capture_dialog import wait_for_worker
from tests.test_native_calibrated_seed_recipe import fixture
from tests.test_reference_calibrated_seed import spec
from tests.test_reference_preprocessing import HAS_REFERENCE_ENV, REFERENCE_PYTHON


@unittest.skipUnless(HAS_REFERENCE_ENV,'Pinned public PlantCV environment is required.')
class CalibratedPublicMethodRoundTripTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _round_trip(self,kind):
        with tempfile.TemporaryDirectory(prefix='calibrated-public-method-') as directory:
            root = Path(directory)
            source,store = root/'source.png',root/'store'
            bgr = fixture()
            ok,encoded = cv2.imencode('.png',bgr)
            self.assertTrue(ok)
            source.write_bytes(encoded.tobytes())
            original = spec(kind)
            dialog = ui.CalibratedPublicReferenceCaptureDialog(original.seed_recipe,
                source_path=source,store=store,card_parameters=_thaw_json(original.card_parameters),
                seed_roi_xyxy=original.seed_roi_xyxy,
                runner_factory=lambda:PlantCVReferenceRunner(REFERENCE_PYTHON))
            self.addCleanup(dialog.close)
            threshold = next(stage for stage in original.seed_recipe.stages
                             if stage.operation_id=='threshold.'+kind)
            if kind=='mean':
                dialog._editors[threshold.stage_id,'ksize'].setText('8')
            selected = dialog.selected_spec()
            requests,results = [],[]
            actual = PlantCVReferenceRunner.run
            def observe(runner,operation,arrays,parameters=None,**kwargs):
                requests.append((operation,deepcopy(parameters)))
                result = actual(runner,operation,arrays,parameters,**kwargs)
                results.append(result)
                return result
            with patch.object(PlantCVReferenceRunner,'run',observe):
                dialog.start_capture()
                wait_for_worker(self.app,dialog)
            self.assertIsNotNone(dialog._reply,dialog.status_label.text())
            session = dialog._reply.session
            self.assertIsNotNone(session.verified,dialog.status_label.text())
            checkpoints,record = session.verified.checkpoints,session.verified.record
            self.assertEqual(len(requests),10)
            self.assertEqual(len(results),10)
            self.assertEqual(len(checkpoints),15)
            self.assertEqual(session.spec,selected)
            self.assertEqual(dialog._recipe,original.seed_recipe)
            self.assertEqual(record['source_sha256'],sha256(encoded.tobytes()).hexdigest())
            self.assertEqual(record['complete_required_verified_public_calls'],10)
            self.assertFalse(record['scientifically_validated'])
            self.assertEqual(record['crop']['local_to_parent_xy'],(600,260))
            self.assertEqual(record['crop']['coordinate_frame']['shape'],(96,128))
            np.testing.assert_array_equal(checkpoints['parent'].arrays['rgb'],bgr[...,::-1])
            # Retention checks compare the actual results of this run with its
            # saved checkpoints; prior direct-source oracle evidence is separate.
            np.testing.assert_array_equal(checkpoints['correction'].arrays['corrected_image'],
                                          results[0].arrays['corrected_image'])
            np.testing.assert_array_equal(checkpoints['seed.preprocessing.1'].arrays['mask'],
                                          results[2].arrays['mask'])
            np.testing.assert_array_equal(checkpoints['seed.watershed'].arrays['labels'],
                                          results[6].arrays['labels'])
            self.assertEqual(_thaw_json(checkpoints['seed.color'].metadata['observations']),
                             dict(results[8].observations))
            self.assertEqual(_thaw_json(checkpoints['scaled_size'].metadata['observations']),
                             dict(results[9].observations))
            self.assertEqual(requests[2][0],'threshold.'+kind)
            metadata = checkpoints['seed.preprocessing.1'].metadata
            if kind=='mean':
                self.assertEqual(requests[2][1]['ksize'],8)
                self.assertEqual(metadata['parameters']['ksize'],8)
                self.assertEqual(metadata['effective_parameters']['ksize'],9)
                self.assertEqual(selected.seed_recipe.parameters_for_runner(threshold.stage_id)['ksize'],8)
            else:
                self.assertEqual(metadata['effective_parameters']['callable'],'plantcv.plantcv.threshold.otsu')
                self.assertEqual(dict(metadata['parameters']),requests[2][1])
            self.assertEqual(checkpoints['seed.size'].measurement.units,'pixel')
            self.assertEqual(checkpoints['scaled_size'].measurement.units,'mm')
            calibration = checkpoints['scaled_size'].metadata['effective_parameters']['calibration']
            self.assertEqual(dict(calibration),requests[-1][1]['calibration'])
            self.assertEqual(record['scaled_measurement']['units'],'mm')
            manifest = session.manifest_path
            source.unlink()
            reopened = ui.CalibratedPublicReferenceCaptureDialog(spec().seed_recipe,
                store=root/'wrong-store',seed_roi_xyxy=(0,0,2,2))
            self.addCleanup(reopened.close)
            with patch.object(ui.inputs,'source_identity',side_effect=AssertionError('Original source I/O')),\
                    patch.object(PlantCVReferenceRunner,'run',side_effect=AssertionError('Public rerun')):
                reopened.open_saved(manifest)
                wait_for_worker(self.app,reopened)
            self.assertIsNotNone(reopened._reply,reopened.status_label.text())
            restored = reopened._reply.session
            self.assertEqual(restored.spec,selected)
            self.assertEqual(reopened.selected_spec(),selected)
            self.assertEqual(reopened._recipe,selected.seed_recipe)
            self.assertIsNone(reopened._source_path)
            self.assertEqual(reopened._store,store.resolve())
            self.assertEqual(restored.manifest_sha256,session.manifest_sha256)
            self.assertEqual(set(restored.verified.checkpoints),set(checkpoints))
            for name,checkpoint in checkpoints.items():
                self.assertEqual(restored.verified.checkpoints[name].checkpoint_sha256,
                                 checkpoint.checkpoint_sha256)
            facts = json.loads(reopened.facts.toPlainText())
            self.assertEqual(facts['run']['composition'],selected.to_dict())
            self.assertIn('Verified complete calibrated capture',reopened.status_label.text())
            if kind=='mean':
                self.assertEqual(reopened._editors[threshold.stage_id,'ksize'].text(),'8')
                self.assertEqual(facts['actual_source_metadata']['seed.preprocessing.1']['effective_parameters']['ksize'],9)
            reopened.close()
            dialog.close()

    def test_actual_mean_even_request_capture_save_and_reopen(self):
        self._round_trip('mean')

    def test_actual_otsu_capture_save_and_reopen(self):
        self._round_trip('otsu')


if __name__ == '__main__':
    unittest.main()
