"""Desktop boundary witnesses; no inferred outputs from thrown exceptions."""

from tests.functional_units import functional_units
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from dataclasses import replace
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import tempfile
import threading
import unittest
from tests.catalogue_fixtures import CatalogueTestCase
from unittest.mock import patch

from PySide6.QtWidgets import QApplication

from seedvision.benchmarking.checkpoints import CheckpointCancelled
from seedvision.benchmarking.native_image_inputs import stamp
from seedvision.benchmarking.recipe_catalogue import build_recipe_catalogue
from seedvision.benchmarking.reference_runner import ReferenceCancelledError
from seedvision.export.method_qc import MethodQC
from seedvision.export.public_reference_attempt import capture_public_reference_attempt
from seedvision.persistence.method_qc_history import load_project_method_qc
from seedvision.ui import public_reference_capture_dialog as ui


@functional_units
class PublicReferenceAttemptBoundaryTests(CatalogueTestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='public-reference-attempt-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'selected.png'
        self.source.write_bytes(b'Immutable selected bytes; decoder failure is injected.')
        self.recipe = build_recipe_catalogue().recipe('plantcv_seed_binary')
        self.request = ui.CaptureRequest(1, self.recipe, self.source, self.root / 'capture',
            stamp(self.source), history_root=self.root)

    def worker(self, request=None, *, factory=None, cancel=None):
        worker = ui._CaptureWorker(request or self.request, cancel or threading.Event(),
            factory or (lambda: object()))
        worker.delivered = []
        worker.signals.completed.connect(lambda *args: worker.delivered.append(args))
        return worker

    def run_failure(self, failure, **kwargs):
        worker = self.worker(**kwargs)
        with patch.object(ui, 'capture_reference_path', side_effect=failure):
            worker.run()
        self.assertIs(worker.delivered[0][2], failure)
        self.assertIsNone(worker.delivered[0][1])
        return worker

    def row(self, worker):
        return worker.qualities[-1].payload['identity']['reference_capture']

    def check_test_failed_wrapper_retains_original_error_and_exact_request_bytes(self):
        failure = ValueError('Actual decoder failure')
        worker = self.run_failure(failure)
        row = self.row(worker)
        self.assertEqual(row['expected_source_sha256'], sha256(self.source.read_bytes()).hexdigest())
        self.assertEqual(row['inputs_before'][0]['observation']['state'], 'hashed')
        self.assertEqual(row['state'], 'failed')
        self.assertEqual(row['phase'], 'capture')
        self.assertTrue(row['invoked_executor'].endswith('capture_reference_path'))
        self.assertIsNone(row['record'])
        history = load_project_method_qc(self.root)
        self.assertFalse(history.diagnostics)
        self.assertEqual([MethodQC(raw.decode()).revision for raw in history.records],
                         [worker.qualities[0].revision])

    def check_test_missing_source_keeps_unknown_identity_and_actual_input_observation(self):
        missing = self.root / 'absent.png'
        worker = self.run_failure(FileNotFoundError('Selected source absent'),
            request=replace(self.request, source_path=missing, source_stamp=None))
        row = self.row(worker)
        self.assertIsNone(row['expected_source_sha256'])
        self.assertEqual(row['inputs_before'][0]['observation']['state'], 'missing')
        self.assertIsNone(row['inputs_before'][0]['observation']['sha256'])
        self.assertEqual(worker.qualities[-1].payload['version'], 5)

    def check_test_supplied_expected_source_is_not_overwritten_by_changed_observation(self):
        original = sha256(self.source.read_bytes()).hexdigest()
        self.source.write_bytes(b'Changed selected bytes')
        worker = self.run_failure(ValueError('Selected image changed before preparation'),
            request=replace(self.request, expected_source_sha256=original))
        row = self.row(worker)
        self.assertEqual(row['expected_source_sha256'], original)
        self.assertNotEqual(row['inputs_before'][0]['observation']['sha256'], original)

    def check_test_nullable_source_version_cannot_relabel_known_or_complete_capture(self):
        import json
        worker = self.run_failure(FileNotFoundError('No source'),
            request=replace(self.request, source_path=self.root / 'missing.png', source_stamp=None))
        missing = worker.qualities[-1].payload
        wrong_version = deepcopy(missing)
        wrong_version['version'] = 4
        with self.assertRaises(ValueError):
            MethodQC(json.dumps(wrong_version))
        known = self.run_failure(ValueError('Known source failure')).qualities[-1].payload
        known['version'] = 5
        with self.assertRaises(ValueError):
            MethodQC(json.dumps(known))

    def check_test_runner_construction_failure_does_not_claim_wrapper_invocation(self):
        failure = OSError('Pinned interpreter unavailable')
        def factory():
            raise failure
        worker = self.worker(factory=factory)
        with patch.object(ui, 'capture_reference_path') as capture:
            worker.run()
        capture.assert_not_called()
        self.assertIs(worker.delivered[0][2], failure)
        self.assertEqual(self.row(worker)['phase'], 'admission')
        self.assertIsNone(self.row(worker)['invoked_executor'])

    def check_test_both_actual_cancellation_types_remain_cancellation(self):
        for failure in (CheckpointCancelled('Session cancelled'), ReferenceCancelledError('Public call cancelled')):
            with self.subTest(type=type(failure).__name__):
                worker = self.run_failure(failure)
                self.assertEqual(self.row(worker)['state'], 'cancelled')
                self.assertEqual(self.row(worker)['error_type'], type(failure).__name__)

    def check_test_stop_before_invocation_retains_request_without_calling_runner(self):
        event = threading.Event()
        event.set()
        worker = self.worker(cancel=event)
        with patch.object(ui, 'capture_reference_path') as capture:
            worker.run()
        capture.assert_not_called()
        self.assertEqual(self.row(worker)['state'], 'cancelled')
        self.assertIsNone(self.row(worker)['invoked_executor'])

    def check_test_history_write_failure_preserves_primary_exception_and_exportable_qc(self):
        failure = ValueError('Primary execution failure')
        with patch('seedvision.persistence.method_qc_history.retain_method_qc',
                   side_effect=OSError('History destination unavailable')):
            worker = self.run_failure(failure)
        self.assertTrue(worker.qualities)
        self.assertIn('History destination unavailable', failure.__notes__[0])
        self.assertTrue(worker.retention_errors)

    def check_test_unknown_source_history_and_exact_ack_reopen_without_source_probes(self):
        from seedvision.export.method_qc import acknowledge_method_qc, load_method_qc_acknowledgment
        from seedvision.persistence.method_qc_history import retain_method_qc
        missing = self.root / 'never-present.png'
        worker = self.run_failure(FileNotFoundError('No captured source'),
            request=replace(self.request, source_path=missing, source_stamp=None))
        quality = worker.qualities[-1]
        path = self.root / 'ack.json'
        acknowledge_method_qc(quality, path, 'fixture-account', reviewed_at_utc='2026-10-13T12:00:00+00:00')
        acknowledgment = load_method_qc_acknowledgment(path, quality)
        retain_method_qc(self.root, quality, acknowledgment=acknowledgment)
        with patch('seedvision.export.public_reference_attempt._observe_inputs',
                   side_effect=AssertionError('Historical loading cannot inspect sources')):
            history = load_project_method_qc(self.root)
        self.assertFalse(history.diagnostics)
        self.assertEqual(len(history.acknowledgments), 1)
        self.assertEqual(MethodQC(history.records[0].decode()).revision, quality.revision)

    def check_test_failed_delivery_exposes_qc_and_performs_no_file_observation(self):
        worker = self.run_failure(ValueError('Public failure'))
        dialog = ui.PublicReferenceCaptureDialog(self.recipe, project_root=self.root)
        self.addCleanup(dialog.close)
        dialog._generation = 1
        dialog._worker = worker
        dialog._cancel = worker.cancel
        with patch.object(ui, 'stamp', side_effect=AssertionError('No Qt source probe')), patch(
                'seedvision.export.public_reference_attempt._observe_inputs',
                side_effect=AssertionError('No Qt byte observation')):
            dialog._completed(*worker.delivered[0], worker=worker)
        self.assertIsNone(dialog._reply)
        self.assertTrue(dialog.qc_button.isEnabled())
        self.assertEqual(dialog._quality, worker.qualities[-1])
        with patch('seedvision.ui.method_qc_dialog.MethodQCDialog') as shared:
            dialog._show_quality()
        self.assertEqual(shared.call_args.args[0], worker.qualities[-1])
        self.assertEqual(shared.call_args.kwargs['project_root'], self.root)

    def check_test_stale_delivery_cannot_clear_replacement_worker_or_install_output(self):
        old = self.run_failure(ValueError('Earlier recorded failure'))
        replacement = self.worker(request=replace(self.request, generation=2))
        dialog = ui.PublicReferenceCaptureDialog(self.recipe, project_root=self.root)
        dialog._generation = 2
        dialog._worker = replacement
        dialog._cancel = replacement.cancel
        sentinel = object()
        dialog._reply = sentinel
        dialog._completed(*old.delivered[0], worker=old)
        self.assertIs(dialog._worker, replacement)
        self.assertIs(dialog._reply, sentinel)
        self.assertFalse(replacement.cancel.is_set())
        dialog._worker = None
        dialog.close()

    def test_validation_and_recovery(self):
        """8 related PublicReferenceAttemptBoundaryTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_failed_wrapper_retains_original_error_and_exact_request_bytes)
        self.scenario(self.check_test_missing_source_keeps_unknown_identity_and_actual_input_observation)
        self.scenario(self.check_test_runner_construction_failure_does_not_claim_wrapper_invocation)
        self.scenario(self.check_test_both_actual_cancellation_types_remain_cancellation)
        self.scenario(self.check_test_stop_before_invocation_retains_request_without_calling_runner)
        self.scenario(self.check_test_history_write_failure_preserves_primary_exception_and_exportable_qc)
        self.scenario(self.check_test_failed_delivery_exposes_qc_and_performs_no_file_observation)
        self.scenario(self.check_test_stale_delivery_cannot_clear_replacement_worker_or_install_output)

    def test_workflow_lifecycle(self):
        """3 related PublicReferenceAttemptBoundaryTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_supplied_expected_source_is_not_overwritten_by_changed_observation)
        self.scenario(self.check_test_nullable_source_version_cannot_relabel_known_or_complete_capture)
        self.scenario(self.check_test_unknown_source_history_and_exact_ack_reopen_without_source_probes)


class PublicReferenceCompletedAttemptTests(CatalogueTestCase):
    """Use one genuine public capture; delivery faults do not change execution."""
    @classmethod
    def setUpClass(cls):
        import cv2
        import numpy as np
        from tests.test_reference_preprocessing import HAS_REFERENCE_ENV, REFERENCE_PYTHON, _selected_recipe, _gray_fixture
        if not HAS_REFERENCE_ENV:
            raise unittest.SkipTest('Pinned public PlantCV environment required')
        from seedvision.benchmarking.reference_runner import PlantCVReferenceRunner
        from seedvision.benchmarking.reference_capture_session import capture_reference_path
        cls.app = QApplication.instance() or QApplication([])
        cls.temporary = tempfile.TemporaryDirectory(prefix='public-reference-completed-')
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.source = cls.root / 'selected.png'
        gray = _gray_fixture((48, 72))
        image = np.stack((np.minimum(gray, 173), np.maximum(gray, 61), gray), axis=-1).astype(np.uint8)
        success, encoded = cv2.imencode('.png', image)
        if not success:
            raise AssertionError('Fixture encoding failed')
        cls.source_bytes = encoded.tobytes()
        cls.source.write_bytes(cls.source_bytes)
        cls.recipe = _selected_recipe('plantcv_seed_binary')
        cls.session = capture_reference_path(cls.recipe, cls.source, cls.root / 'capture',
            PlantCVReferenceRunner(REFERENCE_PYTHON))
        if cls.session.verified is None:
            raise AssertionError('Genuine completed fixture was not verified')

    def setUp(self):
        self.source.write_bytes(self.source_bytes)
        self.request = ui.CaptureRequest(1, self.recipe, self.source, self.root / 'capture',
            stamp(self.source), history_root=self.root)

    def worker(self):
        worker = ui._CaptureWorker(self.request, threading.Event(), lambda: object())
        worker.delivered = []
        worker.signals.completed.connect(lambda *args: worker.delivered.append(args))
        return worker

    def run_completed(self, worker):
        with patch.object(ui, 'capture_reference_path', return_value=self.session), patch.object(
                ui, '_display_artifacts', return_value=()):
            worker.run()
        self.assertIsNone(worker.delivered[0][2])






    def test_actual_empty_return_is_not_recast_as_failed_execution(self):
        import cv2
        import numpy as np
        from seedvision.benchmarking.reference_capture_session import capture_reference_path
        from seedvision.benchmarking.reference_runner import PlantCVReferenceRunner
        from tests.test_reference_preprocessing import REFERENCE_PYTHON
        source = self.root / 'empty.png'
        success, encoded = cv2.imencode('.png', np.zeros((48, 72, 3), np.uint8))
        self.assertTrue(success)
        source.write_bytes(encoded.tobytes())
        session = capture_reference_path(self.recipe, source, self.root / 'empty-capture',
            PlantCVReferenceRunner(REFERENCE_PYTHON))
        self.assertEqual(session.outcome.status, 'empty')
        worker = ui._CaptureWorker(replace(self.request, source_path=source, source_stamp=stamp(source)),
            threading.Event(), lambda: object())
        worker.delivered = []
        worker.signals.completed.connect(lambda *args: worker.delivered.append(args))
        with patch.object(ui, 'capture_reference_path', return_value=session):
            worker.run()
        self.assertIsNone(worker.delivered[0][2])
        row = worker.execution_quality.payload['identity']['reference_capture']
        self.assertEqual(row['state'], 'empty')
        self.assertEqual(row['record']['source_sha256'], sha256(source.read_bytes()).hexdigest())
        self.assertIsNone(row['manifest_sha256'])


if __name__ == '__main__':
    unittest.main()
