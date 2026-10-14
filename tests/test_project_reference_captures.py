"""Real public batch execution and reference-only publication/QC witnesses."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import shutil
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from seedvision.benchmarking.checkpoints import CheckpointCancelled
from seedvision.benchmarking.reference_capture_session import ReferenceCaptureSession, open_reference_capture
from seedvision.benchmarking.reference_runner import PlantCVReferenceRunner, ReferenceIdentityError, ReferenceExecutionError
from seedvision.benchmarking.recipe_catalogue import build_recipe_catalogue
from seedvision.export import method_qc as qc, project_reference_captures as captures
from seedvision.export.reference_capture_qc import qc_from_reference_session, PUBLIC_BACKEND, EXECUTOR
from seedvision.persistence import ProjectAnalysisStore, ProjectImageSpec, analysis_settings_profile_from_graph
from seedvision.persistence import method_qc_history as history, snapshot as snapshots
from seedvision.pipeline import build_default_pipeline
from tests.test_native_result_export import _attachment
from tests.test_reference_preprocessing import REFERENCE_PYTHON, HAS_REFERENCE_ENV, _selected_recipe, _gray_fixture


@unittest.skipUnless(HAS_REFERENCE_ENV, 'Pinned public PlantCV environment required')
class ProjectReferenceCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from seedvision.export.project_runner import run_project
        from tests.test_seed_crop_artifacts import _analysis
        cls.temporary = tempfile.TemporaryDirectory(prefix='public-project-capture-')
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.workspace = cls.root / 'workspace'
        cls.workspace.mkdir()
        cls.source = cls.workspace / 'seeds.png'
        gray = _gray_fixture((48, 72))
        image = np.stack((np.minimum(gray, 173), np.maximum(gray, 61), gray), axis=-1).astype(np.uint8)
        assert cv2.imwrite(str(cls.source), image)
        cls.source_bytes = cls.source.read_bytes()
        cls.source_sha = sha256(cls.source_bytes).hexdigest()
        cls.recipe = _selected_recipe('plantcv_seed_binary')
        cls.attachments = (_attachment(cls.recipe),)
        (cls.workspace / 'config').mkdir()
        shutil.copyfile(Path(__file__).parents[1] / 'config' / 'traits.json', cls.workspace / 'config' / 'traits.json')
        cls.master = ProjectAnalysisStore(cls.workspace).capture_and_save(
            analysis_settings=analysis_settings_profile_from_graph(build_default_pipeline()),
            images=(ProjectImageSpec(cls.source, (48, 72)),), reference_graphs=cls.attachments,
            destination=cls.workspace / 'public-test.seedfiddle-project.json')
        cls.operations, cls.observed_caches = [], []
        actual_call = PlantCVReferenceRunner.run
        actual_capture = captures.capture_reference_path
        from seedvision.export import project_runner as projects
        actual_native = projects.export_project_native_graphs

        def observe_call(runner, operation, *args, **kwargs):
            cls.operations.append(operation)
            return actual_call(runner, operation, *args, **kwargs)

        def observe_capture(*args, **kwargs):
            session = actual_capture(*args, **kwargs)
            cls.session = session
            return session

        def observe_native(*args, **kwargs):
            value = actual_native(*args, **kwargs)
            cls.observed_caches.append(kwargs['cache'])
            kwargs['cache'].values['test_release_sentinel'] = object()
            return value

        def factory():
            assert cls.observed_caches and not cls.observed_caches[-1].values
            return PlantCVReferenceRunner(REFERENCE_PYTHON)

        with patch.object(PlantCVReferenceRunner, 'run', observe_call), patch.object(
                captures, 'capture_reference_path', observe_capture), patch.object(
                projects, 'export_project_native_graphs', observe_native), patch.object(
                projects, 'analyze_path', side_effect=lambda path, **kwargs: _analysis(path)):
            cls.destination = run_project(cls.workspace, cls.master, cls.root / 'batch',
                reference_capture_runner_factory=factory)[0]
        cls.ledger = cls.destination / 'reference_captures' / captures.FILENAME
        cls.payload = json.loads(cls.ledger.read_bytes())
        if cls.payload['rows'][0]['status'] != 'executed':
            raise AssertionError(cls.payload['rows'][0]['reason'] or cls.payload['rows'][0]['method_qc'])
        cls.capture_store = cls.ledger.parent / 'capture-000'
        cls.manifest = cls.ledger.parent / cls.payload['rows'][0]['manifest']
        cls.quality = captures.read_project_reference_qc(cls.ledger)[0]

    def setUp(self):
        self.temporary_case = tempfile.TemporaryDirectory(prefix='public-project-boundary-')
        self.addCleanup(self.temporary_case.cleanup)
        self.case = Path(self.temporary_case.name)
        self.source = self.case / 'seeds.png'
        self.source.write_bytes(self.source_bytes)

    def export(self, name='output', attachments=None, **kwargs):
        return captures.export_project_reference_captures(self.attachments if attachments is None else attachments,
            self.source, self.case / name, expected_source_sha256=self.source_sha,
            runner_factory=lambda: PlantCVReferenceRunner(REFERENCE_PYTHON), **kwargs)

    def replay(self, recipe, source, store, runner, **kwargs):
        """Reuse real recorded artifacts for publication-boundary tests only."""
        self.assertEqual(recipe, self.recipe)
        shutil.copytree(self.capture_store, store)
        manifest = Path(store) / 'chains' / self.manifest.name
        opened = open_reference_capture(manifest)
        return ReferenceCaptureSession(recipe, replace(self.session.outcome, manifest_path=manifest), opened.verified)

    def qualities(self, error):
        self.assertTrue(error.reference_qualities)
        self.assertFalse(list(self.case.glob('.reference-captures-*')))
        return error.reference_qualities

    def test_actual_saved_project_invokes_eight_public_calls_and_releases_native_cache(self):
        self.assertEqual(len(self.operations), 8)
        self.assertEqual(self.operations[-3:], ['segmentation.watershed.capture', 'analysis.size', 'analysis.color'])
        self.assertTrue((self.destination / 'results.json').is_file())
        native = json.loads((self.destination / 'native_graphs' / 'native-graphs.json').read_bytes())
        self.assertEqual(native['graphs'][0]['status'], 'unavailable')
        self.assertEqual(self.payload['rows'][0]['status'], 'executed')
        self.assertEqual(self.payload['rows'][0]['recipe_identity'], self.recipe.identity.to_dict())
        self.assertEqual(ProjectAnalysisStore(self.workspace).load(self.master).document.reference_graphs, self.attachments)
        identity = self.quality.payload['identity']
        self.assertEqual(identity['scope'], 'whole_reference_capture')
        self.assertEqual(identity['reference_capture']['execution_backend'], PUBLIC_BACKEND)
        self.assertIsNone(identity['native_artifact'])
        self.assertIsNone(identity['node_outcomes'])
        self.assertIsNone(identity['scientific_validity'])
        self.assertFalse(self.observed_caches[-1].values)

    def test_independent_reopening_and_recorded_qc_do_not_probe_original_source(self):
        with patch('seedvision.benchmarking.native_image_inputs.snapshot', side_effect=AssertionError('No original decode')):
            opened = open_reference_capture(self.manifest)
            quality = captures.read_project_reference_qc(self.ledger)[0]
        self.assertEqual(qc_from_reference_session(opened), quality)
        self.assertEqual(len(opened.verified.checkpoints), 10)
        for name in ('gray', 'rgb'):
            self.assertIs(opened.verified.checkpoints['input.' + name].metadata['actual_public_reader'], False)
        path = self.case / 'qc.json'
        qc.export_method_qc(quality, path)
        self.source.unlink()
        self.assertEqual(qc.load_method_qc(path), quality)

    def test_desktop_opens_actual_batch_capture_and_reviews_the_same_qc(self):
        from PySide6.QtWidgets import QApplication
        from seedvision.ui.public_reference_capture_dialog import PublicReferenceCaptureDialog
        from seedvision.ui.method_qc_dialog import MethodQCDialog
        app = QApplication.instance() or QApplication([])
        dialog = PublicReferenceCaptureDialog(self.recipe)
        self.addCleanup(dialog.close)
        dialog.open_saved(self.manifest)
        deadline = time.monotonic() + 10
        while dialog._worker is not None and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.005)
        self.assertIsNone(dialog._worker)
        self.assertIsNotNone(dialog._reply, dialog.status_label.text())
        self.assertTrue(dialog.qc_button.isEnabled())
        observed = []
        with patch.object(MethodQCDialog, 'exec', lambda view: observed.append(view.quality)):
            dialog.qc_button.click()
        self.assertEqual(observed, [self.quality])
        view = MethodQCDialog(self.quality)
        self.addCleanup(view.close)
        self.assertIn('verified complete manifest', view.summary.text())
        dialog._invalidate()
        self.assertIsNone(dialog._reply)
        self.assertEqual(dialog.preview.text(), 'No current verified capture')
        self.assertEqual(dialog._quality, self.quality)
        self.assertTrue(dialog.qc_button.isEnabled())
        with patch.object(MethodQCDialog, 'exec', lambda view: observed.append(view.quality)):
            dialog.qc_button.click()
        self.assertEqual(observed, [self.quality, self.quality])

    def test_independent_desktop_and_batch_producers_agree_on_analytical_outputs(self):
        from PySide6.QtWidgets import QApplication
        from seedvision.benchmarking.checkpoints import _thaw_json
        from seedvision.ui.public_reference_capture_dialog import PublicReferenceCaptureDialog
        app = QApplication.instance() or QApplication([])
        dialog = PublicReferenceCaptureDialog(self.recipe, source_path=self.source,
            store=self.case / 'desktop-capture', runner_factory=lambda: PlantCVReferenceRunner(REFERENCE_PYTHON))
        self.addCleanup(dialog.close)
        dialog.start_capture()
        deadline = time.monotonic() + 120
        while dialog._worker is not None and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.005)
        self.assertIsNone(dialog._worker, 'Actual desktop public capture did not drain.')
        self.assertIsNotNone(dialog._reply, dialog.status_label.text())
        desktop = dialog._reply.session
        self.assertIsNotNone(desktop.verified, dialog.status_label.text())
        batch = open_reference_capture(self.manifest)
        self.assertEqual(desktop.recipe.identity.analytical_dict(), batch.recipe.identity.analytical_dict())
        left, right = _thaw_json(desktop.verified.record), _thaw_json(batch.verified.record)
        for key in ('source_sha256', 'coordinate_frame', 'status', 'complete', 'label_facts',
                    'runner_request_count_after_preprocessing', 'verified_public_result_count_after_preprocessing'):
            self.assertEqual(left[key], right[key], key)
        self.assertEqual([(row['operation'], row['status'], row['run_record']['reason'])
                          for row in left['preprocessing']],
                         [(row['operation'], row['status'], row['run_record']['reason'])
                          for row in right['preprocessing']])
        self.assertEqual([(row['operation'], row['status'], row['reason']) for row in left['source_calls']],
                         [(row['operation'], row['status'], row['reason']) for row in right['source_calls']])
        self.assertEqual(set(desktop.verified.checkpoints), set(batch.verified.checkpoints))
        for key, actual in desktop.verified.checkpoints.items():
            expected = batch.verified.checkpoints[key]
            actual_identity, expected_identity = actual.identity.to_dict(), expected.identity.to_dict()
            # Upstream archive revisions include actual call IDs and timings.
            actual_identity.pop('upstream_checkpoint_sha256')
            expected_identity.pop('upstream_checkpoint_sha256')
            self.assertEqual(actual_identity, expected_identity, key)
            self.assertEqual(actual.measurement, expected.measurement, key)
            self.assertEqual(actual.raster_roles, expected.raster_roles, key)
            self.assertEqual(set(actual.arrays), set(expected.arrays), key)
            for name, array in actual.arrays.items():
                self.assertEqual((array.shape, array.dtype), (expected.arrays[name].shape, expected.arrays[name].dtype))
                # The original public call shuffles its debug palette using ambient RNG.
                if name not in {'palette', 'diagnostic', 'diagnostic_bgr'}:
                    self.assertEqual(sha256(array.tobytes()).hexdigest(),
                                     sha256(expected.arrays[name].tobytes()).hexdigest(), (key, name))
            for name in ('requested_parameters', 'parameters_passed', 'runner_identity',
                         'input_array_sha256', 'observations', 'channel_semantics'):
                self.assertEqual(_thaw_json(actual.metadata.get(name)),
                                 _thaw_json(expected.metadata.get(name)), (key, name))
        desktop_qc = qc_from_reference_session(desktop)
        self.assertNotEqual(desktop_qc.revision, self.quality.revision)
        for name, quality in (('desktop', desktop_qc), ('batch', self.quality)):
            capture = quality.payload['identity']['reference_capture']
            self.assertEqual(capture['execution_backend'], PUBLIC_BACKEND)
            self.assertEqual(capture['verification'], 'verified_complete_manifest')
            path = self.case / (name + '-ack.json')
            qc.acknowledge_method_qc(quality, path, 'reference-reviewer')
            self.assertEqual(qc.load_method_qc_acknowledgment(path, quality)['qc_revision'], quality.revision)
            other = self.quality if name == 'desktop' else desktop_qc
            with self.assertRaises(ValueError):
                qc.load_method_qc_acknowledgment(path, other)

    def test_unselected_and_unsupported_originals_never_request_a_runner(self):
        catalogue = build_recipe_catalogue()
        attachments = (_attachment(catalogue.recipe('plantcv_seed_binary')),
                       _attachment(catalogue.recipe('plantcv_color_card_correct_color')))
        with patch.object(captures, 'capture_reference_path', side_effect=AssertionError('No invocation')):
            ledger = self.export(attachments=attachments)
        rows = json.loads(ledger.read_bytes())['rows']
        self.assertEqual([row['status'] for row in rows], ['unavailable', 'unavailable'])
        for quality in captures.read_project_reference_qc(ledger):
            capture = quality.payload['identity']['reference_capture']
            self.assertIsNone(capture['invoked_executor'])
            self.assertIsNone(capture['record'])
            self.assertIsNone(capture['manifest_sha256'])

    def test_nonoriginal_paper_and_native_routes_are_not_substituted(self):
        from seedvision.benchmarking.native_recipe import build_native_seed_recipe
        attachments = (_attachment(build_native_seed_recipe(self.recipe)),
                       _attachment(build_recipe_catalogue().recipe('smpts_paper_reconstruction')))
        with patch.object(captures, 'capture_reference_path', side_effect=AssertionError('No substitute')):
            self.assertIsNone(self.export(attachments=attachments))
        self.assertFalse((self.case / 'output').exists())

    def test_runner_factory_cannot_bypass_public_runner_type_and_verification(self):
        ledger = captures.export_project_reference_captures(self.attachments, self.source, self.case / 'wrong-runner',
            expected_source_sha256=self.source_sha, runner_factory=lambda: object())
        quality = captures.read_project_reference_qc(ledger)[0]
        capture = quality.payload['identity']['reference_capture']
        self.assertEqual(capture['state'], 'failed')
        self.assertIsNone(capture['invoked_executor'])
        self.assertIn('actual pinned public runner', capture['detail'])

    def test_dependency_failure_keeps_actual_partial_call_states(self):
        with patch.object(PlantCVReferenceRunner, 'run', side_effect=ReferenceIdentityError('Wrong environment')):
            ledger = self.export()
        quality = captures.read_project_reference_qc(ledger)[0]
        capture = quality.payload['identity']['reference_capture']
        self.assertEqual(capture['state'], 'partial')
        self.assertEqual(capture['invoked_executor'], EXECUTOR)
        self.assertEqual(capture['record']['preprocessing'][0]['status'], 'unavailable')
        self.assertEqual([call['status'] for call in capture['record']['source_calls']], ['not_run'] * 3)
        self.assertIsNone(capture['manifest_sha256'])

    def test_actual_empty_public_outcome_remains_empty_without_complete_manifest(self):
        self.assertTrue(cv2.imwrite(str(self.source), np.zeros((48, 72, 3), np.uint8)))
        ledger = captures.export_project_reference_captures(self.attachments, self.source, self.case / 'empty',
            expected_source_sha256=sha256(self.source.read_bytes()).hexdigest(),
            runner_factory=lambda: PlantCVReferenceRunner(REFERENCE_PYTHON))
        capture = captures.read_project_reference_qc(ledger)[0].payload['identity']['reference_capture']
        self.assertEqual(capture['state'], 'empty')
        self.assertEqual(capture['record']['source_calls'][0]['status'], 'empty')
        self.assertIsNone(capture['manifest_sha256'])

    def test_changed_source_rejects_admission_and_preserves_prior_destination(self):
        destination = self.case / 'prior'
        destination.mkdir()
        sentinel = destination / 'prior.bin'
        sentinel.write_bytes(b'Prior output')
        with self.assertRaises(FileExistsError) as prior:
            self.export('prior')
        self.assertEqual(self.qualities(prior.exception)[0].payload['identity']['reference_capture']['state'], 'rejected')
        self.assertEqual(sentinel.read_bytes(), b'Prior output')
        self.source.write_bytes(b'Changed source')
        with patch.object(captures, 'capture_reference_path', side_effect=AssertionError('No execution')):
            with self.assertRaises(ValueError) as changed:
                self.export('changed')
        self.assertEqual(self.qualities(changed.exception)[0].payload['identity']['reference_capture']['phase'], 'admission')

    def test_source_change_after_complete_capture_blocks_publication_but_preserves_recorded_success(self):
        def completed_then_changed(*args, **kwargs):
            session = self.replay(*args, **kwargs)
            self.source.write_bytes(b'Changed after capture')
            return session
        with patch.object(captures, 'capture_reference_path', completed_then_changed), self.assertRaises(ValueError) as raised:
            self.export()
        qualities = self.qualities(raised.exception)
        self.assertEqual(qualities[0].payload['identity']['scope'], 'whole_reference_capture')
        self.assertEqual(qualities[-1].payload['identity']['reference_capture']['phase'], 'publication')
        self.assertFalse((self.case / 'output').exists())

    def test_cancel_before_during_and_after_completed_row_preserves_distinct_attempts(self):
        with self.assertRaises(CheckpointCancelled) as before:
            self.export('before', cancel=lambda: True)
        self.assertIsNone(self.qualities(before.exception)[0].payload['identity']['reference_capture']['invoked_executor'])
        with patch.object(captures, 'capture_reference_path', side_effect=CheckpointCancelled('During actual boundary')):
            with self.assertRaises(CheckpointCancelled) as during:
                self.export('during')
        captured = self.qualities(during.exception)[0].payload['identity']['reference_capture']
        self.assertEqual(captured['invoked_executor'], EXECUTOR)
        self.assertIsNone(captured['record'])
        stopped = threading.Event()
        actual = qc.export_method_qc
        def saved_then_stop(*args, **kwargs):
            result = actual(*args, **kwargs)
            stopped.set()
            return result
        with patch.object(captures, 'capture_reference_path', self.replay), patch.object(qc, 'export_method_qc', saved_then_stop):
            with self.assertRaises(CheckpointCancelled) as after:
                self.export('after', cancel=stopped)
        qualities = self.qualities(after.exception)
        self.assertEqual([q.payload['identity']['scope'] for q in qualities],
                         ['whole_reference_capture', 'reference_capture_attempt'])
        self.assertEqual(qualities[-1].payload['identity']['reference_capture']['phase'], 'publication')

    def test_corrupt_capture_fails_independent_verification_without_completed_claim(self):
        def corrupt(*args, **kwargs):
            session = self.replay(*args, **kwargs)
            session.outcome.manifest_path.write_bytes(b'Corrupt manifest')
            return session
        with patch.object(captures, 'capture_reference_path', corrupt):
            ledger = self.export()
        quality = captures.read_project_reference_qc(ledger)[0]
        capture = quality.payload['identity']['reference_capture']
        self.assertEqual((capture['state'], capture['phase']), ('failed', 'verification'))
        self.assertIsNone(capture['manifest_sha256'])

    def test_publication_failure_preserves_competing_destination_and_completed_qc(self):
        actual_rename = captures.os.rename
        def fail_publish(source, destination):
            if not Path(source).name.startswith('.reference-captures-'):
                return actual_rename(source, destination)
            Path(destination).mkdir()
            (Path(destination) / 'prior.bin').write_bytes(b'Competing prior output')
            raise OSError('Publication failed')
        with patch.object(captures, 'capture_reference_path', self.replay), patch.object(captures.os, 'rename', fail_publish):
            with self.assertRaises(OSError) as raised:
                self.export()
        qualities = self.qualities(raised.exception)
        self.assertEqual(qualities[0], self.quality)
        self.assertEqual((self.case / 'output' / 'prior.bin').read_bytes(), b'Competing prior output')
        self.assertEqual(qualities[-1].payload['identity']['reference_capture']['phase'], 'publication')

    def test_actual_partial_public_chain_preserves_each_observed_call_state(self):
        actual = PlantCVReferenceRunner.run
        def fail_size(runner, operation, *args, **kwargs):
            if operation == 'analysis.size':
                raise ReferenceExecutionError('Controlled public size request failure')
            return actual(runner, operation, *args, **kwargs)
        with patch.object(PlantCVReferenceRunner, 'run', fail_size):
            ledger = self.export()
        capture = captures.read_project_reference_qc(ledger)[0].payload['identity']['reference_capture']
        self.assertEqual(capture['state'], 'partial')
        self.assertEqual([call['status'] for call in capture['record']['source_calls']],
                         ['executed', 'failed', 'executed'])
        self.assertEqual(capture['record']['source_calls'][1]['reason'],
                         'ReferenceExecutionError: Controlled public size request failure')
        self.assertIsNone(capture['manifest_sha256'])

    def test_secondary_cleanup_and_retention_failure_preserve_original_exception(self):
        original = OSError('Original row QC publication failure')
        with patch.object(captures, 'capture_reference_path', self.replay), patch.object(
                qc, 'export_method_qc', side_effect=original), patch.object(
                captures, '_remove_owned_directory', side_effect=OSError('Cleanup failure')), patch.object(
                captures, 'retain_method_qc', side_effect=OSError('History failure')):
            with self.assertRaises(OSError) as raised:
                self.export()
        self.assertIs(raised.exception, original)
        self.assertEqual(original.reference_qualities[0], self.quality)
        self.assertTrue(any('Cleanup failure' in note for note in original.__notes__))
        self.assertTrue(any('History failure' in note for note in original.__notes__))
        self.assertFalse((self.case / 'output').exists())

    def test_forged_row_paths_manifest_types_and_qc_revision_reject(self):
        for mutate in (
            lambda p: p['rows'][0].update(manifest='../foreign.json'),
            lambda p: p['rows'][0].update(status='partial'),
            lambda p: p['rows'][0].update(manifest_sha256='f' * 64),
            lambda p: p.update(source_sha256='a' * 64),
            lambda p: p['rows'][0]['method_qc']['identity']['reference_capture']['record'].update(complete=1),
            lambda p: p['rows'][0]['method_qc'].update(qc_revision='b' * 64),
        ):
            payload = deepcopy(self.payload)
            mutate(payload)
            path = self.case / 'forged.json'
            path.write_text(json.dumps(payload), encoding='utf-8')
            with self.assertRaises(ValueError):
                captures.read_project_reference_qc(path)

    def test_malformed_nested_recorded_facts_are_normalized_at_the_shared_qc_boundary(self):
        for change in (
            lambda record: record['inputs']['gray'].update(identity={}),
            lambda record: record['source_calls'][0].update(status=[]),
            lambda record: record['preprocessing'][0].update(run_record=None),
        ):
            payload = self.quality.payload
            capture = payload['identity']['reference_capture']
            change(capture['record'])
            capture['manifest_sha256'] = sha256(qc._bytes(capture['record'])).hexdigest()
            payload['qc_revision'] = sha256(qc._bytes(payload['identity'])).hexdigest()
            with self.assertRaises(ValueError):
                qc.MethodQC(qc._bytes(payload).decode('utf-8'))

    def test_acknowledgment_and_snapshot_transport_bind_exact_reference_qc(self):
        workspace = self.case / 'workspace'
        workspace.mkdir()
        source = workspace / 'seeds.png'
        source.write_bytes(self.source_bytes)
        master = ProjectAnalysisStore(workspace).capture_and_save(
            analysis_settings=analysis_settings_profile_from_graph(build_default_pipeline()),
            images=(ProjectImageSpec(source, (48, 72)),), reference_graphs=self.attachments,
            destination=workspace / 'snapshot-test.seedfiddle-project.json')
        ack_path = self.case / 'ack.json'
        qc.acknowledge_method_qc(self.quality, ack_path, 'reference-reviewer')
        ack = qc.load_method_qc_acknowledgment(ack_path, self.quality)
        history.retain_method_qc(workspace, self.quality, project_path=master, acknowledgment=ack)
        package = snapshots.create_snapshot(workspace, master, self.case / 'snapshots')
        source.unlink()
        restored = self.case / 'restored'
        restored_master = snapshots.restore_snapshot(package, restored)
        captured = history.load_project_method_qc(restored, restored_master)
        self.assertFalse(captured.incomplete, captured.diagnostics)
        self.assertFalse(captured.diagnostics)
        self.assertEqual([qc.MethodQC(raw.decode('utf-8')) for raw in captured.records], [self.quality])
        self.assertEqual(json.loads(captured.acknowledgments[0]), ack)
        with self.assertRaises(FileExistsError) as failed:
            self.export('restored')
        other = self.qualities(failed.exception)[-1]
        with self.assertRaises(ValueError):
            qc.load_method_qc_acknowledgment(ack_path, other)
