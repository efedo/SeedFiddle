from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

import cv2
import numpy as np

from seedvision.benchmarking.checkpoint_bundle import (
    CheckpointBundleLimits, export_checkpoint_bundle, verify_checkpoint_bundle,
)
from seedvision.benchmarking.checkpoints import (
    CheckpointCancelled, ExecutionRecord, ExecutionStatus, InvalidCheckpoint,
    RasterRole, load_checkpoint, publish_checkpoint,
)
from seedvision.persistence.analysis_settings import analysis_settings_profile_from_graph
from seedvision.persistence.project_analysis import ProjectAnalysisStore, ProjectImageSpec
from seedvision.persistence.snapshot import create_snapshot, restore_snapshot
from seedvision.pipeline import build_default_pipeline
from tests.test_benchmark_checkpoints import _identity, _measurement
from tests.test_reference_graph_persistence import _attachment


class CheckpointBundleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.store = self.root/'checkpoints'

    def tearDown(self):
        self.temporary.cleanup()

    def publish(self, upstream=None, *, failed=False):
        return publish_checkpoint(self.store, _identity(upstream=upstream),
            {} if failed else {'mask': np.eye(7, 9, dtype=np.bool_)},
            execution=ExecutionRecord(ExecutionStatus.FAILED, reason='Deliberate test failure.')
                if failed else ExecutionRecord(ExecutionStatus.AVAILABLE, elapsed_ms=1),
            measurement=_measurement(eligible=False),
            raster_roles={} if failed else {'mask': RasterRole('mask', np.dtype(np.bool_).str)},
            allow_cpu_transfer=True)

    def test_real_closure_round_trip_preserves_raw_bytes_and_failure_eligibility(self):
        first = self.publish()
        second = self.publish(first.checkpoint_sha256)
        failed = self.publish(second.checkpoint_sha256, failed=True)
        bundle = export_checkpoint_bundle(self.store, [failed.checkpoint_sha256], self.root/'bundle')
        verified = verify_checkpoint_bundle(bundle.path)
        self.assertEqual(verified.roots, (failed.checkpoint_sha256,))
        self.assertEqual(set(verified.checkpoints), {first.checkpoint_sha256, second.checkpoint_sha256,
            failed.checkpoint_sha256})
        for digest in verified.checkpoints:
            for name in ('manifest.json', 'arrays.npz'):
                self.assertEqual((self.store/f'{digest}.checkpoint'/name).read_bytes(),
                    (bundle.path/f'{digest}.checkpoint'/name).read_bytes())
        restored = load_checkpoint(bundle.path, failed.checkpoint_sha256)
        self.assertEqual(restored.execution.status, ExecutionStatus.FAILED)
        self.assertFalse(restored.measurement.eligible)
        self.assertEqual(dict(restored.arrays), {})

    def test_missing_dependency_and_small_budgets_never_publish(self):
        missing = self.publish('f'*64)
        target = self.root/'missing'
        with self.assertRaises(InvalidCheckpoint):
            export_checkpoint_bundle(self.store, [missing.checkpoint_sha256], target)
        self.assertFalse(target.exists())
        first = self.publish()
        second = self.publish(first.checkpoint_sha256)
        for name, limits in [('count', CheckpointBundleLimits(max_checkpoints=1)),
                             ('bytes', CheckpointBundleLimits(max_encoded_bytes=1))]:
            target = self.root/name
            with self.assertRaises(InvalidCheckpoint):
                export_checkpoint_bundle(self.store, [second.checkpoint_sha256], target, limits=limits)
            self.assertFalse(target.exists())

    def test_cancel_during_copy_never_publishes_or_changes_source(self):
        reference = self.publish()
        original = (reference.path/'arrays.npz').read_bytes()
        calls = 0
        def cancel():
            nonlocal calls
            calls += 1
            return calls >= 5
        with self.assertRaises(CheckpointCancelled):
            export_checkpoint_bundle(self.store, [reference.checkpoint_sha256], self.root/'cancelled', cancel=cancel)
        self.assertFalse((self.root/'cancelled').exists())
        self.assertEqual((reference.path/'arrays.npz').read_bytes(), original)

    def test_tampered_arrays_and_false_upstream_records_are_rejected(self):
        first = self.publish()
        bundle = export_checkpoint_bundle(self.store, [first.checkpoint_sha256], self.root/'bundle')
        index = bundle.path/'checkpoint-bundle.json'
        payload = json.loads(index.read_text(encoding='utf-8'))
        payload['checkpoints'][first.checkpoint_sha256]['upstream'] = 'f'*64
        index.write_text(json.dumps(payload), encoding='utf-8')
        with self.assertRaisesRegex(InvalidCheckpoint, 'upstream'):
            verify_checkpoint_bundle(bundle.path)
        payload['checkpoints'][first.checkpoint_sha256]['upstream'] = None
        index.write_text(json.dumps(payload), encoding='utf-8')
        arrays = bundle.path/f'{first.checkpoint_sha256}.checkpoint'/'arrays.npz'
        arrays.write_bytes(arrays.read_bytes()+b'changed')
        with self.assertRaisesRegex(InvalidCheckpoint, 'raw-file'):
            verify_checkpoint_bundle(bundle.path)

    def test_source_changed_after_copy_is_not_published(self):
        reference = self.publish()
        from seedvision.benchmarking import checkpoint_bundle as module
        original_hash = module._raw_hash
        def changed(path, cancel=None, **kwargs):
            value = original_hash(path, cancel, **kwargs)
            if path == reference.path/'arrays.npz':
                changed.calls += 1
                if changed.calls == 2:
                    return 'f'*64
            return value
        changed.calls = 0
        with patch.object(module, '_raw_hash', side_effect=changed):
            with self.assertRaisesRegex(InvalidCheckpoint, 'changed while bundling'):
                export_checkpoint_bundle(self.store, [reference.checkpoint_sha256], self.root/'changed')
        self.assertFalse((self.root/'changed').exists())

    def test_source_growth_after_preflight_stops_copy_before_publication(self):
        reference = self.publish()
        from seedvision.benchmarking import checkpoint_bundle as module
        original_load = module.load_checkpoint
        def grow_after_load(store, digest, **kwargs):
            checkpoint = original_load(store, digest, **kwargs)
            if store == self.store.resolve():
                with (reference.path/'arrays.npz').open('ab') as stream:
                    stream.write(b'x' * (2 * 1024 * 1024))
            return checkpoint
        with patch.object(module, 'load_checkpoint', side_effect=grow_after_load):
            with self.assertRaisesRegex(InvalidCheckpoint, 'preflight bytes'):
                export_checkpoint_bundle(self.store, [reference.checkpoint_sha256], self.root/'growing')
        self.assertFalse((self.root/'growing').exists())

    def test_snapshot_opt_in_restores_pixels_and_keeps_metadata_unavailable(self):
        first = self.publish()
        second = self.publish(first.checkpoint_sha256)
        attachment = _attachment()
        attachment = replace(attachment, stage_runs=(replace(attachment.stage_runs[0],
            checkpoint_sha256=second.checkpoint_sha256,
            upstream_checkpoint_sha256=first.checkpoint_sha256),))
        image = self.root/'images'/'capture.png'
        image.parent.mkdir()
        self.assertTrue(cv2.imwrite(str(image), np.zeros((7, 9, 3), dtype=np.uint8)))
        store = ProjectAnalysisStore(self.root)
        document = store.capture(analysis_settings=analysis_settings_profile_from_graph(build_default_pipeline()),
            images=(ProjectImageSpec(image, (7, 9)),), reference_graphs=(attachment,))
        project = store.save(document, self.root/'project.seedfiddle-project.json')
        ordinary = create_snapshot(self.root, project, self.root/'snapshots')
        self.assertFalse((ordinary/'reference-checkpoints').exists())
        snapshot = create_snapshot(self.root, project, self.root/'snapshots', reference_checkpoint_store=self.store)
        restored_project = restore_snapshot(snapshot, self.root)
        restored = store.load(restored_project)
        self.assertEqual(restored.document.reference_graphs, (attachment,))
        self.assertEqual(restored.document.reference_graphs[0].checkpoint_data_status, 'unavailable')
        checkpoint = load_checkpoint(restored_project.parent/'reference-checkpoints', second.checkpoint_sha256)
        np.testing.assert_array_equal(checkpoint.arrays['mask'], np.eye(7, 9, dtype=np.bool_))
        self.assertFalse(checkpoint.measurement.eligible)
        manifest = snapshot/'reference-checkpoints'/f'{second.checkpoint_sha256}.checkpoint'/'manifest.json'
        manifest.write_bytes(manifest.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError, 'Snapshot verification failed'):
            restore_snapshot(snapshot, self.root)

    def test_qt_snapshot_action_passes_only_explicitly_selected_store(self):
        from seedvision.ui.results import ResultsController
        window = SimpleNamespace(_root=self.root, _current_project_path=self.root/'project.json',
            _species_library_service=SimpleNamespace(store=None),
            _background_work_is_active=Mock(return_value=False),
            _resolve_unapplied_project_drafts=Mock(return_value=True),
            _write_project=Mock(return_value=True), statusBar=Mock(return_value=Mock()))
        window._create_snapshot_impl = lambda **kwargs: ResultsController._create_snapshot_impl(window, **kwargs)
        with patch('seedvision.ui.results.QFileDialog.getExistingDirectory', side_effect=['snapshots', 'selected-store']), \
             patch('seedvision.persistence.snapshot.create_snapshot', return_value=self.root/'snapshot') as create, \
             patch('seedvision.ui.persistence_jobs.run_persistence', side_effect=lambda owner, callback, title: callback()):
            ResultsController._create_snapshot_with_reference_checkpoints(window)
        self.assertEqual(create.call_args.kwargs['reference_checkpoint_store'], 'selected-store')
        with patch('seedvision.ui.results.QFileDialog.getExistingDirectory', return_value='snapshots'), \
             patch('seedvision.persistence.snapshot.create_snapshot', return_value=self.root/'snapshot') as create, \
             patch('seedvision.ui.persistence_jobs.run_persistence', side_effect=lambda owner, callback, title: callback()):
            ResultsController._create_snapshot(window)
        self.assertIsNone(create.call_args.kwargs['reference_checkpoint_store'])
        with patch('seedvision.ui.results.QFileDialog.getExistingDirectory', side_effect=['snapshots', '']), \
             patch('seedvision.persistence.snapshot.create_snapshot') as create:
            ResultsController._create_snapshot_with_reference_checkpoints(window)
        create.assert_not_called()


if __name__ == '__main__':
    unittest.main()
