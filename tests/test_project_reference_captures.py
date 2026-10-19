"""Real public batch execution and reference-only publication/QC witnesses."""

from tests.functional_units import functional_units
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
from tests.catalogue_fixtures import CatalogueTestCase
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
@functional_units
class ProjectReferenceCaptureTests(CatalogueTestCase):
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


    def check_test_independent_reopening_and_recorded_qc_do_not_probe_original_source(self):
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






    def check_test_dependency_failure_keeps_actual_partial_call_states(self):
        with patch.object(PlantCVReferenceRunner, 'run', side_effect=ReferenceIdentityError('Wrong environment')):
            ledger = self.export()
        quality = captures.read_project_reference_qc(ledger)[0]
        capture = quality.payload['identity']['reference_capture']
        self.assertEqual(capture['state'], 'partial')
        self.assertEqual(capture['invoked_executor'], EXECUTOR)
        self.assertEqual(capture['record']['preprocessing'][0]['status'], 'unavailable')
        self.assertEqual([call['status'] for call in capture['record']['source_calls']], ['not_run'] * 3)
        self.assertIsNone(capture['manifest_sha256'])





    def check_test_corrupt_capture_fails_independent_verification_without_completed_claim(self):
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







    def test_workflow_lifecycle(self):
        """1 related ProjectReferenceCaptureTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_independent_reopening_and_recorded_qc_do_not_probe_original_source)

    def test_validation_and_recovery(self):
        """2 related ProjectReferenceCaptureTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_dependency_failure_keeps_actual_partial_call_states)
        self.scenario(self.check_test_corrupt_capture_fails_independent_verification_without_completed_claim)
