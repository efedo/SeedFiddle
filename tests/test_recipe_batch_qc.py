"""Actual graph-ledger, interrupted publication and portable batch QC witnesses."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from seedvision.benchmarking import native_recipe as runner
from seedvision.benchmarking.native_correction_recipe import build_native_correction_recipe
from seedvision.export import method_qc as qc, recipe_attempt_qc as attempts, recipe_batch_qc as batches
from seedvision.export import project_native_graphs as graphs
from seedvision.persistence import (ProjectAnalysisStore, ProjectImageSpec,
    analysis_settings_profile_from_graph, method_qc_history as history, snapshot as snapshots)
from seedvision.pipeline import build_default_pipeline
from tests.test_native_recipe import fixture, selected_recipe
from tests.test_native_result_export import _attachment
from tests import test_project_correction_inputs as correction_fixtures


class _Fixture:
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root/'seeds.png'
        gray, _ = fixture()
        self.assertTrue(cv2.imwrite(str(self.path), np.repeat(gray[..., None], 3, axis=2)))
        self.source_sha = sha256(self.path.read_bytes()).hexdigest()
        self.recipe = runner.build_native_seed_recipe(selected_recipe())

    def export(self, name, attachments=None, **options):
        return graphs.export_project_native_graphs((_attachment(self.recipe),) if attachments is None else attachments,
            self.path, self.root/name, expected_source_sha256=self.source_sha, **options)

    def qualities(self, destination):
        captured = batches.load_project_native_graph_history(destination)
        self.assertFalse(captured.incomplete, captured.diagnostics)
        self.assertFalse(captured.diagnostics)
        return [qc.MethodQC(raw.decode('utf-8')) for raw in captured.records]

    def terminal(self, destination):
        qualities = self.qualities(destination)
        selected = [value for value in qualities if value.payload['identity']['scope'] == 'recipe_batch_attempt']
        self.assertEqual(len(selected), 1)
        return selected[0], qualities

    def cancel_after_export(self, destination, attachments=None, **options):
        stopped = threading.Event()
        original = graphs.export_native_recipe_result
        captured = []
        def export_then_stop(*args, **kwargs):
            report = original(*args, **kwargs)
            captured.append((report, (report.parent/qc.FILENAME).read_bytes()))
            stopped.set()
            return report
        with patch.object(graphs, 'export_native_recipe_result', side_effect=export_then_stop):
            with self.assertRaises(runner.NativeRecipeCancelled) as raised:
                self.export(destination, attachments, cancel=stopped.is_set, **options)
        return raised.exception, captured


class RecipeBatchQCTests(_Fixture, unittest.TestCase):




    def test_legacy_rows_and_forged_embedded_links_never_invent_recorded_qc(self):
        ledger = self.export('legacy', (_attachment(selected_recipe()),))
        original = json.loads(ledger.read_bytes())
        row = original['graphs'][0]
        for mutate in (
            lambda value:value['method_qc'].update(raw_sha256='0'*64),
            lambda value:value['method_qc'].update(qc_revision='0'*64),
            lambda value:value['method_qc'].update(archive_path='../escaped/native-results.json'),
            lambda value:value.update(recipe_identity=self.recipe.identity.to_dict()),
            lambda value:value.update(status='executed'),
            lambda value:value.update(nodes=[{'stage_id':'invented output'}]),
        ):
            forged = deepcopy(original)
            mutate(forged['graphs'][0])
            path = self.root/'forged.json'
            path.write_text(json.dumps(forged), encoding='utf-8')
            with self.assertRaises(ValueError):
                batches.load_project_graph_qc(path, 0)
        legacy = deepcopy(original)
        legacy['graphs'][0].pop('method_qc')
        path = self.root/'legacy.json'
        path.write_text(json.dumps(legacy), encoding='utf-8')
        self.assertIsNone(batches.load_project_graph_qc(path, 0))
        for index in (-1, True, 1):
            with self.assertRaises(ValueError):
                batches.load_project_graph_qc(path, index)
        malformed = deepcopy(original)
        malformed['graphs'][0] = 1
        path.write_text(json.dumps(malformed), encoding='utf-8')
        with self.assertRaises(ValueError):
            batches.read_project_graph_qc(path)



@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for actual stored graph outcomes')
class ActualRecipeBatchQCTests(_Fixture, unittest.TestCase):

    def test_actual_selected_model_failure_records_binding_phase_without_native_archive(self):
        from tests.test_native_naive_bayes_recipe import _recipe_with_model
        from tests.test_plantcv_naive_bayes import _pdf
        model = self.root/'model.tsv'
        model.write_bytes(_pdf())
        recipe, _ = _recipe_with_model(model)
        model.write_bytes(b'Changed selected model bytes.')
        ledger = self.export('model-failure', (_attachment(recipe),))
        quality = batches.load_project_graph_qc(ledger, 0)
        row = quality.payload['identity']['attempt_record']
        self.assertEqual(row['state'], 'failed')
        self.assertEqual(row['invoked_executor'], attempts.NATIVE_PATH_EXECUTOR)
        self.assertFalse(row['result_returned'])
        observed = next(item for item in row['inputs_before'] if item['name'].startswith('model.'))
        self.assertEqual(observed['observation']['sha256'],sha256(model.read_bytes()).hexdigest())
        self.assertIsNone(quality.payload['identity']['native_artifact'])
        self.assertEqual(list(ledger.parent.iterdir()), [ledger])
