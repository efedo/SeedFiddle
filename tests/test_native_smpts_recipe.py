from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from seedvision.benchmarking import native_smpts_recipe as factory
from seedvision.benchmarking import native_recipe as runner
from seedvision.benchmarking.checkpoints import CoordinateFrame
from seedvision.benchmarking.recipe_catalogue import build_recipe_catalogue
from seedvision.benchmarking.recipes import RecipeSpec, RecipeProvenanceMode
from seedvision.export.native_results import export_native_recipe_result, load_native_recipe_result
from seedvision.export.project_native_graphs import export_project_native_graphs
from seedvision.persistence import native_recipes as working
from seedvision.persistence.reference_regions import file_sha256
from seedvision.segmentation.baseline import PipelineAnalysisCache
from tests.test_smpts_paper import contact_fixture


def parameter(recipe, stage_id, name, value):
    return replace(recipe, stages=tuple(replace(stage, parameters=tuple(
        replace(p, value=value) if p.parameter_id == name else p for p in stage.parameters))
        if stage.stage_id == stage_id else stage for stage in recipe.stages))


def selected_recipe():
    return parameter(factory.build_native_smpts_recipe(), factory.CONTACTS, 'contact_box_area', 20)


class PaperRecipeContractTests(unittest.TestCase):
    def test_additive_canonical_paper_graph_and_working_selection_roundtrip(self):
        history = build_recipe_catalogue().to_json()
        recipe = selected_recipe()
        runner._validate_recipe(recipe)
        self.assertEqual(recipe.provenance.mode, RecipeProvenanceMode.PAPER_RECONSTRUCTION)
        self.assertEqual(RecipeSpec.from_json(recipe.to_json()), recipe)
        self.assertEqual(len(recipe.stages), 10)
        self.assertTrue(all(p.affects_calculation for s in recipe.stages for p in s.parameters))
        payload = working.native_recipes_to_payload((recipe,))
        self.assertEqual(working.native_recipes_from_payload(payload), (recipe,))
        for bad in (build_recipe_catalogue().recipe('smpts_paper_reconstruction'),
                replace(recipe, provenance=replace(recipe.provenance, mode=RecipeProvenanceMode.NATIVE_COMPATIBLE)),
                replace(recipe, stages=recipe.stages[:-1], layout=recipe.layout[:-1],
                    dependencies=tuple(e for e in recipe.dependencies if e.target_stage != factory.CROPS))):
            with self.assertRaises(ValueError):
                working.validate_native_recipes((bad,))
        self.assertEqual(build_recipe_catalogue().to_json(), history)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class PaperRecipeExecutionTests(unittest.TestCase):
    def setUp(self):
        mask, self.rgb = contact_fixture()
        self.gray = np.where(mask, 220, 10).astype(np.uint8)
        self.gray_gpu, self.rgb_gpu = (torch.as_tensor(v, device='cuda') for v in (self.gray, self.rgb))
        self.recipe, self.cache = selected_recipe(), PipelineAnalysisCache()
        self.frame = CoordinateFrame('source.decoded', self.gray.shape, 'top-left', 'x-right,y-down', 'pixel')

    def run_recipe(self, recipe=None, **kwargs):
        return runner.run_native_recipe(recipe or self.recipe, self.gray_gpu, self.rgb_gpu,
            source_sha256='a'*64, coordinate_frame=self.frame, preparation={'gray': 'explicit gray', 'rgb': 'explicit RGB'},
            cache=self.cache, **kwargs)

    def test_executable_graph_exact_outputs_crop_lineage_and_node_local_reuse(self):
        actual = self.run_recipe()
        self.assertEqual(actual.status, 'executed', [(n.stage_id, n.status, n.reason) for n in actual.nodes])
        expected = factory.paper.reconstruct(self.gray_gpu, self.rgb_gpu, contact_box_area=20)
        crops = actual.node(factory.CROPS).outputs
        self.assertTrue(torch.equal(actual.labels, expected['final_labels']))
        self.assertEqual(actual.labels_coordinate_frame, self.frame)
        self.assertTrue(torch.equal(crops['packed_rgb'], expected['packed_rgb']))
        self.assertTrue(torch.equal(crops['packed_mask'], expected['packed_mask']))
        self.assertEqual(crops['crops'], expected['crops'])
        self.assertFalse(crops['metadata']['original_execution'])
        self.assertFalse(crops['metadata']['scientific_validation'])
        self.assertEqual(self.run_recipe().status, 'executed')
        self.assertEqual(self.cache.last_computed_nodes, ())
        changed = parameter(self.recipe, factory.CONTACTS, 'contact_box_area', 1_000_000)
        unselected = self.run_recipe(changed)
        self.assertEqual(unselected.status, 'executed')
        self.assertEqual(self.cache.last_computed_nodes,
            (factory.CONTACTS, factory.EROSION, factory.RELOCALIZE, factory.CROPS))
        self.assertEqual(unselected.node(factory.CONTACTS).outputs['selected_parent_ids'], ())
        self.assertTrue(torch.equal(unselected.node(factory.EROSION).outputs['mask'],
            actual.node(factory.INTERSECTION).outputs['mask']))
        erosion = parameter(changed, factory.EROSION, 'ksize', 1)
        self.run_recipe(erosion)
        self.assertEqual(self.cache.last_computed_nodes, (factory.EROSION, factory.RELOCALIZE, factory.CROPS))

    def test_resource_failure_clears_descendants_and_mutation_cancel_withhold_run(self):
        self.gray_gpu = torch.full((1100, 1000), 220, dtype=torch.uint8, device='cuda')
        self.rgb_gpu = torch.full((1100, 1000, 3), 220, dtype=torch.uint8, device='cuda')
        self.frame = CoordinateFrame('source.decoded', (1100, 1000), 'top-left', 'x-right,y-down', 'pixel')
        self.recipe = parameter(self.recipe, factory.MEDIAN, 'ksize', 3)
        self.assertEqual(self.run_recipe().status, 'executed')
        changed = parameter(self.recipe, factory.MEDIAN, 'ksize', 31)
        failed = self.run_recipe(changed)
        self.assertEqual(failed.node(factory.MEDIAN).status, 'unavailable')
        for stage in (factory.INTERSECTION, factory.COMPONENTS, factory.CONTACTS, factory.EROSION,
                factory.RELOCALIZE, factory.CROPS):
            self.assertEqual(failed.node(stage).status, 'not_run')
            self.assertFalse(failed.node(stage).outputs)
        with self.assertRaises(runner.NativeRecipeCancelled):
            self.run_recipe(cancel=lambda: True)
        def mutate():
            self.rgb_gpu[0, 0, 0] = 1
            return False
        with self.assertRaises(runner.NativeRecipeError):
            self.run_recipe(cancel=mutate)

    def test_export_and_saved_project_batch_preserve_paper_provenance(self):
        result = self.run_recipe()
        self.assertEqual(result.status, 'executed')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            export_native_recipe_result(root/'run', self.recipe, result)
            report, arrays = load_native_recipe_result(root/'run')
            self.assertEqual(report['recipe']['provenance']['mode'], 'paper_reconstruction')
            self.assertTrue(arrays)
            image = root/'seed.png'
            self.assertTrue(cv2.imwrite(str(image), self.rgb[:, :, ::-1]))
            ledger = export_project_native_graphs((), image, root/'batch',
                expected_source_sha256=file_sha256(image), native_recipes=(self.recipe,))
            rows = json.loads(ledger.read_bytes())['graphs']
            self.assertEqual([r['status'] for r in rows], ['executed'])
            self.assertEqual(rows[0]['recipe_identity'], self.recipe.identity.to_dict())
            self.assertEqual(rows[0]['historical_stage_runs'], [])

    def test_empty_run_export_keeps_zero_crops_as_success(self):
        self.gray_gpu = torch.zeros_like(self.gray_gpu)
        result = self.run_recipe()
        self.assertEqual(result.status, 'executed', [(n.stage_id, n.status, n.reason) for n in result.nodes])
        self.assertEqual(result.node(factory.CROPS).outputs['crops'], ())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)/'empty'
            export_native_recipe_result(root, self.recipe, result)
            report, _ = load_native_recipe_result(root)
            self.assertEqual(report['status'], 'executed')


class PaperRecipeQtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        from PySide6.QtWidgets import QApplication
        cls.application = QApplication.instance() or QApplication([])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_actual_async_crop_preview_controls_and_project_selection(self):
        from seedvision.ui.native_recipe_dialog import NativeRecipeDialog
        from seedvision.ui.reference_graph_dialog import ReferenceGraphDialog
        selected = []
        dialog = NativeRecipeDialog(selected_recipe(), keep_in_project=selected.append)
        mask, _ = contact_fixture()
        rgb = np.repeat(np.where(mask, 220, 10).astype(np.uint8)[..., None], 3, axis=2)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'seed.png'
            cv2.imwrite(str(path), rgb)
            dialog.show()
            try:
                dialog.set_source_path(path)
                dialog.start_run()
                deadline = time.monotonic()+30
                while dialog._worker is not None and time.monotonic() < deadline:
                    self.application.processEvents()
                    time.sleep(.01)
                self.assertIsNone(dialog._worker)
                self.assertIsNotNone(dialog.result, dialog.status_label.text())
                self.assertEqual(dialog.result.status, 'executed')
                dialog._select_stage(factory.CROPS)
                self.assertGreater(dialog.crop_selector.count(), 1)
                dialog.crop_selector.setCurrentIndex(1)
                self.assertFalse(dialog.preview.pixmap().isNull())
                self.assertIn('parent_id', dialog.facts.toPlainText())
                dialog._keep_recipe_in_project()
                self.assertEqual(len(selected), 1)
                self.assertEqual(selected[0].provenance.mode, RecipeProvenanceMode.PAPER_RECONSTRUCTION)
                dialog.inspector.parameter_changed.emit(factory.CONTACTS, 'contact_box_area', '1000000')
                self.assertIsNone(dialog.result)
                self.assertEqual(dialog.recipe.parameters_for_runner(factory.CONTACTS)['contact_box_area'], 1_000_000)
            finally:
                dialog.reject()
                while dialog._worker is not None:
                    self.application.processEvents()
                    time.sleep(.01)
                dialog.deleteLater()
        reference = ReferenceGraphDialog(native_recipes=(selected_recipe(),))
        try:
            with patch('seedvision.ui.native_recipe_dialog.NativeRecipeDialog.exec', return_value=0):
                reference._open_native_smpts_paper()
            self.assertEqual(len(reference.native_recipes), 1)
        finally:
            reference.reject()
            reference.deleteLater()


if __name__ == '__main__':
    unittest.main()
