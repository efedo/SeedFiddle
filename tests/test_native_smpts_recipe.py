
from tests.catalogue_fixtures import CatalogueTestCase
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
    pass


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class PaperRecipeExecutionTests(CatalogueTestCase):
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



class PaperRecipeQtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        from PySide6.QtWidgets import QApplication
        cls.application = QApplication.instance() or QApplication([])



if __name__ == '__main__':
    unittest.main()
