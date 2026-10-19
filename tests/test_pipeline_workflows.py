"""Editable workspace contracts and real CUDA node-local execution."""

from tests.catalogue_fixtures import CatalogueTestCase
from dataclasses import replace
import json
import unittest
import torch
from seedvision.pipeline.workflows import (PipelineWorkflow, create_workflow, workflow_graph,
    workflow_from_graph, workflow_catalogue)
from seedvision.benchmarking.recipes import DependencyContract, RecipeProvenanceMode
from seedvision.benchmarking.native_recipe import run_native_recipe, NativeRecipeError
from seedvision.segmentation.baseline import PipelineAnalysisCache
from tests.test_native_recipe import selected_recipe, fixture, FRAME, SOURCE, PREPARATION
from seedvision.persistence.project_analysis import ProjectAnalysisDocument, _document_payload, _document_from_payload
from tests import test_project_analysis_persistence as project_test


class PipelineWorkflowTests(CatalogueTestCase):
    def workflow(self):
        return create_workflow(selected_recipe())

    def test_roundtrip_and_old_project_default(self):
        workflow = replace(self.workflow(), display_state={'selected_node': 'pcv_seed_binary_median_blur',
            'selected_output': 'gray_image', 'image_visible': True, 'pipeline_visible': False})
        graph = workflow_graph(workflow)
        graph.node('pcv_seed_binary_median_blur').x += 10
        # Resolve the actual authored stage ID rather than inventing an operation.
        edited = workflow_from_graph(workflow, graph)
        self.assertEqual(PipelineWorkflow.from_json(edited.to_json()), edited)
        legacy = edited.to_dict()
        del legacy['display_state']
        self.assertEqual(dict(PipelineWorkflow.from_dict(legacy).display_state), {})
        self.assertEqual(edited.recipe.identity.analytical_dict(), workflow.recipe.identity.analytical_dict())
        self.assertEqual(graph.reference_recipe_identity, edited.recipe.identity)
        document = ProjectAnalysisDocument(project_test.ProjectAnalysisPersistenceTests._profile(), (), pipeline_workflow=edited)
        payload = _document_payload(document)
        self.assertEqual(_document_from_payload(payload).pipeline_workflow, edited)
        payload['version'] = 6
        del payload['pipeline_workflow']
        self.assertIsNone(_document_from_payload(payload).pipeline_workflow)

    def test_source_and_working_operation_contracts_reject_tampering(self):
        workflow = self.workflow()
        changed = replace(workflow.recipe.stages[0], operation_id='arbitrary.import')
        with self.assertRaises(ValueError):
            replace(workflow, recipe=replace(workflow.recipe, stages=(changed,)+workflow.recipe.stages[1:]))
        with self.assertRaises(ValueError):
            replace(workflow, source_recipe=replace(workflow.source_recipe, stages=(changed,)+workflow.recipe.stages[1:]))
        self.assertEqual(len(workflow_catalogue()), 18)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_real_run_to_node_rewire_bypass_and_layout_cache(self):
        workflow = self.workflow()
        graph = workflow_graph(workflow)
        median = next(s for s in workflow.recipe.stages if s.operation_id == 'filter.median_blur')
        threshold = next(s for s in workflow.recipe.stages if s.operation_id == 'threshold.binary')
        source = next(s for s in workflow.recipe.stages if s.operation_id == 'external.image_inputs')
        gray, rgb = (torch.as_tensor(a, device='cuda') for a in fixture())
        cache = PipelineAnalysisCache()
        def run(w):
            return run_native_recipe(w.recipe, gray, rgb, source_sha256=SOURCE, coordinate_frame=FRAME,
                preparation=PREPARATION, cache=cache, working_workflow=w, target_node_id=threshold.stage_id)
        first = run(workflow)
        self.assertEqual({n.stage_id for n in first.nodes if n.status != 'not_run'}, {source.stage_id, median.stage_id, threshold.stage_id})
        graph.node(median.stage_id).x += 5
        run(replace(workflow_from_graph(workflow, graph), display_state={'selected_node': median.stage_id,
            'selected_output': 'gray_image', 'image_visible': True, 'pipeline_visible': False}))
        self.assertEqual(set(cache.last_reused_nodes), {source.stage_id, median.stage_id, threshold.stage_id})
        graph.node(median.stage_id).enabled = False
        bypass = workflow_from_graph(workflow, graph)
        result = run(bypass)
        self.assertEqual(result.node(median.stage_id).status, 'bypassed')
        self.assertEqual(set(cache.last_computed_nodes), {median.stage_id, threshold.stage_id})
        self.assertEqual(result.node(source.stage_id).reused, True)
        # Rewire threshold directly to the existing typed source input.
        edge = next(e for e in workflow.recipe.dependencies if e.target_stage == threshold.stage_id)
        incoming = next(e for e in workflow.recipe.dependencies if e.target_stage == median.stage_id)
        deps = tuple(replace(e, source_stage=source.stage_id, source_port=incoming.source_port)
            if e == edge else e for e in workflow.recipe.dependencies)
        rewired_recipe = replace(workflow.recipe, dependencies=deps)
        from seedvision.pipeline.workflows import _working_provenance
        rewired = PipelineWorkflow(workflow.source_recipe, replace(rewired_recipe, provenance=_working_provenance(workflow.source_recipe)))
        result = run(rewired)
        self.assertEqual({n.stage_id for n in result.nodes if n.status != 'not_run'}, {source.stage_id, threshold.stage_id})
        self.assertTrue(result.node(source.stage_id).reused)
        self.assertEqual(cache.last_computed_nodes, (threshold.stage_id,))
        self.assertEqual(result.status, 'partial')
        # Partial execution still archives the exact graph with withheld descendants.
        import tempfile
        from seedvision.export.native_results import export_native_recipe_result, load_native_recipe_result
        with tempfile.TemporaryDirectory() as temporary:
            from pathlib import Path
            report = export_native_recipe_result(Path(temporary)/'partial', rewired.recipe, result)
            reopened, _ = load_native_recipe_result(report.parent)
            self.assertEqual(reopened['status'], 'partial')
            self.assertEqual(next(n for n in reopened['nodes'] if n['stage_id'] == median.stage_id)['status'], 'not_run')
        def mutate():
            gray[0, 0] += 1
            return False
        with self.assertRaises(NativeRecipeError):
            run_native_recipe(rewired.recipe, gray, rgb, source_sha256=SOURCE, coordinate_frame=FRAME,
                preparation=PREPARATION, cache=cache, working_workflow=rewired,
                target_node_id=threshold.stage_id, cancel=mutate)
        self.assertFalse(any(k.startswith('reference.recipe:workspace:') for k in cache.values))
        with self.assertRaises(NativeRecipeError):
            from seedvision.benchmarking.native_recipe import _validate_recipe
            _validate_recipe(rewired.recipe)

class PipelineWorkflowModelPersistenceTests(CatalogueTestCase):
    from tests import test_native_result_export as fixtures
    setUp = fixtures.NativeResultExportTests.setUp
    tearDown = fixtures.NativeResultExportTests.tearDown
    make_recipe = fixtures.NativeResultExportTests.make_recipe


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class PipelineWorkflowCorrectionPersistenceTests(CatalogueTestCase):
    from tests import test_project_correction_inputs as correction_fixture
    setUp = correction_fixture.ProjectCorrectionInputTests.setUp
    tearDown = correction_fixture.ProjectCorrectionInputTests.tearDown
    make_binding = correction_fixture.ProjectCorrectionInputTests.make_binding


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class PipelineWorkflowCardAuditTests(CatalogueTestCase):
    def test_complete_card_and_calibrated_otsu_workflows_share_final_audit_scope(self):
        import tempfile
        import cv2
        from pathlib import Path
        from tests import test_native_card_recipe as card_test, test_native_calibrated_seed_recipe as calibrated_test
        from seedvision.benchmarking.native_card_recipe import build_native_card_recipe
        from seedvision.pipeline.workflows import run_workflow_path
        with tempfile.TemporaryDirectory() as temporary:
            for index, (recipe, image) in enumerate(((build_native_card_recipe(), card_test.fixture(varied=True)),
                (calibrated_test.selected_recipe('otsu'), calibrated_test.fixture()))):
                path = Path(temporary)/f'card-{index}.png'
                self.assertTrue(cv2.imwrite(str(path), image))
                result = run_workflow_path(create_workflow(recipe), path)
                self.assertEqual(result.status, 'executed', [(n.stage_id,n.reason) for n in result.nodes])
                self.assertIn('workspace_workflow_adapter', result.runtime)
