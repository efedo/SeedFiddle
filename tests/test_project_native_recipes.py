"""Editable native project selections coexist with exact historical graph records."""

from tests.catalogue_fixtures import CatalogueTestCase

from tests.functional_units import functional_units
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import time
import unittest
from unittest.mock import patch

import torch

from seedvision.benchmarking import native_kmeans_recipe as kmeans
from seedvision.benchmarking.native_recipe import run_native_recipe_path
from seedvision.benchmarking.recipes import LayoutPosition, RecipeProvenanceMode
from seedvision.export import native_results as outputs
from seedvision.export.project_native_graphs import export_project_native_graphs
from seedvision.persistence import native_recipes as working
from seedvision.persistence import reference_models as assets
from seedvision.persistence.analysis_settings import analysis_settings_profile_from_graph
from seedvision.persistence.project_analysis import ProjectAnalysisStore, ProjectImageSpec, InvalidProjectAnalysis
from seedvision.persistence.snapshot import create_snapshot, restore_snapshot
from seedvision.pipeline import build_default_pipeline
from seedvision.segmentation.baseline import PipelineAnalysisCache
from tests import test_native_result_export as fixtures
from tests.test_native_kmeans_recipe import _parameter


@functional_units
class ProjectNativeRecipeTests(CatalogueTestCase):
    setUp = fixtures.NativeResultExportTests.setUp
    tearDown = fixtures.NativeResultExportTests.tearDown
    make_recipe = fixtures.NativeResultExportTests.make_recipe

    def alternate(self):
        path = self.root/'alternate.json'
        payload = json.loads(self.model_path.read_bytes())
        payload['centres'] = list(reversed(payload['centres']))
        path.write_text(json.dumps(payload), encoding='utf-8')
        recipe = _parameter(self.recipe, kmeans.MODEL_STAGE_ID, 'model_source', kmeans.model_source_for_path(path))
        recipe = _parameter(recipe, kmeans.MASK_STAGE_ID, 'cat_list', [1])
        return replace(recipe, layout=tuple(LayoutPosition(stage.stage_id, 100.+index*240., 75.+index*50.)
                                            for index, stage in enumerate(recipe.stages)))

    def project(self, native=None, *, historical=True):
        workspace = self.root/'workspace'
        workspace.mkdir()
        shutil.copyfile(self.source, workspace/'seeds.png')
        (workspace/'config').mkdir()
        shutil.copyfile(Path(__file__).parents[1]/'config'/'traits.json', workspace/'config'/'traits.json')
        graphs = (fixtures._attachment(self.recipe),) if historical else ()
        selected = (self.alternate(),) if native is None else native
        master = ProjectAnalysisStore(workspace).capture_and_save(
            analysis_settings=analysis_settings_profile_from_graph(build_default_pipeline()),
            images=(ProjectImageSpec(workspace/'seeds.png', (13, 17)),), reference_graphs=graphs,
            native_recipes=selected, destination=workspace/'test.seedfiddle-project.json')
        return workspace, master, graphs, selected

    def check_test_schema5_roundtrip_preserves_overlapping_history_and_working_selections(self):
        workspace, master, history, native = self.project()
        loaded = ProjectAnalysisStore(workspace).load(master)
        self.assertFalse(loaded.issues)
        self.assertEqual(json.loads(master.read_bytes())['version'], 6)
        self.assertEqual(loaded.document.reference_graphs, history)
        self.assertEqual(loaded.document.native_recipes, native)
        self.assertEqual(working.recipe_key(history[0].recipe), working.recipe_key(native[0]))
        self.assertNotEqual(history[0].recipe.identity, native[0].identity)
        self.assertNotEqual(history[0].recipe.layout, native[0].layout)
        self.assertEqual(len(loaded.document.reference_model_assets), 2)

    def check_test_schemas1_through4_migrate_without_fabricated_working_recipes(self):
        workspace, master, _, _ = self.project((), historical=False)
        store = ProjectAnalysisStore(workspace)
        current = json.loads(master.read_bytes())
        for version in range(1, 5):
            payload = deepcopy(current)
            payload['version'] = version
            payload.pop('correction_bindings')
            payload.pop('native_recipes')
            if version < 4:
                payload.pop('reference_model_assets')
            if version < 3:
                payload.pop('reference_graphs')
            if version == 1:
                payload['project'] = {key: payload['project'][key] for key in ('species', 'selected_image')}
                for row in payload['images']:
                    row.pop('biological_context')
                    row.pop('capture_group_id')
            legacy = workspace/f'v{version}.seedfiddle-project.json'
            legacy.write_text(json.dumps(payload), encoding='utf-8')
            with self.subTest(version=version):
                document = store.load(legacy).document
                self.assertEqual(document.native_recipes, ())
                migrated = store.save(document, workspace/f'migrated{version}.seedfiddle-project.json')
                self.assertEqual(json.loads(migrated.read_bytes())['native_recipes'], [])
                payload['native_recipes'] = []
                legacy.write_text(json.dumps(payload), encoding='utf-8')
                with self.assertRaisesRegex(InvalidProjectAnalysis, 'unknown field'):
                    store.load(legacy)

    def check_test_invalid_working_metadata_leaves_master_and_model_reads_untouched(self):
        workspace, master, _, native = self.project()
        store = ProjectAnalysisStore(workspace)
        document = store.load(master).document
        original = master.read_bytes()
        wrong_mode = replace(native[0], provenance=replace(native[0].provenance,
                                                            mode=RecipeProvenanceMode.ORIGINAL_REFERENCE))
        for value in ([native[0]], (native[0], native[0]), (wrong_mode,)):
            with self.subTest(value_type=type(value).__name__), patch.object(assets, 'verified_asset_bytes') as read:
                with self.assertRaises(InvalidProjectAnalysis):
                    store.save(replace(document, native_recipes=value), master)
                read.assert_not_called()
                self.assertEqual(master.read_bytes(), original)
        payload = json.loads(original)
        payload['native_recipes'][0]['unknown'] = True
        master.write_text(json.dumps(payload), encoding='utf-8')
        with self.assertRaises(InvalidProjectAnalysis):
            store.load(master)

    def check_test_working_count_and_aggregate_caps_reject_before_asset_reads(self):
        native = self.alternate()
        for name, limit in (('MAX_NATIVE_RECIPES', 0), ('MAX_NATIVE_RECIPES_BYTES', 1)):
            with patch.object(working, name, limit), patch.object(assets, 'verified_asset_bytes') as read:
                with self.assertRaises(ValueError):
                    assets.capture_reference_model_assets(self.root, (), native_recipes=(native,))
                read.assert_not_called()
        with patch.object(assets, 'MAX_MODEL_ASSETS', 1), patch.object(assets, 'verified_asset_bytes') as read:
            with self.assertRaises(ValueError):
                assets.capture_reference_model_assets(self.root, (fixtures._attachment(self.recipe),), native_recipes=(native,))
            read.assert_not_called()

    def check_test_same_selection_in_both_collections_deduplicates_binding_and_object(self):
        workspace, master, history, native = self.project((self.recipe,))
        document = ProjectAnalysisStore(workspace).load(master).document
        self.assertEqual(len(document.reference_model_assets), 1)
        self.assertEqual(len(list((workspace/'native-models').iterdir())), 1)
        assets.verify_reference_model_assets(workspace, document.reference_model_assets, history,
                                             native_recipes=native, require_complete=True)

    def check_test_removing_working_selection_prunes_only_its_binding(self):
        workspace, master, history, native = self.project()
        store = ProjectAnalysisStore(workspace)
        document = store.load(master).document
        previous = assets.model_asset_paths(workspace, history[0].recipe, document.reference_model_assets)
        captured = store.capture(analysis_settings=document.analysis_settings,
            images=(ProjectImageSpec(workspace/'seeds.png', (13, 17)),), reference_graphs=history,
            native_recipes=(), reference_model_assets=document.reference_model_assets)
        self.assertEqual(captured.reference_graphs, history)
        self.assertEqual(captured.native_recipes, ())
        self.assertEqual(len(captured.reference_model_assets), 1)
        self.assertEqual(assets.model_asset_paths(workspace, history[0].recipe, captured.reference_model_assets), previous)
        self.assertEqual(assets.model_asset_paths(workspace, native[0], captured.reference_model_assets), {})

    def check_test_selection_replaces_same_key_atomically_and_preserves_other_recipes(self):
        other = replace(self.recipe, recipe_id='another_native_workflow')
        selected = self.alternate()
        current = (self.recipe, other)
        self.assertEqual(working.select_native_recipe(current, selected), (selected, other))
        self.assertEqual(current, (self.recipe, other))
        with patch.object(working, 'MAX_NATIVE_RECIPES', 2):
            with self.assertRaises(ValueError):
                working.select_native_recipe(current, replace(other, recipe_id='third'))
        self.assertEqual(current, (self.recipe, other))

    def check_test_working_model_copy_failure_cannot_replace_previous_project(self):
        workspace, master, history, _ = self.project(())
        store = ProjectAnalysisStore(workspace)
        before = master.read_bytes()
        document = store.load(master).document
        selected = self.alternate()
        with patch.object(assets.os, 'replace', side_effect=OSError('interrupted model publication')):
            with self.assertRaises(InvalidProjectAnalysis):
                store.capture_and_save(analysis_settings=document.analysis_settings,
                    images=(ProjectImageSpec(workspace/'seeds.png', (13, 17)),), reference_graphs=history,
                    reference_model_assets=document.reference_model_assets, native_recipes=(selected,),
                    destination=master)
        self.assertEqual(master.read_bytes(), before)
        self.assertEqual(store.load(master).document.native_recipes, ())
        self.assertFalse(list((workspace/'native-models').glob('*.incomplete')))





    def check_test_unknown_native_recipe_failure_is_explicit_and_preserves_history(self):
        workspace, master, history, _ = self.project(())
        document = ProjectAnalysisStore(workspace).load(master).document
        unsupported = replace(self.recipe, recipe_id='future_native_workflow')
        original_recipe = replace(history[0].recipe,
            provenance=replace(history[0].recipe.provenance, mode=RecipeProvenanceMode.ORIGINAL_REFERENCE))
        # Use a complete valid historical original graph rather than inventing run records.
        original = fixtures._attachment(original_recipe)
        ledger = export_project_native_graphs((original,), workspace/'seeds.png', self.root/'unsupported',
            expected_source_sha256=document.images[0].source.sha256, native_recipes=(unsupported,))
        rows = json.loads(ledger.read_bytes())['graphs']
        self.assertEqual([row['status'] for row in rows], ['unavailable', 'failed'])
        self.assertTrue(rows[1]['reason'])
        self.assertEqual(rows[1]['historical_stage_runs'], [])
        self.assertIsNone(rows[1]['report'])
        self.assertEqual(list(ledger.parent.iterdir()), [ledger])

    def qt(self):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        from PySide6.QtWidgets import QApplication
        return QApplication.instance() or QApplication([])


    def check_test_keep_failure_and_remove_leave_historical_records_unchanged(self):
        from seedvision.ui.native_kmeans_dialog import NativeKMeansDialog
        app = self.qt()
        workspace, master, history, native = self.project()
        selected = [native]
        events = []
        def keep(recipe):
            updated = working.select_native_recipe(selected[0], recipe)
            selected[0] = updated
            events.append(updated)
        dialog = NativeKMeansDialog(self.recipe, keep_in_project=keep)
        try:
            with patch.object(working, 'MAX_NATIVE_RECIPES', 0):
                dialog.project_button.click()
            self.assertIn('Could not keep', dialog.status_label.text())
            self.assertEqual(selected[0], native)
            self.assertEqual(events, [])
            selected[0] = working.validate_native_recipes(tuple(recipe for recipe in selected[0]
                if working.recipe_key(recipe) != working.recipe_key(self.recipe)))
            events.append(selected[0])
            self.assertEqual(selected[0], ())
            self.assertEqual(events, [()])
            self.assertEqual(ProjectAnalysisStore(workspace).load(master).document.reference_graphs, history)
        finally:
            dialog.reject()
            dialog.deleteLater()
            app.processEvents()


    def test_workflow_lifecycle(self):
        """5 related ProjectNativeRecipeTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_schema5_roundtrip_preserves_overlapping_history_and_working_selections)
        self.scenario(self.check_test_schemas1_through4_migrate_without_fabricated_working_recipes)
        self.scenario(self.check_test_same_selection_in_both_collections_deduplicates_binding_and_object)
        self.scenario(self.check_test_removing_working_selection_prunes_only_its_binding)
        self.scenario(self.check_test_selection_replaces_same_key_atomically_and_preserves_other_recipes)

    def test_validation_and_recovery(self):
        """5 related ProjectNativeRecipeTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_invalid_working_metadata_leaves_master_and_model_reads_untouched)
        self.scenario(self.check_test_working_count_and_aggregate_caps_reject_before_asset_reads)
        self.scenario(self.check_test_working_model_copy_failure_cannot_replace_previous_project)
        self.scenario(self.check_test_unknown_native_recipe_failure_is_explicit_and_preserves_history)
        self.scenario(self.check_test_keep_failure_and_remove_leave_historical_records_unchanged)
