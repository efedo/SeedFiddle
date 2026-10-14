"""Editable native project selections coexist with exact historical graph records."""
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


class ProjectNativeRecipeTests(unittest.TestCase):
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

    def test_schema5_roundtrip_preserves_overlapping_history_and_working_selections(self):
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

    def test_schemas1_through4_migrate_without_fabricated_working_recipes(self):
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

    def test_invalid_working_metadata_leaves_master_and_model_reads_untouched(self):
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

    def test_working_count_and_aggregate_caps_reject_before_asset_reads(self):
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

    def test_same_selection_in_both_collections_deduplicates_binding_and_object(self):
        workspace, master, history, native = self.project((self.recipe,))
        document = ProjectAnalysisStore(workspace).load(master).document
        self.assertEqual(len(document.reference_model_assets), 1)
        self.assertEqual(len(list((workspace/'native-models').iterdir())), 1)
        assets.verify_reference_model_assets(workspace, document.reference_model_assets, history,
                                             native_recipes=native, require_complete=True)

    def test_removing_working_selection_prunes_only_its_binding(self):
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

    def test_selection_replaces_same_key_atomically_and_preserves_other_recipes(self):
        other = replace(self.recipe, recipe_id='another_native_workflow')
        selected = self.alternate()
        current = (self.recipe, other)
        self.assertEqual(working.select_native_recipe(current, selected), (selected, other))
        self.assertEqual(current, (self.recipe, other))
        with patch.object(working, 'MAX_NATIVE_RECIPES', 2):
            with self.assertRaises(ValueError):
                working.select_native_recipe(current, replace(other, recipe_id='third'))
        self.assertEqual(current, (self.recipe, other))

    def test_working_model_copy_failure_cannot_replace_previous_project(self):
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

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_overlapping_model_selections_use_exact_transport_and_distinct_cache_products(self):
        workspace, master, history, native = self.project()
        document = ProjectAnalysisStore(workspace).load(master).document
        original_paths = assets.model_asset_paths(workspace, self.recipe, document.reference_model_assets)
        working_paths = assets.model_asset_paths(workspace, native[0], document.reference_model_assets)
        self.assertNotEqual(original_paths, working_paths)
        self.model_path.unlink()
        (self.root/'alternate.json').unlink()
        cache = PipelineAnalysisCache()
        before = run_native_recipe_path(self.recipe, workspace/'seeds.png', cache=cache, model_asset_paths=original_paths)
        after = run_native_recipe_path(native[0], workspace/'seeds.png', cache=cache, model_asset_paths=working_paths)
        a, b = before.node(kmeans.PREDICT_STAGE_ID), after.node(kmeans.PREDICT_STAGE_ID)
        self.assertTrue(torch.equal(a.outputs['labels'], 1-b.outputs['labels']))
        self.assertNotEqual(a.product_key, b.product_key)
        self.assertFalse(b.reused)
        self.assertTrue(after.nodes[0].reused)
        self.assertEqual(b.outputs['labels'].device.type, 'cuda')
        Path(next(iter(working_paths.values()))).write_bytes(b'broken working model')
        # A different retained selection must still run; the broken exact one must fail.
        run_native_recipe_path(self.recipe, workspace/'seeds.png', cache=cache, model_asset_paths=original_paths)
        with self.assertRaises(ValueError):
            run_native_recipe_path(native[0], workspace/'seeds.png', cache=cache, model_asset_paths=working_paths)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_snapshot_restore_retains_both_recipes_and_model_closure_after_original_deletion(self):
        workspace, master, history, native = self.project()
        snapshot = create_snapshot(workspace, master, self.root/'snapshots')
        self.model_path.unlink()
        (self.root/'alternate.json').unlink()
        shutil.rmtree(workspace)
        restored = self.root/'restored'
        project_path = restore_snapshot(snapshot, restored)
        loaded = ProjectAnalysisStore(restored).load(project_path)
        document = loaded.document
        self.assertEqual(document.reference_graphs, history)
        self.assertEqual(document.native_recipes, native)
        assets.verify_reference_model_assets(restored, document.reference_model_assets, history,
                                             native_recipes=native, require_complete=True)
        for recipe in (history[0].recipe, native[0]):
            paths = assets.model_asset_paths(restored, recipe, document.reference_model_assets)
            result = run_native_recipe_path(recipe, loaded.images[0].path, model_asset_paths=paths)
            self.assertEqual(result.recipe_identity, recipe.identity)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_batch_distinguishes_history_and_working_runs_without_fabricated_records(self):
        workspace, master, history, native = self.project()
        document = ProjectAnalysisStore(workspace).load(master).document
        self.model_path.unlink()
        (self.root/'alternate.json').unlink()
        ledger = export_project_native_graphs(history, workspace/'seeds.png', self.root/'runs',
            expected_source_sha256=document.images[0].source.sha256, project_root=workspace,
            reference_model_assets=document.reference_model_assets, native_recipes=native)
        payload = json.loads(ledger.read_bytes())
        self.assertEqual(payload['version'], 3)
        rows = payload['graphs']
        self.assertEqual([row['kind'] for row in rows], ['historical_graph', 'working_recipe'])
        self.assertEqual([row['status'] for row in rows], ['executed', 'executed'])
        self.assertEqual(rows[0]['historical_stage_runs'], [record.to_dict() for record in history[0].stage_runs])
        self.assertEqual(rows[1]['historical_stage_runs'], [])
        self.assertEqual(rows[1]['historical_checkpoint_data'], 'not_recorded')
        self.assertIsNone(rows[1]['attachment_index'])
        self.assertEqual(rows[1]['working_recipe_index'], 0)
        for recipe, row in zip((self.recipe, native[0]), rows, strict=True):
            report, arrays = outputs.load_native_recipe_result(ledger.parent/Path(row['report']).parent)
            self.assertEqual(report['recipe_identity'], recipe.identity.to_dict())
            self.assertTrue(arrays)
        self.assertFalse(next(row for row in rows[1]['nodes'] if row['stage_id'] == kmeans.PREDICT_STAGE_ID)['reused'])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_working_only_saved_project_batch_uses_same_runner(self):
        from seedvision.export.project_runner import run_project
        from tests.test_seed_crop_artifacts import _analysis
        workspace, master, history, native = self.project(historical=False)
        self.model_path.unlink()
        (self.root/'alternate.json').unlink()
        with patch('seedvision.export.project_runner.analyze_path', side_effect=lambda path, **kw: _analysis(path)):
            reports = run_project(workspace, master, self.root/'batch')
        ledger = json.loads((reports[0]/'native_graphs'/'native-graphs.json').read_bytes())
        self.assertEqual(len(ledger['graphs']), 1)
        self.assertEqual(ledger['graphs'][0]['kind'], 'working_recipe')
        self.assertEqual(ledger['graphs'][0]['status'], 'executed')
        self.assertEqual(ProjectAnalysisStore(workspace).load(master).document.reference_graphs, history)

    def test_unknown_native_recipe_failure_is_explicit_and_preserves_history(self):
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

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_desktop_keep_save_as_reopen_menu_and_new_preserve_exact_working_selection(self):
        from PySide6.QtWidgets import QFileDialog
        from seedvision.ui.reference_graph_dialog import ReferenceGraphDialog
        from seedvision.ui.native_kmeans_dialog import NativeKMeansDialog
        from tests import test_project_analysis_ui as ui
        app = self.qt()
        ui.ProjectAnalysisUiTests.application = app
        workspace, master, history, _ = self.project(())
        selected = self.alternate()
        window = ui.ProjectAnalysisUiTests._window(workspace)
        kept = []
        def edit(dialog):
            self.assertFalse(dialog.project_button.isHidden())
            dialog._set_model_source(dict(selected.parameters_for_runner(kmeans.MODEL_STAGE_ID)['model_source']))
            dialog._parameter_changed(kmeans.MASK_STAGE_ID, 'cat_list', '[1]')
            for position in selected.layout:
                dialog.canvas.node_items[position.stage_id].setPos(position.x, position.y)
            dialog.set_source_path(workspace/'seeds.png')
            dialog.start_run()
            self.assertFalse(dialog.project_button.isEnabled())
            deadline = time.monotonic()+20
            while dialog._worker is not None and time.monotonic() < deadline:
                app.processEvents()
                time.sleep(.01)
            self.assertIsNotNone(dialog.result, dialog.status_label.text())
            result, cache_values = dialog.result, dict(dialog.cache.values)
            dialog.project_button.click()
            self.assertIs(dialog.result, result)
            self.assertEqual(dialog.cache.values.keys(), cache_values.keys())
            for key, value in cache_values.items():
                self.assertIs(dialog.cache.values[key], value)
            kept.append(dialog._selected_recipe_with_layout())
            return 0
        def interact(graph):
            with patch.object(NativeKMeansDialog, 'exec', edit):
                graph._open_native_kmeans()
            self.assertEqual(graph.native_recipes, (selected,))
            self.assertTrue(graph._saved_native_button.isEnabled())
            return 0
        try:
            with patch.object(window, '_confirm_project_replacement', return_value=True):
                self.assertTrue(window._open_project(master))
            with patch.object(ReferenceGraphDialog, 'exec', interact):
                window._show_reference_graphs()
            self.assertEqual(kept, [selected])
            self.assertTrue(window._project_dirty)
            self.assertEqual(window._project_reference_graphs[:len(history)], history)
            destination = workspace/'other-folder'/'saved-as.seedfiddle-project.json'
            with patch.object(QFileDialog, 'getSaveFileName', return_value=(str(destination), '')):
                window.save_project_as_action.trigger()
            self.assertEqual(window._current_project_path, destination)
            self.model_path.unlink()
            (self.root/'alternate.json').unlink()
            document = ProjectAnalysisStore(workspace).load(destination).document
            self.assertEqual(document.native_recipes, (selected,))
            self.assertEqual(document.reference_graphs[:len(history)], history)
            with patch.object(window, '_confirm_project_replacement', return_value=True):
                self.assertTrue(window._open_project(destination))
            observed = []
            def reopen(dialog):
                self.assertEqual(dialog.recipe, selected)
                self.assertEqual(dialog._selected_recipe_with_layout(), selected)
                self.assertEqual(dialog.model_asset_paths,
                    assets.model_asset_paths(workspace, selected, document.reference_model_assets))
                observed.append(dialog.recipe)
                return 0
            def menu(graph):
                action = graph._native_menu.actions()[0].menu().actions()[0]
                with patch.object(NativeKMeansDialog, 'exec', reopen):
                    action.trigger()
                    graph._open_native_kmeans()
                return 0
            with patch.object(ReferenceGraphDialog, 'exec', menu):
                window._show_reference_graphs()
            self.assertEqual(observed, [selected, selected])
            with patch.object(window, '_confirm_project_replacement', return_value=True):
                self.assertTrue(window._new_project())
            self.assertEqual(window._project_native_recipes, ())
            self.assertEqual(window._project_reference_model_assets, ())
            self.assertEqual(ProjectAnalysisStore(workspace).load(destination).document.native_recipes, (selected,))
        finally:
            ui.ProjectAnalysisUiTests._dispose(window)

    def test_keep_failure_and_remove_leave_historical_records_unchanged(self):
        from seedvision.ui.reference_graph_dialog import ReferenceGraphDialog
        from seedvision.ui.native_kmeans_dialog import NativeKMeansDialog
        app = self.qt()
        workspace, master, history, native = self.project()
        graph = ReferenceGraphDialog(reference_graphs=history, native_recipes=native, project_root=workspace)
        events = []
        graph.native_recipes_changed.connect(events.append)
        dialog = NativeKMeansDialog(self.recipe, keep_in_project=graph._keep_native_recipe)
        try:
            with patch.object(working, 'MAX_NATIVE_RECIPES', 0):
                dialog.project_button.click()
            self.assertIn('Could not keep', dialog.status_label.text())
            self.assertEqual(graph.native_recipes, native)
            self.assertEqual(events, [])
            graph._native_menu.actions()[0].menu().actions()[1].trigger()
            self.assertEqual(graph.native_recipes, ())
            self.assertEqual(events, [()])
            self.assertEqual(graph.reference_graphs[:len(history)], history)
        finally:
            dialog.reject()
            graph.reject()
            dialog.deleteLater()
            graph.deleteLater()
            app.processEvents()

    def test_seed_extended_threshold_and_bayes_dialogs_keep_and_reopen_through_shared_path(self):
        from seedvision.benchmarking.native_recipe import build_native_seed_recipe
        from seedvision.benchmarking.native_extended_threshold_recipes import build_native_extended_threshold_recipes
        from seedvision.benchmarking.native_naive_bayes_recipe import build_native_naive_bayes_recipe
        from seedvision.ui.reference_graph_dialog import ReferenceGraphDialog
        from seedvision.ui.native_recipe_dialog import NativeRecipeDialog
        app = self.qt()
        selected = (build_native_seed_recipe(fixtures.selected_recipe()),
                    next(iter(build_native_extended_threshold_recipes().values())),
                    build_native_naive_bayes_recipe())
        graph = ReferenceGraphDialog(project_root=self.root)
        history = graph.reference_graphs
        kept = []
        def keep(dialog):
            self.assertFalse(dialog.project_button.isHidden())
            dialog.project_button.click()
            self.assertIn('Working recipe kept', dialog.status_label.text())
            kept.append(dialog._selected_recipe_with_layout())
            return 0
        try:
            with patch.object(NativeRecipeDialog, 'exec', keep):
                for recipe in selected:
                    graph._open_working_native_recipe(recipe)
            self.assertEqual(graph.native_recipes, tuple(kept))
            observed = []
            def reopen(dialog):
                observed.append(dialog._selected_recipe_with_layout())
                return 0
            with patch.object(NativeRecipeDialog, 'exec', reopen):
                for recipe in selected:
                    graph._open_working_native_recipe(recipe)
            self.assertEqual(observed, kept)
            self.assertEqual(graph.reference_graphs, history)
        finally:
            graph.reject()
            graph.deleteLater()
            app.processEvents()
