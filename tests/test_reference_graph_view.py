from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch


class ReferenceGraphViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        cls.application = QApplication.instance() or QApplication([])

    @classmethod
    def tearDownClass(cls) -> None:
        cls.application.processEvents()

    @staticmethod
    def _catalogue():
        from seedvision.benchmarking.recipe_catalogue import build_recipe_catalogue

        return build_recipe_catalogue()

    def _recipe(self, recipe_id: str):
        return next(
            recipe for recipe in self._catalogue().recipes
            if recipe.recipe_id == recipe_id
        )

    def _dialog(self, recipe_id: str | None = None, attachments=()):
        from seedvision.ui.reference_graph_dialog import ReferenceGraphDialog

        dialog = ReferenceGraphDialog(reference_graphs=tuple(attachments))
        if recipe_id is not None:
            index = dialog.recipe_selector.findData(recipe_id)
            self.assertGreaterEqual(index, 0)
            dialog.recipe_selector.setCurrentIndex(index)
            self.application.processEvents()
        dialog.resize(1440, 900)
        dialog.show()
        self.application.processEvents()
        return dialog

    @staticmethod
    def _select_exact_entry(dialog, recipe_id: str, version: str, kind: str):
        from PySide6.QtCore import Qt

        for index in range(dialog.recipe_selector.count()):
            entry = dialog.recipe_selector.itemData(
                index, Qt.ItemDataRole.UserRole + 1
            )
            if entry == (kind, recipe_id, version):
                dialog.recipe_selector.setCurrentIndex(index)
                return index
        raise AssertionError(f"No {kind} entry for {recipe_id} v{version}.")

    @staticmethod
    def _attachment(recipe, *, recipe_id: str | None = None, version: str | None = None):
        from seedvision.benchmarking.recipes import initial_stage_run
        from seedvision.persistence.reference_graphs import ReferenceGraphAttachment

        if recipe_id is not None or version is not None:
            recipe = replace(
                recipe,
                recipe_id=recipe_id if recipe_id is not None else recipe.recipe_id,
                recipe_version=version if version is not None else recipe.recipe_version,
            )
        return ReferenceGraphAttachment(
            recipe,
            tuple(initial_stage_run(recipe, stage.stage_id) for stage in recipe.stages),
            selected_stage_id=recipe.stages[0].stage_id,
        )

    @staticmethod
    def _attachment_json_size(attachment) -> int:
        return len(
            json.dumps(
                attachment.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )

    def _main_window(self, root: Path):
        from PySide6.QtCore import QSettings
        from seedvision.ui.main_window import MainWindow

        settings = QSettings(
            str(root / "view-test-settings.ini"), QSettings.Format.IniFormat
        )
        settings.clear()
        with patch("seedvision.ui.main_window.QSettings", return_value=settings):
            window = MainWindow(root)
        window._test_application_settings = settings
        return window

    def _dispose_main_window(self, window) -> None:
        window._project_tracking_enabled = False
        window._project_dirty = False
        window._reference_masks_dirty.clear()
        window._instance_annotations_dirty.clear()
        window._active_tasks.clear()
        window._learning_training_task = None
        window._procedural_fit_task = None
        window.close()
        self.application.processEvents()

    def test_frozen_catalogue_topology_and_read_only_canvas_are_projected(self) -> None:
        from seedvision.benchmarking.recipes import initial_stage_run, to_pipeline_graph

        catalogue = self._catalogue()
        self.assertEqual(len(catalogue.recipes), 5)
        dialog = self._dialog("plantcv_seed_binary")
        self.addCleanup(dialog.close)

        recipe = self._recipe("plantcv_seed_binary")
        expected = to_pipeline_graph(recipe)
        self.assertTrue(dialog.canvas.read_only)
        self.assertEqual(tuple(dialog.canvas.graph.nodes), tuple(expected.nodes))
        self.assertEqual(dialog.canvas.graph.connections, expected.connections)
        self.assertEqual(dialog.graph_splitter.count(), 2)
        self.assertEqual(
            dialog.canvas.graph.reference_recipe_identity.analytical_dict(),
            recipe.identity.analytical_dict(),
        )
        self.assertIn("plantcv.reference", dialog.backend_label.text())
        self.assertIn("4.11.3", dialog.backend_label.text())
        self.assertIn("original_reference", dialog.backend_label.text())
        self.assertIn("SeedFiddle", dialog.claim_label.text())
        self.assertIn("not a published end-to-end", dialog.claim_label.text())
        self.assertIn("does not execute", dialog.claim_label.text())

        selected = dialog.active_attachment.selected_stage_id
        stage = recipe.stage(selected)
        self.assertIn(stage.stage_id, dialog._facts_browser.toPlainText())
        self.assertIn("Typed ports", dialog._facts_browser.toPlainText())
        self.assertIn("Checkpoint arrays: UNAVAILABLE", dialog._facts_browser.toPlainText())
        self.assertIn("No output preview", dialog._facts_browser.toPlainText())
        self.assertEqual(
            tuple(run.state.value for run in dialog.active_attachment.stage_runs),
            tuple(initial_stage_run(recipe, item.stage_id).state.value for item in recipe.stages),
        )

    def test_stage_selection_shows_pinned_function_and_unselected_parameters(self) -> None:
        dialog = self._dialog("plantcv_seed_otsu")
        self.addCleanup(dialog.close)
        recipe = self._recipe("plantcv_seed_otsu")
        stage = next(
            stage for stage in recipe.stages
            if stage.operation_id == "filter.median_blur"
        )

        dialog.canvas.select_node(stage.stage_id)
        self.application.processEvents()
        text = dialog._facts_browser.toPlainText()
        self.assertEqual(dialog.active_attachment.selected_stage_id, stage.stage_id)
        self.assertIn("filter.median_blur", text)
        self.assertIn("plantcv.plantcv.median_blur.median_blur", text)
        self.assertIn("66988db126ba849be2c39353ac8bf301ebe0e839932cbdd5eb2cf46269e363ee", text)
        self.assertIn("ksize", text)
        self.assertIn("UNSELECTED", text)
        self.assertIn("effective runtime parameter values are not recorded", text)
        self.assertIn("uint8", text)
        self.assertIn("RGB", text)

    def test_visible_canvas_click_selects_recipe_stage(self) -> None:
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest

        dialog = self._dialog("plantcv_seed_otsu")
        self.addCleanup(dialog.close)
        recipe = self._recipe("plantcv_seed_otsu")
        source_stage = recipe.stages[0]
        target_stage = next(
            stage for stage in recipe.stages
            if stage.operation_id == "filter.median_blur"
        )
        dialog.canvas.select_node(source_stage.stage_id)
        self.application.processEvents()
        item = dialog.canvas.node_items[target_stage.stage_id]
        viewport_point = dialog.canvas.mapFromScene(
            item.mapToScene(item.boundingRect().center())
        )
        self.assertTrue(dialog.canvas.viewport().rect().contains(viewport_point))
        QTest.mouseClick(
            dialog.canvas.viewport(), Qt.MouseButton.LeftButton, pos=viewport_point
        )
        self.application.processEvents()
        self.assertEqual(dialog.active_attachment.selected_stage_id, target_stage.stage_id)
        self.assertIn(target_stage.stage_id, dialog._facts_browser.toPlainText())

    def test_color_correction_is_described_as_one_original_call(self) -> None:
        dialog = self._dialog("plantcv_color_card_correct_color")
        self.addCleanup(dialog.close)
        text = dialog._facts_browser.toPlainText()
        self.assertIn("one correct_color operation", dialog.claim_label.text())
        self.assertIn("not a complete color-calibration workflow", dialog.claim_label.text())
        self.assertIn("transform.correct_color", text)
        self.assertIn("plantcv.plantcv.transform.color_correction.correct_color", text)
        self.assertIn("Single-operation reference contract", text)

    def test_smpts_graph_keeps_paper_reconstruction_and_condition_unresolved(self) -> None:
        dialog = self._dialog("smpts_paper_reconstruction")
        self.addCleanup(dialog.close)
        recipe = self._recipe("smpts_paper_reconstruction")
        stage = recipe.stage("smpts_paper_targeted_erosion")
        dialog.canvas.select_node(stage.stage_id)
        self.application.processEvents()
        text = dialog._facts_browser.toPlainText()

        self.assertEqual(dialog.backend_label.text().split(" · ")[-1], "paper_reconstruction")
        self.assertIn("original implementation is unavailable", dialog.claim_label.text())
        self.assertIn("does not claim recovered code", dialog.claim_label.text())
        self.assertIn("Conditional stage", text)
        self.assertIn("Contact candidate selected", text)
        self.assertIn("not been evaluated", text)
        self.assertIn("UNSELECTED", text)
        self.assertIn("UNAVAILABLE", text)
        self.assertIn("smpts_original v1 — unavailable", text.lower())
        self.assertIn("no PlantCV erosion is substituted", text)

    def test_executed_history_never_claims_checkpoint_arrays_or_preview(self) -> None:
        from seedvision.benchmarking.recipes import StageRunRecord, StageRunState
        from seedvision.persistence.reference_graphs import ReferenceGraphAttachment

        recipe = self._recipe("plantcv_color_card_correct_color")
        digest = "a" * 64
        runs = (StageRunRecord(
            recipe.stages[0].stage_id,
            recipe.identity,
            StageRunState.EXECUTED,
            checkpoint_sha256=digest,
        ),)
        attachment = ReferenceGraphAttachment(
            recipe, runs, selected_stage_id=recipe.stages[0].stage_id
        )
        dialog = self._dialog(recipe.recipe_id, (attachment,))
        self.addCleanup(dialog.close)

        text = dialog._facts_browser.toPlainText()
        self.assertIn("HISTORICAL CHECKPOINT DIGEST", text.upper())
        self.assertIn(digest, text)
        self.assertIn("Checkpoint arrays: UNAVAILABLE", text)
        self.assertIn("No output preview", text)

    def test_stored_identity_mismatch_is_explicit_and_preserved(self) -> None:
        from seedvision.benchmarking.recipes import initial_stage_run
        from seedvision.persistence.reference_graphs import ReferenceGraphAttachment

        current = self._recipe("plantcv_color_card_correct_color")
        stale_provenance = replace(
            current.provenance, backend_version="4.11.2-stale"
        )
        stale = replace(current, provenance=stale_provenance)
        attachment = ReferenceGraphAttachment(
            stale,
            tuple(initial_stage_run(stale, item.stage_id) for item in stale.stages),
            selected_stage_id=stale.stages[0].stage_id,
        )
        dialog = self._dialog(stale.recipe_id, (attachment,))
        self.addCleanup(dialog.close)
        self._select_exact_entry(
            dialog, stale.recipe_id, stale.recipe_version, "stored"
        )
        self.application.processEvents()
        self.assertIn("STALE CATALOGUE IDENTITY", dialog.attachment_label.text())
        self.assertIn("4.11.2-stale", dialog.backend_label.text())
        self.assertEqual(dialog.reference_graphs, (attachment,))

    def test_historical_versions_and_unknown_recipe_ids_select_exact_stored_recipe(self) -> None:
        current = self._recipe("plantcv_color_card_correct_color")
        historical = self._attachment(current, version="99.4")
        unknown = self._attachment(current, recipe_id="removed_reference_recipe")
        attachments = (historical, unknown)

        dialog = self._dialog(attachments=attachments)
        self.addCleanup(dialog.close)
        self.assertIs(dialog.active_attachment.recipe, historical.recipe)
        self.assertEqual(dialog.active_attachment.recipe.recipe_version, "99.4")
        self.assertIn("STALE CATALOGUE IDENTITY", dialog.attachment_label.text())
        self.assertEqual(dialog.reference_graphs, attachments)

        unknown_index = self._select_exact_entry(
            dialog, "removed_reference_recipe", current.recipe_version, "stored"
        )
        self.application.processEvents()
        self.assertIn("no current catalogue entry", dialog.recipe_selector.itemText(unknown_index))
        self.assertIs(dialog.active_attachment.recipe, unknown.recipe)
        self.assertEqual(dialog.active_attachment.recipe.recipe_id, "removed_reference_recipe")
        self.assertIn("exact saved recipe", dialog.attachment_label.text())
        self.assertEqual(dialog.reference_graphs, attachments)

    def test_full_attachment_capacity_keeps_catalogue_graph_transient(self) -> None:
        current = self._recipe("plantcv_seed_binary")
        attachments = tuple(
            self._attachment(current, recipe_id=f"historical_recipe_{index:02d}")
            for index in range(16)
        )
        dialog = self._dialog(attachments=attachments)
        self.addCleanup(dialog.close)
        self.assertEqual(dialog.reference_graphs, attachments)

        index = dialog.recipe_selector.findData("plantcv_seed_binary")
        self.assertGreaterEqual(index, 0)
        dialog.recipe_selector.setCurrentIndex(index)
        self.application.processEvents()
        self.assertTrue(dialog._active_is_transient)
        self.assertIn("TRANSIENT", dialog.attachment_label.text())
        self.assertIn("NOT SAVED", dialog.attachment_label.text())
        before = dialog.reference_graphs

        dialog.canvas.select_node(dialog.active_attachment.recipe.stages[-1].stage_id)
        dialog.canvas.auto_arrange()
        dialog.canvas.set_cable_bundling_enabled(True)
        self.application.processEvents()
        self.assertEqual(dialog.reference_graphs, before)
        self.assertEqual(len(dialog.reference_graphs), 16)
        self.assertTrue(dialog._active_is_transient)
        self.assertIn("TRANSIENT", dialog.attachment_label.text())

    def test_exact_production_aggregate_byte_cap_keeps_new_recipe_view_consistent(self) -> None:
        from seedvision.persistence.reference_graphs import (
            MAX_REFERENCE_GRAPHS_BYTES,
            validate_reference_graphs,
        )

        base = self._recipe("plantcv_color_card_correct_color")
        attachments = [
            self._attachment(base, recipe_id="historical_small")
        ]
        for index in range(4):
            recipe = replace(
                base,
                recipe_id=f"historical_large_{index}",
                description=base.description + " " + "x" * 1_600_000,
            )
            attachments.append(self._attachment(recipe))
        fifth_id = "historical_large_4"
        fifth_base = self._attachment(base, recipe_id=fifth_id)
        remaining = (
            MAX_REFERENCE_GRAPHS_BYTES
            - sum(self._attachment_json_size(item) for item in attachments)
            - self._attachment_json_size(fifth_base)
        )
        self.assertGreater(remaining, 1)
        fifth_recipe = replace(
            base,
            recipe_id=fifth_id,
            description=base.description + " " + "x" * (remaining - 1),
        )
        attachments.append(self._attachment(fifth_recipe))
        frozen = tuple(attachments)
        self.assertEqual(
            sum(self._attachment_json_size(item) for item in frozen),
            MAX_REFERENCE_GRAPHS_BYTES,
        )
        self.assertIs(validate_reference_graphs(frozen), frozen)

        dialog = self._dialog(attachments=frozen)
        self.addCleanup(dialog.close)
        index = dialog.recipe_selector.findData("plantcv_seed_binary")
        self.assertGreaterEqual(index, 0)
        dialog.recipe_selector.setCurrentIndex(index)
        self.application.processEvents()

        self.assertIs(dialog.reference_graphs, frozen)
        self.assertTrue(dialog._active_is_transient)
        self.assertIn("NOT SAVED", dialog.attachment_label.text())
        self.assertEqual(dialog.active_attachment.recipe.recipe_id, "plantcv_seed_binary")
        self.assertEqual(
            dialog.canvas.graph.reference_recipe_identity.recipe_id,
            "plantcv_seed_binary",
        )
        selected = dialog.active_attachment.selected_stage_id
        self.assertIn(selected, dialog._facts_browser.toPlainText())

        dialog.canvas.auto_arrange()
        dialog.canvas.set_obstacle_routing_enabled(True)
        self.application.processEvents()
        self.assertEqual(dialog.active_attachment.recipe.recipe_id, "plantcv_seed_binary")
        self.assertEqual(
            dialog.canvas.graph.reference_recipe_identity.recipe_id,
            "plantcv_seed_binary",
        )
        self.assertIn(dialog.active_attachment.selected_stage_id, dialog._facts_browser.toPlainText())
        self.assertIs(dialog.reference_graphs, frozen)

    def test_small_aggregate_cap_turns_same_recipe_layout_edit_transient(self) -> None:
        from seedvision.persistence import reference_graphs as reference_graphs_module

        attachment = self._attachment(self._recipe("plantcv_seed_binary"))
        original = (attachment,)
        base_size = self._attachment_json_size(attachment)
        with patch.object(
            reference_graphs_module,
            "MAX_REFERENCE_GRAPHS_BYTES",
            base_size + 12,
        ):
            dialog = self._dialog(attachments=original)
            self.addCleanup(dialog.close)
            self.assertFalse(dialog._active_is_transient)

            dialog.canvas.auto_arrange()
            self.application.processEvents()

            self.assertTrue(dialog._active_is_transient)
            self.assertIn("aggregate limit", dialog.attachment_label.text())
            self.assertIs(dialog.reference_graphs, original)
            self.assertEqual(dialog.active_attachment.recipe.recipe_id, "plantcv_seed_binary")
            self.assertEqual(
                dialog.canvas.graph.reference_recipe_identity.recipe_id,
                "plantcv_seed_binary",
            )
            self.assertTrue(dialog.active_attachment.recipe.layout)

    def test_single_attachment_byte_cap_turns_layout_update_transient(self) -> None:
        from seedvision.persistence import reference_graphs as reference_graphs_module

        dialog = self._dialog("plantcv_seed_binary")
        self.addCleanup(dialog.close)
        original = dialog.reference_graphs
        original_size = self._attachment_json_size(original[0])

        with patch.object(
            reference_graphs_module,
            "MAX_REFERENCE_GRAPH_BYTES",
            original_size + 12,
        ):
            dialog.canvas.auto_arrange()
            self.application.processEvents()

        self.assertTrue(dialog._active_is_transient)
        self.assertIn("byte limit", dialog.attachment_label.text())
        self.assertIs(dialog.reference_graphs, original)
        self.assertEqual(dialog.active_attachment.recipe.recipe_id, "plantcv_seed_binary")
        self.assertEqual(
            dialog.canvas.graph.reference_recipe_identity.recipe_id,
            "plantcv_seed_binary",
        )

    def test_item_cap_during_layout_is_transient_and_does_not_escape_callback(self) -> None:
        from seedvision.persistence import reference_graphs as reference_graphs_module

        dialog = self._dialog("plantcv_seed_binary")
        self.addCleanup(dialog.close)
        original = dialog.reference_graphs

        with patch.object(reference_graphs_module, "MAX_REFERENCE_GRAPH_ITEMS", 0):
            dialog.canvas.auto_arrange()
            self.application.processEvents()

        self.assertTrue(dialog._active_is_transient)
        self.assertIn("item limit", dialog.attachment_label.text())
        self.assertIs(dialog.reference_graphs, original)
        self.assertEqual(dialog.active_attachment.recipe.recipe_id, "plantcv_seed_binary")
        self.assertEqual(
            dialog.canvas.graph.reference_recipe_identity.recipe_id,
            "plantcv_seed_binary",
        )

    def test_invalid_metadata_error_is_not_downgraded_to_transient_capacity(self) -> None:
        from seedvision.persistence.reference_graphs import ReferenceGraphPersistenceError

        dialog = self._dialog("plantcv_seed_binary")
        self.addCleanup(dialog.close)
        from seedvision.ui import reference_graph_dialog as dialog_module

        with patch.object(
            dialog_module,
            "validate_reference_graphs",
            side_effect=ReferenceGraphPersistenceError(
                "Reference recipe contains an invalid stage identity."
            ),
        ):
            with self.assertRaisesRegex(
                ReferenceGraphPersistenceError, "invalid stage identity"
            ):
                dialog._upsert_attachment(dialog.active_attachment)
        self.assertFalse(dialog._active_is_transient)

    def test_non_original_provenance_modes_never_claim_audited_functions(self) -> None:
        from seedvision.benchmarking.recipes import RecipeProvenanceMode

        source = self._recipe("plantcv_seed_binary")
        for mode in (RecipeProvenanceMode.NATIVE_COMPATIBLE, RecipeProvenanceMode.ENHANCED):
            with self.subTest(mode=mode.value):
                recipe = replace(
                    source,
                    provenance=replace(source.provenance, mode=mode),
                )
                attachment = self._attachment(recipe)
                dialog = self._dialog(attachments=(attachment,))
                try:
                    self.assertIn("SeedFiddle", dialog.claim_label.text())
                    self.assertNotIn("Audited original PlantCV functions", dialog.claim_label.text())
                    self.assertNotIn("Audited original PlantCV function:", dialog.claim_label.text())
                    self.assertEqual(dialog.active_attachment.recipe.provenance.mode, mode)
                finally:
                    dialog.close()

    def test_layout_selection_and_routing_update_metadata_without_analytic_change(self) -> None:
        from seedvision.persistence.reference_graphs import validate_reference_graphs
        from seedvision.segmentation.baseline import PipelineAnalysisCache

        dialog = self._dialog("plantcv_seed_binary")
        self.addCleanup(dialog.close)
        original_identity = dialog.active_attachment.recipe.identity.analytical_dict()
        graph = dialog.canvas.graph
        original_revision = graph.revision
        original_connections = graph.connections
        cache_value = object()
        cache = PipelineAnalysisCache(values={"existing-raster": cache_value})

        changed: list[tuple] = []
        dialog.reference_graphs_changed.connect(changed.append)
        other_stage = dialog.active_attachment.recipe.stages[-1].stage_id
        dialog.canvas.select_node(other_stage)
        dialog.canvas.auto_arrange()
        dialog.canvas.route_around_nodes_button.click()
        self.application.processEvents()

        exported = validate_reference_graphs(dialog.reference_graphs)
        self.assertTrue(changed)
        self.assertEqual(exported, dialog.reference_graphs)
        self.assertEqual(dialog.active_attachment.selected_stage_id, other_stage)
        self.assertTrue(dialog.active_attachment.recipe.layout)
        self.assertTrue(dialog.active_attachment.route_around_nodes)
        self.assertEqual(
            dialog.active_attachment.recipe.identity.analytical_dict(),
            original_identity,
        )
        self.assertEqual(graph.revision, original_revision)
        self.assertEqual(graph.connections, original_connections)
        self.assertEqual(cache.values["existing-raster"], cache_value)
        self.assertIs(cache.values["existing-raster"], cache_value)

    def test_view_menu_integrates_metadata_without_touching_native_cache(self) -> None:
        from PySide6.QtWidgets import QDialog

        from seedvision.benchmarking.recipe_catalogue import build_recipe_catalogue
        from seedvision.segmentation.baseline import PipelineAnalysisCache
        from seedvision.ui.reference_graph_dialog import ReferenceGraphDialog

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            window = self._main_window(root)
            try:
                view_menu = next(
                    action.menu()
                    for action in window.menuBar().actions()
                    if action.text().replace("&", "") == "View"
                )
                self.assertIn(
                    "Reference graphs…",
                    {action.text() for action in view_menu.actions()},
                )
                self.assertTrue(window.reference_graphs_action.isEnabled())

                raster = object()
                cache = PipelineAnalysisCache(values={"kept": raster})
                analysis = object()
                window._analysis_caches["cache-survival-test"] = cache
                window._analyses["cache-survival-test"] = analysis
                original_pipeline_revision = window.pipeline.revision
                window._project_tracking_enabled = True
                window._project_dirty = False

                def interact_with_view(dialog):
                    index = dialog.recipe_selector.findData("plantcv_seed_binary")
                    dialog.recipe_selector.setCurrentIndex(index)
                    recipe = next(
                        item for item in build_recipe_catalogue().recipes
                        if item.recipe_id == "plantcv_seed_binary"
                    )
                    dialog.canvas.select_node(recipe.stages[-1].stage_id)
                    dialog.canvas.auto_arrange()
                    dialog.canvas.route_around_nodes_button.click()
                    self.application.processEvents()
                    return QDialog.DialogCode.Rejected

                with patch.object(ReferenceGraphDialog, "exec", interact_with_view):
                    window.reference_graphs_action.trigger()

                self.assertEqual(
                    {item.recipe.recipe_id for item in window._project_reference_graphs},
                    {"plantcv_color_card_correct_color", "plantcv_seed_binary"},
                )
                binary = next(
                    item for item in window._project_reference_graphs
                    if item.recipe.recipe_id == "plantcv_seed_binary"
                )
                self.assertEqual(
                    binary.selected_stage_id, binary.recipe.stages[-1].stage_id
                )
                self.assertTrue(binary.recipe.layout)
                self.assertTrue(binary.route_around_nodes)
                self.assertTrue(window._project_dirty)
                self.assertEqual(window.pipeline.revision, original_pipeline_revision)
                self.assertIs(window._analysis_caches["cache-survival-test"], cache)
                self.assertIs(cache.values["kept"], raster)
                self.assertIs(window._analyses["cache-survival-test"], analysis)
            finally:
                self._dispose_main_window(window)

    def test_main_window_releases_closed_dialog_after_normal_and_exceptional_exit(self) -> None:
        from PySide6.QtCore import QCoreApplication, QEvent

        from seedvision.ui.reference_graph_dialog import ReferenceGraphDialog

        with tempfile.TemporaryDirectory() as temporary:
            window = self._main_window(Path(temporary))
            self.addCleanup(lambda: self._dispose_main_window(window))

            with patch.object(ReferenceGraphDialog, "exec", return_value=0):
                for _ in range(3):
                    window._show_reference_graphs()
                    QCoreApplication.sendPostedEvents(
                        None, QEvent.Type.DeferredDelete
                    )
                    self.application.processEvents()
                    self.assertEqual(
                        window.findChildren(ReferenceGraphDialog), []
                    )

            with patch.object(
                ReferenceGraphDialog,
                "exec",
                side_effect=RuntimeError("simulated dialog failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated dialog failure"):
                    window._show_reference_graphs()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            self.application.processEvents()
            self.assertEqual(window.findChildren(ReferenceGraphDialog), [])


if __name__ == "__main__":
    unittest.main()
