"""Exact-image correction Keep, worker handoff and reopened project pins."""
from dataclasses import replace
import os
from pathlib import Path
import shutil
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import numpy as np

from seedvision.benchmarking.native_correction_recipe import STAGE_ID
from seedvision.benchmarking.native_recipe import NativeRecipeCancelled
from seedvision.persistence import correction_inputs as assets
from seedvision.persistence.correction_session import (
    CorrectionKeepRequest, capture_correction_context, capture_correction_keep,
    validate_correction_keep_result,
)
from tests import test_project_correction_inputs as fixtures


class _Fixture:
    def setUp(self):
        # Import the fixture module, not its TestCase into discovery's namespace.
        self.fixture = fixtures.ProjectCorrectionInputTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.context = capture_correction_context(self.fixture.workspace, self.fixture.source)
        self.request = CorrectionKeepRequest(self.context, self.fixture.recipe, self.fixture.binding, 1)


class CorrectionSessionTests(_Fixture, unittest.TestCase):
    def test_context_uses_exact_locator_for_identical_bytes_and_preserves_loaded_record(self):
        f = self.fixture
        second = f.workspace/'second.png'
        shutil.copyfile(f.source, second)
        other = capture_correction_context(f.workspace, second)
        self.assertEqual(other.image.source.sha256, self.context.image.source.sha256)
        self.assertNotEqual(other.image.identifier, self.context.image.identifier)
        self.assertEqual(other.image.source.path, 'second.png')
        loaded = capture_correction_context(f.workspace, f.source, original_records=f.document.images)
        self.assertIs(loaded.image, f.document.images[0])
        f.source.write_bytes(f.source.read_bytes()[:-1]+b'x')
        with self.assertRaises(ValueError):
            capture_correction_context(f.workspace, f.source, original_records=f.document.images)

    def test_context_hashing_is_bounded_before_raster_metadata(self):
        f = self.fixture
        with patch('seedvision.benchmarking.native_correction_inputs.MAX_INPUT_FILE_BYTES', 8), \
                patch('seedvision.persistence.correction_session.read_source_raster_shape') as decode:
            with self.assertRaisesRegex(ValueError, 'byte limit'):
                capture_correction_context(f.workspace, f.source)
            decode.assert_not_called()

    def test_fresh_capture_and_pinned_capture_preserve_origins_without_fallback(self):
        captured = capture_correction_keep(self.request)
        self.assertEqual(validate_correction_keep_result(self.request, captured), captured)
        self.assertEqual(captured.input_sources, self.request.binding.input_sources)
        self.assertFalse(list(self.fixture.workspace.glob('*.seedfiddle-project.json')))
        pinned = replace(self.request, binding=captured, pinned=True)
        for port, path in self.fixture.paths.items():
            path.unlink()
        self.assertEqual(capture_correction_keep(pinned), captured)
        asset = captured.inputs[0]
        pin = self.fixture.workspace/asset.path
        pin.unlink()
        # Recreate the original; a reopened request must still reject the pin.
        original = Path(asset.origin_path)
        shutil.copyfile(self.fixture.source, original)
        with self.assertRaises(OSError):
            capture_correction_keep(pinned)

    def test_last_callback_source_or_pin_mutation_is_rejected(self):
        capture_correction_keep(self.request)
        count = 0
        def count_calls():
            nonlocal count
            count += 1
            return False
        capture_correction_keep(self.request, cancel=count_calls)
        for kind in ('source', 'pin'):
            with self.subTest(kind=kind):
                path = (self.fixture.source if kind == 'source' else
                        self.fixture.workspace/self.request.binding.inputs[0].path)
                original = path.read_bytes()
                calls = 0
                def mutate_last():
                    nonlocal calls
                    calls += 1
                    if calls == count:
                        path.write_bytes(original[:-1]+bytes([original[-1] ^ 1]))
                    return False
                try:
                    with self.assertRaisesRegex(ValueError, 'identity|source|bytes'):
                        capture_correction_keep(self.request, cancel=mutate_last)
                    self.assertEqual(calls, count)
                finally:
                    path.write_bytes(original)

    def test_cancellation_and_wrong_capture_result_do_not_publish_master(self):
        with self.assertRaises(NativeRecipeCancelled):
            capture_correction_keep(self.request, cancel=lambda: True)
        captured = capture_correction_keep(self.request)
        bad = replace(captured, inputs=tuple(replace(item, decode=item.decode+' changed')
                                             for item in captured.inputs))
        with self.assertRaisesRegex(ValueError, 'frozen selection'):
            validate_correction_keep_result(self.request, bad)
        self.assertFalse(list(self.fixture.workspace.glob('*.seedfiddle-project.json')))


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for actual correction runs')
class CorrectionProjectDialogTests(_Fixture, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        from PySide6.QtWidgets import QApplication
        cls.application = QApplication.instance() or QApplication([])

    def wait_until(self, predicate, timeout=40):
        end = time.monotonic()+timeout
        while not predicate() and time.monotonic() < end:
            self.application.processEvents()
            time.sleep(.01)
        self.application.processEvents()
        self.assertTrue(predicate(), 'Correction worker did not finish.')

    def dialog(self, *, context=True, binding=None, callback=None, run=False):
        from seedvision.ui.native_correction_dialog import NativeCorrectionDialog
        dialog = NativeCorrectionDialog(self.fixture.recipe,
            project_context=self.context if context else None, project_binding=binding,
            keep_in_project=callback or (lambda request, cancel, current:
                capture_correction_keep(request, cancel=cancel.is_set)))
        if binding is None:
            for role, path in (self.fixture.paths | {'source_rgb': self.fixture.source}).items():
                dialog.set_input_path(role, path)
        dialog.show()
        self.addCleanup(self.dispose_dialog, dialog)
        if run:
            dialog.start_run()
            self.wait_until(lambda: dialog._worker is None)
            self.assertIsNotNone(dialog.result, dialog.status_label.text())
        return dialog

    def dispose_dialog(self, dialog):
        dialog.reject()
        self.wait_until(lambda: dialog._worker is None and not dialog._keeping)
        dialog.deleteLater()
        self.application.processEvents()

    def window(self):
        from tests import test_project_analysis_ui as ui
        f = self.fixture
        (f.workspace/'config').mkdir(exist_ok=True)
        shutil.copyfile(Path(__file__).parents[1]/'config'/'traits.json', f.workspace/'config'/'traits.json')
        ui.ProjectAnalysisUiTests.application = self.application
        window = ui.ProjectAnalysisUiTests._window(f.workspace)
        self.addCleanup(ui.ProjectAnalysisUiTests._dispose, window)
        window._add_image(f.source, open_now=True, mark_project_dirty=False)
        return window

    def test_keep_hidden_without_context_and_enabled_only_for_complete_current_run(self):
        from seedvision.ui.native_correction_dialog import _Completion
        local = self.dialog(context=False, run=True)
        self.assertTrue(local.project_button.isHidden())
        self.assertFalse(local.project_button.isEnabled())
        dialog = self.dialog()
        self.assertTrue(dialog.project_button.isVisible())
        self.assertFalse(dialog.project_button.isEnabled())
        dialog.start_run()
        self.wait_until(lambda: dialog._worker is None)
        result = dialog.result
        self.assertTrue(dialog.project_button.isEnabled(), dialog.status_label.text())
        for invalid in (replace(result, status='partial'), replace(result, nodes=()),
                        replace(result, nodes=(replace(result.nodes[0], status='failed'),)),
                        ValueError('Failed run')):
            dialog._completed(_Completion(invalid, None, dialog._generation))
            self.assertFalse(dialog.project_button.isEnabled())
        dialog._completed(_Completion(result, None, dialog._generation))
        dialog.set_input_path('source_mask', self.fixture.paths['source_mask'])
        self.assertIsNone(dialog.result)
        self.assertFalse(dialog.project_button.isEnabled())

    def test_same_bytes_at_other_source_path_cannot_keep_for_selected_project_image(self):
        dialog = self.dialog()
        dialog.set_input_path('source_rgb', self.fixture.paths['source_rgb'])
        dialog.start_run()
        self.wait_until(lambda: dialog._worker is None)
        self.assertIsNotNone(dialog.result)
        self.assertFalse(dialog.project_button.isEnabled())
        with self.assertRaisesRegex(ValueError, 'exact selected project image'):
            dialog._current_keep_binding()

    def test_old_completion_cannot_release_newer_worker_or_restore_keep(self):
        from seedvision.ui.native_correction_dialog import _Completion
        dialog = self.dialog(run=True)
        old = dialog._generation
        dialog._invalidate_view(STAGE_ID)
        newer = SimpleNamespace(generation=dialog._generation)
        dialog._worker = newer
        dialog._completed(_Completion(ValueError('old'), None, old))
        self.assertIs(dialog._worker, newer)
        self.assertFalse(dialog.project_button.isEnabled())
        dialog._worker = None

    def test_keep_captures_only_compact_request_and_current_canvas_layout(self):
        seen = []
        def keep(request, cancel, current):
            self.assertTrue(current())
            seen.append(request)
            self.assertTrue(all(not button.isEnabled() for button in dialog._input_buttons.values()))
            return capture_correction_keep(request, cancel=cancel.is_set)
        dialog = self.dialog(callback=keep, run=True)
        dialog.canvas.node_items[STAGE_ID].setPos(220, 310)
        dialog._keep_recipe_in_project()
        self.assertEqual(len(seen), 1, dialog.status_label.text())
        request = seen[0]
        self.assertEqual(request.recipe.layout[0].x, 220)
        self.assertEqual(request.recipe.layout[0].y, 310)
        self.assertEqual(request.binding.image_id, self.context.image.identifier)
        self.assertFalse(hasattr(request, 'result'))
        self.assertIn('kept in the project', dialog.status_label.text())
        self.assertTrue(dialog.project_button.isEnabled())

    def test_wrong_callback_binding_is_rejected_by_dialog(self):
        def bad(request, cancel, current):
            return replace(request.binding, inputs=tuple(replace(item, decode=item.decode+' changed')
                                                         for item in request.binding.inputs))
        dialog = self.dialog(callback=bad, run=True)
        dialog._keep_recipe_in_project()
        self.assertIn('frozen selection', dialog.status_label.text())

    def test_malformed_output_completions_clear_preview_export_and_keep(self):
        from seedvision.ui.native_correction_dialog import _Completion
        seen = []
        dialog = self.dialog(callback=lambda *args: seen.append(args), run=True)
        result = dialog.result
        outputs = dict(result.nodes[0].outputs)
        immutable = lambda value: np.frombuffer(np.ascontiguousarray(value).tobytes(),
                                               dtype=value.dtype).reshape(value.shape)
        nonfinite = outputs['target_matrix'].copy()
        nonfinite[0, 0] = np.nan
        cases = {
            'missing': {key: value for key, value in outputs.items() if key != 'target_matrix'},
            'cpu': outputs | {'corrected_image': outputs['corrected_image'].cpu()},
            'dtype': outputs | {'corrected_image': outputs['corrected_image'].float()},
            'shape': outputs | {'corrected_image': outputs['corrected_image'][:1]},
            'layout': outputs | {'corrected_image': outputs['corrected_image'].to_sparse()},
            'writable': outputs | {'target_matrix': outputs['target_matrix'].copy()},
            'nonfinite': outputs | {'target_matrix': immutable(nonfinite)},
            'oversized': outputs | {'source_matrix': immutable(np.zeros((256, 4), np.float64))},
            'transform': outputs | {'transformation_matrix': immutable(np.zeros((8, 8), np.float64))},
        }
        invalid_results = [replace(result, nodes=(replace(result.nodes[0], outputs=value),))
                           for value in cases.values()]
        invalid_results += [replace(result, nodes=(replace(result.nodes[0], operation_id='wrong'),)),
                            replace(result, nodes=(replace(result.nodes[0], parameters={'unused': 1}),))]
        for invalid in invalid_results:
            with self.subTest(outputs=tuple(invalid.nodes[0].outputs)):
                dialog._completed(_Completion(invalid, dialog._prepared, dialog._generation))
                self.assertIsNone(dialog.result)
                self.assertIsNone(dialog._prepared)
                self.assertFalse(dialog.export_button.isEnabled())
                self.assertFalse(dialog.project_button.isEnabled())
                dialog._keep_recipe_in_project()
                self.assertEqual(seen, [])

    def test_reopened_mask_edit_rerun_and_keep_uses_exact_project_source(self):
        window = self.window()
        captured = capture_correction_keep(self.request)
        window._project_native_recipes = (self.fixture.recipe,)
        window._project_correction_bindings = (captured,)
        from tests import test_project_analysis_ui as ui
        history = (ui.ProjectAnalysisUiTests._reference_graph_attachment(),)
        window._project_reference_graphs = history
        for path in self.fixture.paths.values():
            path.unlink()
        dialog = self.dialog(binding=captured, callback=window._keep_project_correction, run=True)
        new_mask = self.fixture.workspace/'new-mask.png'
        shutil.copyfile(self.fixture.workspace/captured.inputs[3].path, new_mask)
        dialog.set_input_path('source_mask', new_mask)
        self.assertIsNone(dialog._project_binding)
        self.assertEqual(dialog._paths['source_rgb'], self.context.source_path)
        dialog.start_run()
        self.wait_until(lambda: dialog._worker is None)
        self.assertIsNotNone(dialog.result, dialog.status_label.text())
        self.assertIsNone(dialog.result.input_origins)
        self.assertTrue(dialog.project_button.isEnabled(), dialog.status_label.text())
        dialog._keep_recipe_in_project()
        self.assertIn('kept in the project', dialog.status_label.text())
        self.assertEqual(window._project_correction_bindings[0].image_id, self.context.image.identifier)
        self.assertEqual(window._project_reference_graphs, history)
        self.assertEqual(window._project_correction_bindings[0].input_sources['source_mask']['path'], str(new_mask))
        # An explicit source choice remains authoritative after later mask edits.
        pinned_again = self.dialog(binding=window._project_correction_bindings[0])
        other = self.fixture.workspace/'same-bytes-other.png'
        shutil.copyfile(self.fixture.source, other)
        pinned_again.set_input_path('source_rgb', other)
        pinned_again.set_input_path('source_mask', new_mask)
        self.assertEqual(pinned_again._paths['source_rgb'], other)

    def test_reopened_run_reads_pins_preserves_origins_and_rejects_missing_pin(self):
        captured = capture_correction_keep(self.request)
        for path in self.fixture.paths.values():
            path.unlink()
        dialog = self.dialog(binding=captured, run=True)
        self.assertEqual(dialog.result.input_origins.binding, captured)
        self.assertTrue(dialog.project_button.isEnabled(), dialog.status_label.text())
        for asset in captured.inputs:
            actual = dialog.result.input_sources[asset.role]
            self.assertEqual(Path(actual['path']), self.fixture.workspace/asset.path)
            self.assertEqual(actual['sha256'], asset.source_sha256)
        dialog._keep_recipe_in_project()
        self.assertIn('kept in the project', dialog.status_label.text())
        (self.fixture.workspace/captured.inputs[0].path).unlink()
        dialog.start_run()
        self.wait_until(lambda: dialog._worker is None)
        self.assertIsNone(dialog.result)
        self.assertFalse(dialog.project_button.isEnabled())

    def test_actual_window_keep_registers_unsaved_image_then_save_and_reopen(self):
        window = self.window()
        self.context = window._capture_correction_project_context()
        self.assertEqual(window._project_original_image_records, {})
        self.assertFalse(window._project_dirty)
        dialog = self.dialog(callback=window._keep_project_correction, run=True)
        dialog._keep_recipe_in_project()
        self.assertIn('kept in the project', dialog.status_label.text())
        self.assertTrue(window._project_dirty)
        self.assertEqual(tuple(window._project_original_image_records), (self.context.image.identifier,))
        self.assertEqual(len(window._project_native_recipes), 1)
        self.assertEqual(window._project_correction_bindings[0].input_sources, dialog._current_keep_binding().input_sources)
        master = self.fixture.workspace/'kept.seedfiddle-project.json'
        self.assertFalse(master.exists())
        self.assertTrue(window._write_project(master))
        saved = self.fixture.store.load(master)
        self.assertFalse(saved.issues)
        self.assertEqual(saved.document.correction_bindings, window._project_correction_bindings)
        with patch.object(window, '_confirm_project_replacement', return_value=True):
            self.assertTrue(window._open_project(master))
        self.context = window._capture_correction_project_context()
        reopened = self.dialog(binding=window._project_correction_bindings[0], callback=window._keep_project_correction, run=True)
        self.assertEqual(reopened.result.input_origins.binding, saved.document.correction_bindings[0])

    def test_window_preserves_distinct_same_byte_image_binding_and_history(self):
        window = self.window()
        first = capture_correction_keep(self.request)
        window._project_native_recipes = (self.fixture.recipe,)
        window._project_correction_bindings = (first,)
        window._project_original_image_records = {self.context.image.identifier: self.context.image}
        window._project_original_image_order = (self.context.image.identifier,)
        second = self.fixture.workspace/'second.png'
        shutil.copyfile(self.fixture.source, second)
        window._add_image(second, open_now=True, mark_project_dirty=False)
        context = window._capture_correction_project_context()
        binding = replace(self.fixture.binding, image_id=context.image.identifier)
        request = CorrectionKeepRequest(context, self.fixture.recipe, binding, 2)
        history = window._project_reference_graphs
        captured = window._keep_project_correction(request, threading.Event(), lambda: True)
        self.assertEqual(window._project_correction_bindings, (first, captured))
        self.assertNotEqual(first.image_id, captured.image_id)
        self.assertEqual(window._project_reference_graphs, history)

    def test_window_rejects_capture_failure_stale_state_cancel_and_invalid_result(self):
        window = self.window()
        context = window._capture_correction_project_context()
        request = replace(self.request, context=context)
        captured = capture_correction_keep(request)
        master = self.fixture.store.save(self.fixture.document, self.fixture.workspace/'unchanged.seedfiddle-project.json')
        window._current_project_path = master
        master_bytes = master.read_bytes()
        # Exercise handoff rather than re-test the blob copier for each state.
        for kind in ('failure', 'cancel', 'root', 'image', 'recipes', 'closing', 'master', 'record', 'invalid'):
            with self.subTest(kind=kind):
                old_root, old_image = window._root, window.image_view.image_path
                old_recipes = window._project_native_recipes
                old_closing = getattr(window, '_closing_after_work', False)
                old_master = window._current_project_path
                before_bindings = window._project_correction_bindings
                before_records = dict(window._project_original_image_records)
                before_order = window._project_original_image_order
                before_dirty = window._project_dirty
                cancel = threading.Event()
                def capture(_window, callback, title):
                    if kind == 'failure':
                        raise ValueError('capture failed')
                    if kind == 'cancel':
                        cancel.set()
                    elif kind == 'root':
                        window._root = self.fixture.root
                    elif kind == 'image':
                        window.image_view._image_path = self.fixture.paths['source_rgb']
                    elif kind == 'recipes':
                        window._project_native_recipes = (self.fixture.recipe,)
                    elif kind == 'closing':
                        window._closing_after_work = True
                    elif kind == 'master':
                        window._current_project_path = self.fixture.workspace/'different.seedfiddle-project.json'
                    elif kind == 'record':
                        window._project_original_image_records = {context.image.identifier: context.image}
                    elif kind == 'invalid':
                        return replace(captured, inputs=tuple(replace(item, decode=item.decode+' changed')
                                                             for item in captured.inputs))
                    return captured
                try:
                    with patch('seedvision.ui.main_window.run_persistence', side_effect=capture):
                        with self.assertRaises((ValueError, NativeRecipeCancelled)):
                            window._keep_project_correction(request, cancel, lambda: True)
                    self.assertEqual(window._project_correction_bindings, before_bindings)
                    if kind != 'record':
                        self.assertEqual(window._project_original_image_records, before_records)
                    self.assertEqual(window._project_original_image_order, before_order)
                    self.assertEqual(window._project_dirty, before_dirty)
                    self.assertEqual(master.read_bytes(), master_bytes)
                finally:
                    window._root, window.image_view._image_path = old_root, old_image
                    window._project_native_recipes = old_recipes
                    window._closing_after_work = old_closing
                    window._current_project_path = old_master
                    window._project_original_image_records = before_records

    def test_nested_worker_layout_parameter_and_close_cancel_without_session_merge(self):
        from PySide6.QtCore import QTimer
        for kind in ('layout', 'parameter', 'close'):
            with self.subTest(kind=kind):
                window = self.window()
                self.context = window._capture_correction_project_context()
                dialog = self.dialog(callback=window._keep_project_correction, run=True)
                entered, release = threading.Event(), threading.Event()
                real_capture = capture_correction_keep
                def controlled(request, *, cancel):
                    entered.set()
                    if not release.wait(10):
                        raise AssertionError('Keep test did not release its worker.')
                    return real_capture(request, cancel=cancel)
                def change():
                    self.assertTrue(entered.wait(2))
                    self.assertTrue(dialog._keeping)
                    if kind == 'layout':
                        dialog.canvas.node_items[STAGE_ID].setPos(430, 250)
                        dialog.canvas.layout_changed.emit()
                    elif kind == 'parameter':
                        dialog._invalidate_view(STAGE_ID)
                    else:
                        dialog.close()
                        self.assertTrue(dialog.isVisible())
                    release.set()
                with patch('seedvision.persistence.correction_session.capture_correction_keep', side_effect=controlled):
                    QTimer.singleShot(0, change)
                    dialog._keep_recipe_in_project()
                self.assertFalse(dialog._keeping)
                self.assertIsNone(window._persistence_job)
                self.assertEqual(window._project_native_recipes, ())
                self.assertEqual(window._project_correction_bindings, ())
                self.assertEqual(window._project_original_image_records, {})
                self.assertFalse(window._project_dirty)
                if kind == 'close':
                    self.assertFalse(dialog.isVisible())

    def test_window_does_not_restore_binding_removed_during_pinned_capture(self):
        window = self.window()
        captured = capture_correction_keep(self.request)
        window._project_native_recipes = (self.fixture.recipe,)
        window._project_correction_bindings = (captured,)
        request = replace(self.request, binding=captured, pinned=True)
        def replaced(_window, callback, title):
            window._project_correction_bindings = ()
            return captured
        with patch('seedvision.ui.main_window.run_persistence', side_effect=replaced):
            with self.assertRaises(NativeRecipeCancelled):
                window._keep_project_correction(request, threading.Event(), lambda: True)
        self.assertEqual(window._project_correction_bindings, ())
        self.assertEqual(window._project_original_image_records, {})
        self.assertFalse(window._project_dirty)

    def test_window_calls_freshness_closure_before_capture_only_and_rejects_replaced_image(self):
        window = self.window()
        calls = 0
        def current():
            nonlocal calls
            calls += 1
            if calls > 1:
                raise AssertionError('Caller closure must not run after the final file audit.')
            return True
        captured = window._keep_project_correction(self.request, threading.Event(), current)
        self.assertEqual(calls, 1)
        self.assertEqual(window._project_correction_bindings, (captured,))
        changed = replace(self.context.image, source=replace(self.context.image.source, sha256='0'*64))
        window._project_original_image_records[self.context.image.identifier] = changed
        with patch('seedvision.ui.main_window.run_persistence') as worker:
            with self.assertRaisesRegex(ValueError, 'image record changed'):
                window._keep_project_correction(self.request, threading.Event(), lambda: True)
            worker.assert_not_called()

    def test_context_factory_discards_project_change_during_metadata_capture(self):
        window = self.window()
        def changed(_window, callback, title):
            context = callback()
            window._current_project_path = self.fixture.workspace/'other.seedfiddle-project.json'
            return context
        with patch('seedvision.ui.main_window.run_persistence', side_effect=changed):
            with self.assertRaisesRegex(ValueError, 'project or selected image changed'):
                window._capture_correction_project_context()
        self.assertEqual(window._project_original_image_records, {})
        self.assertEqual(window._project_correction_bindings, ())
        self.assertFalse(window._project_dirty)

    def test_reference_dialog_routes_correction_and_looks_up_exact_image_binding(self):
        from seedvision.ui.reference_graph_dialog import ReferenceGraphDialog
        from seedvision.benchmarking.recipe_catalogue import build_recipe_catalogue
        captured = capture_correction_keep(self.request)
        reference = ReferenceGraphDialog(project_root=self.fixture.workspace,
            native_recipes=(self.fixture.recipe,), correction_bindings=(captured,),
            correction_context_factory=lambda: self.context,
            keep_correction=lambda request, cancel, current: captured)
        self.addCleanup(reference.deleteLater)
        with patch('seedvision.ui.native_correction_dialog.NativeCorrectionDialog') as constructor:
            reference._open_saved_native_recipe(self.fixture.recipe)
            self.assertEqual(constructor.call_args.kwargs['project_binding'], captured)
            self.assertIs(constructor.call_args.kwargs['project_context'], self.context)
            canonical = build_recipe_catalogue().recipe(self.fixture.recipe.recipe_id)
            reference._open_correction_recipe(canonical)
            self.assertEqual(constructor.call_args.kwargs['project_binding'], captured)
            reference.recipe_selector.setCurrentIndex(reference.recipe_selector.findData(self.fixture.recipe.recipe_id))
            reference._open_native_recipe()
            self.assertEqual(constructor.call_args.kwargs['project_binding'], captured)
            second = self.fixture.workspace/'other.png'
            shutil.copyfile(self.fixture.source, second)
            reference._correction_context_factory = lambda: capture_correction_context(self.fixture.workspace, second)
            reference._open_saved_native_recipe(self.fixture.recipe)
            self.assertIsNone(constructor.call_args.kwargs['project_binding'])
        reference._remove_native_recipe((self.fixture.recipe.recipe_id, self.fixture.recipe.recipe_version))
        self.assertEqual(reference._correction_bindings, ())
