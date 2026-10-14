"""Per-image correction selections, byte-bound copies and portable projects."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import io
import inspect
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import cv2
import torch

from seedvision.benchmarking.checkpoints import CoordinateFrame
from seedvision.benchmarking.native_correction_recipe import build_native_correction_recipe, STAGE_ID
from seedvision.benchmarking.recipes import LayoutPosition
from seedvision.persistence import correction_inputs as assets
from seedvision.persistence.analysis_settings import analysis_settings_profile_from_graph
from seedvision.persistence.project_analysis import (
    ProjectAnalysisStore, ProjectImageSpec, InvalidProjectAnalysis, ProjectFileStatus,
)
from seedvision.persistence.snapshot import create_snapshot, restore_snapshot
from seedvision.pipeline import build_default_pipeline
from tests.test_native_correction_dialog import _write_inputs


class ProjectCorrectionInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        origin = self.root/'origin'
        origin.mkdir()
        self.paths = _write_inputs(origin)
        self.workspace = self.root/'workspace'
        self.workspace.mkdir()
        self.source = self.workspace/'source.png'
        shutil.copyfile(self.paths['source_rgb'], self.source)
        self.recipe = build_native_correction_recipe()
        self.store = ProjectAnalysisStore(self.workspace)
        self.profile = analysis_settings_profile_from_graph(build_default_pipeline())
        self.document = self.store.capture(analysis_settings=self.profile,
            images=(ProjectImageSpec(self.source, (17, 25)),), native_recipes=(self.recipe,))
        self.binding = self.make_binding(self.document.images[0].identifier)

    def tearDown(self):
        self.temp.cleanup()

    def make_binding(self, image_id, paths=None):
        # Declared prior preparation metadata: capture verifies bytes, not decode.
        paths = self.paths if paths is None else paths
        inputs = []
        for role in assets.PORTS:
            path = paths[role]
            digest = sha256(path.read_bytes()).hexdigest()
            frame = CoordinateFrame('target.decoded' if role.startswith('target_') else 'source.decoded',
                                    (17, 25), 'top-left', 'x-right,y-down', 'pixel')
            inputs.append(assets.CorrectionInputAsset(role, str(path.resolve()), digest, path.stat().st_size,
                frame, 'Declared prior preparation for '+role, 'native-correction/'+digest+'.image'))
        return assets.ProjectCorrectionBinding(image_id, self.recipe.recipe_id, self.recipe.recipe_version,
                                                self.recipe.identity.recipe_sha256, tuple(inputs))

    def project(self, bindings=None, images=None):
        bindings = (self.binding,) if bindings is None else bindings
        images = self.document.images if images is None else images
        captured = assets.capture_correction_bindings(self.workspace, bindings, images,
                                                      native_recipes=(self.recipe,), existing=())
        document = replace(self.document, images=images, correction_bindings=captured)
        master = self.store.save(document, self.workspace/'correction.seedfiddle-project.json')
        return master, document

    def test_schema6_roundtrip_preserves_four_origins_and_distinct_transport(self):
        master, document = self.project()
        loaded = self.store.load(master)
        self.assertFalse(loaded.issues)
        from seedvision.persistence.project_analysis import _document_payload
        self.assertEqual(_document_payload(loaded.document), _document_payload(document))
        self.assertEqual(loaded.document.correction_bindings, document.correction_bindings)
        self.assertEqual(json.loads(master.read_bytes())['version'], 6)
        self.assertLess(master.stat().st_size, 100_000)
        binding = document.correction_bindings[0]
        self.assertEqual(tuple(item.role for item in binding.inputs), assets.PORTS)
        self.assertEqual(binding.input_sources, self.binding.input_sources)
        self.assertEqual(binding.selection_sha256, self.binding.selection_sha256)
        for asset in binding.inputs:
            self.assertNotEqual(asset.origin_path, str(self.workspace/asset.path))
            self.assertEqual(sha256((self.workspace/asset.path).read_bytes()).hexdigest(), asset.source_sha256)
        self.assertEqual(len(list((self.workspace/'native-correction').iterdir())), 3)

    def test_schemas1_through5_migrate_empty_without_invented_input_bindings(self):
        master = self.store.save(self.document, self.workspace/'empty.seedfiddle-project.json')
        current = json.loads(master.read_bytes())
        for version in range(1, 6):
            value = deepcopy(current)
            value['version'] = version
            value.pop('correction_bindings')
            if version < 5:
                value.pop('native_recipes')
            if version < 4:
                value.pop('reference_model_assets')
            if version < 3:
                value.pop('reference_graphs')
            if version == 1:
                value['project'] = {key: value['project'][key] for key in ('species', 'selected_image')}
                for image in value['images']:
                    image.pop('biological_context')
                    image.pop('capture_group_id')
            with self.subTest(version=version):
                master.write_text(json.dumps(value), encoding='utf-8')
                document = self.store.load(master).document
                self.assertEqual(document.correction_bindings, ())
                output = self.store.save(document, self.workspace/'migrated.seedfiddle-project.json')
                self.assertEqual(json.loads(output.read_bytes())['correction_bindings'], [])
                value['correction_bindings'] = []
                master.write_text(json.dumps(value), encoding='utf-8')
                with self.assertRaisesRegex(InvalidProjectAnalysis, 'unknown field'):
                    self.store.load(master)
        value = deepcopy(current)
        value.pop('correction_bindings')
        master.write_text(json.dumps(value), encoding='utf-8')
        with self.assertRaises(InvalidProjectAnalysis):
            self.store.load(master)

    def test_source_sha_frame_and_image_id_are_exact_before_asset_reads(self):
        image = self.document.images[0]
        cases = ((replace(self.binding, image_id='other-image'),),
                 (self.binding, self.binding),)
        for bindings in cases:
            with self.subTest(bindings=len(bindings)), patch.object(assets, 'copy_verified_input') as read:
                with self.assertRaises(ValueError):
                    assets.capture_correction_bindings(self.workspace, bindings, self.document.images,
                                                        native_recipes=(self.recipe,))
                read.assert_not_called()
        for record in (replace(image, source=replace(image.source, sha256='f'*64)),
                       replace(image, source_shape=(17, 24))):
            with patch.object(assets, 'copy_verified_input') as read:
                with self.assertRaisesRegex(ValueError, 'exact project image'):
                    assets.capture_correction_bindings(self.workspace, (self.binding,), (record,),
                                                        native_recipes=(self.recipe,))
                read.assert_not_called()

    def test_four_ordered_roles_strict_fields_and_canonical_frames(self):
        payload = self.binding.to_dict()
        malformed = []
        for field, value in (('source_bytes', True), ('path', '../escape.png'),
                             ('origin_path', 'relative.png'), ('role', 'fifth_role')):
            current = deepcopy(payload)
            current['inputs'][0][field] = value
            malformed.append(current)
        current = deepcopy(payload)
        current['inputs'].reverse()
        malformed.append(current)
        current = deepcopy(payload)
        current['inputs'][0]['frame']['shape'] = [17, 24]
        malformed.append(current)
        current = deepcopy(payload)
        current['inputs'][0]['frame']['transform_sha256'] = 'a'*64
        malformed.append(current)
        current = deepcopy(payload)
        current['inputs'][0]['unknown'] = 1
        malformed.append(current)
        current = deepcopy(payload)
        current['unknown'] = 1
        malformed.append(current)
        for index, value in enumerate(malformed):
            with self.subTest(index=index), self.assertRaises((ValueError, TypeError)):
                assets.ProjectCorrectionBinding.from_dict(value)

    def test_layout_and_presentation_preserve_bindings_but_removed_selection_does_not(self):
        changed = replace(self.recipe, title='Edited presentation',
                          layout=(LayoutPosition(STAGE_ID, 570., 220.),))
        self.assertEqual(changed.identity.recipe_sha256, self.recipe.identity.recipe_sha256)
        self.assertEqual(assets.validate_correction_bindings((self.binding,), self.document.images,
                                                            native_recipes=(changed,)), (self.binding,))
        self.assertEqual(assets.retain_selected_correction_bindings((self.binding,), native_recipes=(changed,)),
                         (self.binding,))
        self.assertEqual(assets.retain_selected_correction_bindings((self.binding,), native_recipes=()), ())
        with self.assertRaises(ValueError):
            assets.validate_correction_bindings((self.binding,), self.document.images)

    def test_count_file_and_aggregate_bounds_precede_reads(self):
        unique = {(item.source_sha256, item.source_bytes) for item in self.binding.inputs}
        total = sum(size for _, size in unique)
        with patch.object(assets, 'MAX_TOTAL_INPUT_BYTES', total):
            self.project()
        for name, limit in (('MAX_TOTAL_INPUT_BYTES', total-1), ('MAX_CORRECTION_BINDINGS', 0),
                            ('MAX_INPUT_BYTES', min(item.source_bytes for item in self.binding.inputs)-1)):
            with self.subTest(name=name), patch.object(assets, name, limit), \
                    patch.object(assets, 'copy_verified_input') as read:
                with self.assertRaises(ValueError):
                    assets.capture_correction_bindings(self.workspace, (self.binding,), self.document.images,
                                                        native_recipes=(self.recipe,))
                read.assert_not_called()

    def test_conflicting_transport_identity_rejects_before_reads(self):
        inputs = list(self.binding.inputs)
        inputs[1] = replace(inputs[1], path=inputs[0].path)
        binding = replace(self.binding, inputs=tuple(inputs))
        with patch.object(assets, 'copy_verified_input') as read:
            with self.assertRaisesRegex(ValueError, 'conflicting byte'):
                assets.capture_correction_bindings(self.workspace, (binding,), self.document.images,
                                                    native_recipes=(self.recipe,))
            read.assert_not_called()

    def test_growing_input_reads_at_most_expected_size_plus_one(self):
        source = self.root/'eight.bytes'
        source.write_bytes(b'x'*8)
        asset = replace(self.binding.inputs[0], source_bytes=8, source_sha256=sha256(b'x'*8).hexdigest())
        class Stream(io.BytesIO):
            def read(self, count=-1):
                self.requests.append(count)
                return super().read(count)
        stream = Stream(b'x'*100)
        stream.requests = []
        with patch.object(Path, 'open', return_value=stream), patch.object(assets, 'COPY_CHUNK_BYTES', 3):
            with self.assertRaisesRegex(ValueError, 'grew'):
                assets.copy_verified_input(source, asset)
        self.assertEqual(stream.requests, [3, 3, 3])

    def test_retained_capture_uses_only_pinned_transport_after_original_deletion(self):
        master, document = self.project()
        for path in self.paths.values():
            path.unlink()
        captured = self.store.capture(analysis_settings=self.profile,
            images=(ProjectImageSpec(self.source, (17, 25)),), native_recipes=(self.recipe,),
            correction_bindings=document.correction_bindings,
            existing_correction_bindings=document.correction_bindings)
        self.assertEqual(captured.correction_bindings, document.correction_bindings)
        self.store.save(captured, master)
        self.assertFalse(self.store.load(master).issues)

    def test_corrupt_pinned_transport_never_falls_back_and_master_stays_unchanged(self):
        master, document = self.project()
        original = master.read_bytes()
        asset = document.correction_bindings[0].inputs[0]
        path = self.workspace/asset.path
        path.write_bytes(b'x'*asset.source_bytes)
        self.assertTrue(Path(asset.origin_path).exists())
        with self.assertRaises(InvalidProjectAnalysis):
            self.store.capture_and_save(analysis_settings=self.profile,
                images=(ProjectImageSpec(self.source, (17, 25)),), native_recipes=(self.recipe,),
                correction_bindings=document.correction_bindings,
                existing_correction_bindings=document.correction_bindings, destination=master)
        self.assertEqual(master.read_bytes(), original)
        loaded = self.store.load(master)
        self.assertEqual(loaded.issues[0].role, 'correction_input:target_rgb')
        self.assertEqual(loaded.issues[0].status, ProjectFileStatus.FINGERPRINT_MISMATCH)
        with patch('seedvision.persistence.project_analysis.copy_verified_input') as read:
            self.assertEqual(self.store.load(master, verify_files=False).document.correction_bindings,
                             document.correction_bindings)
            read.assert_not_called()

    def test_interrupted_capture_cleans_temporary_files_and_cannot_publish_master(self):
        master = self.store.save(self.document, self.workspace/'prior.seedfiddle-project.json')
        original = master.read_bytes()
        calls = 0
        def cancel():
            nonlocal calls
            calls += 1
            return calls >= 8
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            assets.capture_correction_bindings(self.workspace, (self.binding,), self.document.images,
                                                native_recipes=(self.recipe,), existing=(), cancel=cancel)
        self.assertEqual(master.read_bytes(), original)
        self.assertFalse(list(self.workspace.rglob('*.incomplete')))

    def test_final_callback_cannot_publish_changed_asset_bytes(self):
        master, document = self.project()
        path = self.workspace/document.correction_bindings[0].inputs[0].path
        original = master.read_bytes()
        # Inject at the final callback, immediately before the no-callback audit.
        changed = False
        def cancel():
            nonlocal changed
            caller = inspect.currentframe().f_back.f_back
            if caller.f_code.co_name == 'capture_correction_bindings' and 'result' in caller.f_locals:
                path.write_bytes(b'x'*path.stat().st_size)
                changed = True
            return False
        with self.assertRaises(ValueError):
            assets.capture_correction_bindings(self.workspace, document.correction_bindings,
                document.images, native_recipes=(self.recipe,), existing=document.correction_bindings,
                cancel=cancel)
        self.assertTrue(changed)
        self.assertEqual(master.read_bytes(), original)

    def test_standalone_copy_rejects_same_size_change_in_final_callback(self):
        source = self.root/'eight.bytes'
        source.write_bytes(b'x'*8)
        asset = replace(self.binding.inputs[0], source_bytes=8, source_sha256=sha256(b'x'*8).hexdigest())
        calls = 0
        def cancel():
            nonlocal calls
            calls += 1
            if calls == 4:
                source.write_bytes(b'y'*8)
            return False
        with self.assertRaisesRegex(ValueError, 'selected byte identity'):
            assets.copy_verified_input(source, asset, cancel=cancel)
        self.assertEqual(calls, 4)

    def test_missing_pinned_asset_is_reported_without_original_fallback(self):
        master, document = self.project()
        asset = document.correction_bindings[0].inputs[0]
        (self.workspace/asset.path).unlink()
        self.assertTrue(Path(asset.origin_path).is_file())
        loaded = self.store.load(master)
        self.assertEqual(loaded.issues[0].status, ProjectFileStatus.MISSING)
        with self.assertRaises(FileNotFoundError):
            assets.capture_correction_bindings(self.workspace, document.correction_bindings, document.images,
                                               native_recipes=(self.recipe,), existing=document.correction_bindings)

    def test_omitted_prior_state_keeps_missing_pins_and_never_reads_origins(self):
        master, document = self.project()
        asset = document.correction_bindings[0].inputs[0]
        (self.workspace/asset.path).unlink()
        original = master.read_bytes()
        with patch.object(assets, 'resolve_model_path', side_effect=AssertionError('No origin fallback')) as origin:
            with self.assertRaises(FileNotFoundError):
                assets.capture_correction_bindings(self.workspace, document.correction_bindings, document.images,
                                                   native_recipes=(self.recipe,))
            with self.assertRaises(InvalidProjectAnalysis):
                self.store.capture_and_save(analysis_settings=self.profile,
                    images=(ProjectImageSpec(self.source, (17, 25)),), native_recipes=(self.recipe,),
                    correction_bindings=document.correction_bindings, destination=master)
            origin.assert_not_called()
        self.assertEqual(master.read_bytes(), original)

    def test_store_initial_capture_requires_explicit_prior_selection(self):
        options = dict(analysis_settings=self.profile, images=(ProjectImageSpec(self.source, (17, 25)),),
                       native_recipes=(self.recipe,), correction_bindings=(self.binding,))
        with self.assertRaises(InvalidProjectAnalysis):
            self.store.capture(**options)
        captured = self.store.capture(**options, existing_correction_bindings=())
        self.assertEqual(captured.correction_bindings, (self.binding,))
        self.assertEqual(assets.verify_correction_bindings(self.workspace, captured.correction_bindings,
                            captured.images, native_recipes=(self.recipe,)), (self.binding,))

    def test_foreign_origins_are_archival_metadata_without_local_probes(self):
        master, document = self.project()
        binding = document.correction_bindings[0]
        foreign = '/foreign-machine/input.png' if os.name == 'nt' else 'C:\\foreign-machine\\input.png'
        changed = replace(binding, inputs=tuple(replace(item, origin_path=foreign) for item in binding.inputs))
        self.store.save(replace(document, correction_bindings=(changed,)), master)
        with patch.object(assets, 'resolve_model_path', side_effect=AssertionError('Origin must not be probed')):
            loaded = self.store.load(master)
            self.assertFalse(loaded.issues)
            self.assertEqual(loaded.document.correction_bindings[0].input_sources, changed.input_sources)

    def test_resolved_asset_containment_is_checked_without_symlink_privileges(self):
        relative = self.workspace/'redirect'/'input.image'
        outside = self.root/'outside.image'
        real = Path.resolve
        def resolve(path, *args, **kwargs):
            return outside if path == relative else real(path, *args, **kwargs)
        with patch.object(Path, 'resolve', resolve):
            with self.assertRaisesRegex(ValueError, 'escapes'):
                assets.resolve_correction_asset_path(self.workspace, 'redirect/input.image')

    def test_snapshot_restore_keeps_identical_images_separate_with_distinct_masks(self):
        other = self.workspace/'duplicate.png'
        shutil.copyfile(self.source, other)
        duplicate_document = self.store.capture(analysis_settings=self.profile,
            images=(ProjectImageSpec(self.source, (17, 25)), ProjectImageSpec(other, (17, 25))),
            native_recipes=(self.recipe,))
        paths = dict(self.paths)
        paths['source_rgb'] = other
        mask = cv2.imread(str(paths['source_mask']), cv2.IMREAD_UNCHANGED)
        mask[0, 0] = 10
        paths['source_mask'] = self.root/'other-mask.png'
        self.assertTrue(cv2.imwrite(str(paths['source_mask']), mask))
        second = self.make_binding(duplicate_document.images[1].identifier, paths)
        master, document = self.project((self.binding, second), duplicate_document.images)
        snapshot = create_snapshot(self.workspace, master, self.root/'snapshots')
        portable = ProjectAnalysisStore(snapshot).load(snapshot/'snapshot.seedfiddle-project.json')
        self.assertFalse(portable.issues)
        self.assertEqual(len({image.identifier for image in portable.document.images}), 2)
        self.assertEqual(len({image.source.path for image in portable.document.images}), 2)
        self.assertEqual(len({binding.image_id for binding in portable.document.correction_bindings}), 2)
        inventory = json.loads((snapshot/'snapshot-index.json').read_bytes())['files']
        self.assertTrue(all(asset.path in inventory for binding in portable.document.correction_bindings
                            for asset in binding.inputs))
        for path in {*self.paths.values(), paths['source_mask'], self.source, other}:
            path.unlink()
        restored_root = self.root/'restored'
        restored_master = restore_snapshot(snapshot, restored_root)
        restored = ProjectAnalysisStore(restored_root).load(restored_master)
        self.assertFalse(restored.issues)
        self.assertEqual([binding.input_sources for binding in restored.document.correction_bindings],
                         [binding.input_sources for binding in document.correction_bindings])
        for image, binding in zip(restored.document.images, restored.document.correction_bindings, strict=True):
            self.assertEqual(binding.image_id, image.identifier)
            self.assertEqual(binding.inputs[2].source_sha256, image.source.sha256)
        self.assertNotEqual(restored.document.correction_bindings[0].inputs[3].source_sha256,
                            restored.document.correction_bindings[1].inputs[3].source_sha256)

    def test_missing_snapshot_asset_or_copy_failure_never_publishes_completion(self):
        master, _ = self.project()
        with patch('seedvision.persistence.snapshot._copy_correction', side_effect=OSError('interrupted copy')):
            with self.assertRaises(OSError):
                create_snapshot(self.workspace, master, self.root/'failed-snapshots')
        self.assertFalse(list((self.root/'failed-snapshots').rglob('snapshot-index.json')))
        snapshot = create_snapshot(self.workspace, master, self.root/'snapshots')
        portable = ProjectAnalysisStore(snapshot).load(snapshot/'snapshot.seedfiddle-project.json')
        (snapshot/portable.document.correction_bindings[0].inputs[0].path).unlink()
        with self.assertRaises(ValueError):
            restore_snapshot(snapshot, self.root/'restore-failure')
        self.assertFalse((self.root/'restore-failure').exists())

    def test_bound_path_cannot_resolve_outside_workspace(self):
        outside = self.root/'outside'
        outside.mkdir()
        link = self.workspace/'redirect'
        try:
            link.symlink_to(outside, target_is_directory=True)
        except OSError as error:
            self.skipTest('Directory symlinks unavailable: '+str(error))
        with self.assertRaisesRegex(ValueError, 'escapes'):
            assets.resolve_correction_asset_path(self.workspace, 'redirect/input.png')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_actual_result_selection_and_relocated_four_files_preserve_pixels(self):
        from seedvision.benchmarking.native_correction_inputs import run_native_correction_files
        first, prepared = run_native_correction_files(self.recipe, self.paths)
        binding = assets.binding_for_result(self.recipe, self.document.images[0].identifier, first)
        master, document = self.project((binding,))
        expected = first.node(STAGE_ID).outputs['corrected_image'].clone()
        del first, prepared
        for path in self.paths.values():
            path.unlink()
        moved = self.root/'moved'
        shutil.move(str(self.workspace), moved)
        loaded = ProjectAnalysisStore(moved).load(moved/master.name)
        self.assertFalse(loaded.issues)
        paths = {asset.role: moved/asset.path for asset in loaded.document.correction_bindings[0].inputs}
        result, prepared = run_native_correction_files(self.recipe, paths)
        self.assertTrue(torch.equal(result.node(STAGE_ID).outputs['corrected_image'], expected))
        self.assertEqual(loaded.document.correction_bindings[0].input_sources, binding.input_sources)
        changed = replace(self.recipe, title='Only presentation changed')
        self.assertEqual(assets.binding_for_result(changed, binding.image_id, result).recipe_sha256,
                         binding.recipe_sha256)

    def test_desktop_open_save_as_new_retains_loaded_bindings(self):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        from PySide6.QtWidgets import QApplication, QFileDialog
        from tests import test_project_analysis_ui as ui
        application = QApplication.instance() or QApplication([])
        ui.ProjectAnalysisUiTests.application = application
        master, document = self.project()
        (self.workspace/'config').mkdir()
        shutil.copyfile(Path(__file__).parents[1]/'config'/'traits.json', self.workspace/'config'/'traits.json')
        window = ui.ProjectAnalysisUiTests._window(self.workspace)
        try:
            with patch.object(window, '_confirm_project_replacement', return_value=True):
                self.assertTrue(window._open_project(master))
            self.assertEqual(window._project_correction_bindings, document.correction_bindings)
            destination = self.workspace/'other-folder'/'save-as.seedfiddle-project.json'
            with patch.object(QFileDialog, 'getSaveFileName', return_value=(str(destination), '')):
                window.save_project_as_action.trigger()
            self.assertEqual(window._current_project_path, destination)
            self.assertEqual(self.store.load(destination).document.correction_bindings, document.correction_bindings)
            window._set_project_native_recipes(())
            self.assertEqual(window._project_correction_bindings, ())
            with patch.object(window, '_confirm_project_replacement', return_value=True):
                self.assertTrue(window._open_project(destination))
                self.assertTrue(window._new_project())
            self.assertEqual(window._project_correction_bindings, ())
        finally:
            ui.ProjectAnalysisUiTests._dispose(window)

    def test_degraded_desktop_save_preserves_unresolved_image_binding_until_source_returns(self):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        from PySide6.QtWidgets import QApplication, QMessageBox
        from tests import test_project_analysis_ui as ui
        application = QApplication.instance() or QApplication([])
        ui.ProjectAnalysisUiTests.application = application
        live = self.workspace/'available.png'
        shutil.copyfile(self.source, live)
        images = self.store.capture(analysis_settings=self.profile,
            images=(ProjectImageSpec(self.source, (17, 25)), ProjectImageSpec(live, (17, 25))),
            native_recipes=(self.recipe,)).images
        master, document = self.project(images=images)
        document = replace(document, selected_image_id=images[0].identifier)
        self.store.save(document, master)
        self.source.unlink()
        (self.workspace/'config').mkdir()
        shutil.copyfile(Path(__file__).parents[1]/'config'/'traits.json', self.workspace/'config'/'traits.json')
        window = ui.ProjectAnalysisUiTests._window(self.workspace)
        try:
            def warning(_parent, title, *_args, **_kwargs):
                return (QMessageBox.StandardButton.Yes if title == 'Some project images are unavailable'
                        else QMessageBox.StandardButton.Ok)
            with patch.object(window, '_confirm_project_replacement', return_value=True), \
                    patch.object(QMessageBox, 'warning', side_effect=warning):
                self.assertTrue(window._open_project(master))
            self.assertIn(images[0].identifier, window._project_unresolved_image_records)
            self.assertTrue(window._save_project())
            preserved = self.store.load(master, verify_files=False).document
            self.assertEqual(preserved.images[0], images[0])
            self.assertEqual(preserved.correction_bindings, document.correction_bindings)
            self.assertEqual(preserved.selected_image_id, images[0].identifier)
            shutil.copyfile(self.paths['source_rgb'], self.source)
            window._add_image(self.source, open_now=False)
            self.assertTrue(window._save_project())
            restored = self.store.load(master)
            self.assertFalse(restored.issues)
            self.assertEqual(restored.document.correction_bindings, document.correction_bindings)
        finally:
            ui.ProjectAnalysisUiTests._dispose(window)
