"""Per-image correction selections, byte-bound copies and portable projects."""

from tests.catalogue_fixtures import CatalogueTestCase

from tests.functional_units import functional_units
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


@functional_units
class ProjectCorrectionInputTests(CatalogueTestCase):
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




    def check_test_four_ordered_roles_strict_fields_and_canonical_frames(self):
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

    def check_test_layout_and_presentation_preserve_bindings_but_removed_selection_does_not(self):
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


    def check_test_conflicting_transport_identity_rejects_before_reads(self):
        inputs = list(self.binding.inputs)
        inputs[1] = replace(inputs[1], path=inputs[0].path)
        binding = replace(self.binding, inputs=tuple(inputs))
        with patch.object(assets, 'copy_verified_input') as read:
            with self.assertRaisesRegex(ValueError, 'conflicting byte'):
                assets.capture_correction_bindings(self.workspace, (binding,), self.document.images,
                                                    native_recipes=(self.recipe,))
            read.assert_not_called()

    def check_test_growing_input_reads_at_most_expected_size_plus_one(self):
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





    def check_test_standalone_copy_rejects_same_size_change_in_final_callback(self):
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



    def check_test_store_initial_capture_requires_explicit_prior_selection(self):
        options = dict(analysis_settings=self.profile, images=(ProjectImageSpec(self.source, (17, 25)),),
                       native_recipes=(self.recipe,), correction_bindings=(self.binding,))
        with self.assertRaises(InvalidProjectAnalysis):
            self.store.capture(**options)
        captured = self.store.capture(**options, existing_correction_bindings=())
        self.assertEqual(captured.correction_bindings, (self.binding,))
        self.assertEqual(assets.verify_correction_bindings(self.workspace, captured.correction_bindings,
                            captured.images, native_recipes=(self.recipe,)), (self.binding,))


    def check_test_resolved_asset_containment_is_checked_without_symlink_privileges(self):
        relative = self.workspace/'redirect'/'input.image'
        outside = self.root/'outside.image'
        real = Path.resolve
        def resolve(path, *args, **kwargs):
            return outside if path == relative else real(path, *args, **kwargs)
        with patch.object(Path, 'resolve', resolve):
            with self.assertRaisesRegex(ValueError, 'escapes'):
                assets.resolve_correction_asset_path(self.workspace, 'redirect/input.image')



    def check_test_bound_path_cannot_resolve_outside_workspace(self):
        outside = self.root/'outside'
        outside.mkdir()
        link = self.workspace/'redirect'
        try:
            link.symlink_to(outside, target_is_directory=True)
        except OSError as error:
            self.skipTest('Directory symlinks unavailable: '+str(error))
        with self.assertRaisesRegex(ValueError, 'escapes'):
            assets.resolve_correction_asset_path(self.workspace, 'redirect/input.png')

    def check_test_actual_result_selection_and_relocated_four_files_preserve_pixels(self):
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



    def test_bound_path_cannot_resolve_outside_workspace(self):
        self.scenario(self.check_test_bound_path_cannot_resolve_outside_workspace, labelled=False)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_actual_result_selection_and_relocated_four_files_preserve_pixels(self):
        self.scenario(self.check_test_actual_result_selection_and_relocated_four_files_preserve_pixels, labelled=False)

    def test_workflow_lifecycle(self):
        """5 related ProjectCorrectionInputTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_four_ordered_roles_strict_fields_and_canonical_frames)
        self.scenario(self.check_test_layout_and_presentation_preserve_bindings_but_removed_selection_does_not)
        self.scenario(self.check_test_growing_input_reads_at_most_expected_size_plus_one)
        self.scenario(self.check_test_store_initial_capture_requires_explicit_prior_selection)
        self.scenario(self.check_test_resolved_asset_containment_is_checked_without_symlink_privileges)

    def test_validation_and_recovery(self):
        """2 related ProjectCorrectionInputTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_conflicting_transport_identity_rejects_before_reads)
        self.scenario(self.check_test_standalone_copy_rejects_same_size_change_in_final_callback)
