"""Selected grid references, CUDA sampling, cache boundaries and saved facts."""

from tests.functional_units import functional_units
from copy import deepcopy
from dataclasses import replace
import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from seedvision.calibration import image as c
from seedvision.cuda import CudaContext, image_to_tensor, tensor_to_image, warp_perspective


def fixture():
    yy, xx = np.indices((64, 96))
    image = np.stack((70+xx+yy//3, 60+xx//2+yy, 90+(xx*yy)%71), axis=2).astype(np.uint8)
    tensor = image_to_tensor(image, CudaContext.resolve())
    swatches = []
    for column in range(2):
        x = 10.3+column*39
        corners = ((x, 11.2), (x+26.7, 11.2), (x+26.7, 43.8), (x, 43.8))
        mean = c._sample_quadrilateral_mean(tensor, np.asarray(corners))
        swatches.append(c.ColourSwatch(0, column, x+13.35, 27.5, corners, mean))
    return image, c.ColourCardDetection(tuple(swatches), ((8., 8.), (90., 8.), (90., 55.), (8., 55.)), 0., .9, 2)


def ruler(*args, minor_tick_mm=1., **kwargs):
    return c.RulerDetection((45., 50.), (90., 50.), 67.5, 50., 45., 6., 0., .8,
        tick_pitch_px=4., metric_pixels_per_mm=4./minor_tick_mm)


@functional_units
class NeutralCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.image, self.card = fixture()
        self.settings = c.CalibrationSettings(apply_perspective_correction=False)

    def run_calibration(self, settings=None, **kwargs):
        with patch.object(c, 'detect_colour_card', return_value=self.card), patch.object(c, 'detect_ruler', side_effect=ruler):
            return c.calibrate_image(self.image, settings or self.settings, **kwargs)

    def check_test_default_source_mode_preserves_legacy_gain_and_pixel_arithmetic(self):
        result = self.run_calibration()
        gains = c._neutral_balance_gains(self.card)
        np.testing.assert_array_equal(result.channel_gains_bgr, tuple(float(x) for x in gains))
        old = warp_perspective(image_to_tensor(self.image, CudaContext.resolve()),
            result.affine_matrix, result.corrected_bgr.shape[:2])
        balanced = old*torch.as_tensor(gains, device=old.device).reshape(1, 3, 1, 1)
        maximum = balanced.amax(dim=1, keepdim=True).clamp_min(1.)
        expected = tensor_to_image((balanced*torch.minimum(torch.ones_like(maximum), 255./maximum)).clamp(0., 255.))
        np.testing.assert_array_equal(result.corrected_bgr, expected)
        self.assertEqual(result.neutral_balance.sample_frame, 'source')
        self.assertEqual(result.neutral_balance.state, 'applied')
        self.assertEqual(result.gpu_corrected_bgr.device.type, 'cuda')
        self.assertEqual(result.source_shape_hw, self.image.shape[:2])

    def check_test_manual_single_patch_changes_gains_and_pixels(self):
        a = self.run_calibration(replace(self.settings, neutral_patch_selection='1'))
        b = self.run_calibration(replace(self.settings, neutral_patch_selection='2'))
        self.assertEqual(a.neutral_balance.used_patch_ids, (1,))
        self.assertEqual(a.neutral_balance.selection_mode, 'manual')
        self.assertNotEqual(a.channel_gains_bgr, b.channel_gains_bgr)
        self.assertFalse(np.array_equal(a.corrected_bgr, b.corrected_bgr))
        self.assertEqual(a.neutral_balance.patches[0].source_corners, self.card.swatches[0].corners)

    def check_test_automatic_threshold_and_fallback_record_the_actual_grid_ids(self):
        for means, mode, ids in (((100., 102., 104.), 'automatic_threshold', (1, 2)),
                                 ((40., 100., 200.), 'automatic_fallback', (1, 2))):
            with self.subTest(mode=mode):
                self.card = replace(self.card, swatches=tuple(replace(s, mean_bgr=means) for s in self.card.swatches))
                result = self.run_calibration()
                self.assertEqual(result.neutral_balance.selection_mode, mode)
                self.assertEqual(result.neutral_balance.used_patch_ids, ids)
                np.testing.assert_array_equal(result.channel_gains_bgr, c._neutral_balance_gains(self.card))

    def check_test_missing_selection_has_no_partial_or_automatic_fallback(self):
        result = self.run_calibration(replace(self.settings, neutral_patch_selection='1,24'))
        self.assertEqual(result.neutral_balance.state, 'unavailable')
        self.assertEqual(result.neutral_balance.available_patch_ids, (1, 2))
        self.assertEqual(result.neutral_balance.used_patch_ids, ())
        self.assertEqual(result.channel_gains_bgr, (1., 1., 1.))
        self.assertIsNotNone(result.gpu_corrected_bgr)
        self.assertTrue(any('identity gains' in warning for warning in result.warnings))

    def check_test_disabled_balance_skips_sampling_even_for_missing_manual_patch(self):
        settings = replace(self.settings, apply_colour_balance=False,
            neutral_patch_selection='24', neutral_sample_frame='corrected')
        with patch.object(c, '_sample_quadrilateral_mean', side_effect=AssertionError('unexpected sampling')):
            result = self.run_calibration(settings)
        self.assertEqual(result.neutral_balance.state, 'disabled')
        self.assertEqual(result.channel_gains_bgr, (1., 1., 1.))
        self.assertEqual(result.neutral_balance.used_patch_ids, ())

    def check_test_no_card_keeps_pixels_available_with_identity_gains(self):
        with patch.object(c, 'detect_colour_card', side_effect=c.CalibrationDetectionError('No card.')), patch.object(c, 'detect_ruler', side_effect=ruler):
            result = c.calibrate_image(self.image, self.settings)
        self.assertEqual(result.neutral_balance.state, 'unavailable')
        self.assertEqual(result.neutral_balance.available_patch_ids, ())
        self.assertEqual(result.channel_gains_bgr, (1., 1., 1.))
        self.assertGreater(result.corrected_bgr.size, 0)

    def check_test_corrected_frame_samples_transformed_unbalanced_cuda_pixels(self):
        projective = np.array([[1., .11, .2], [.025, 1., .3], [.001, -.0007, 1.]], np.float32)
        settings = replace(self.settings, apply_perspective_correction=True, neutral_patch_selection='1')
        with patch.object(c, '_projective_deskew_matrix', return_value=(projective, .1)):
            source = self.run_calibration(settings)
            corrected = self.run_calibration(replace(settings, neutral_sample_frame='corrected'))
        witness = corrected.neutral_balance.patches[0]
        np.testing.assert_array_equal(witness.sample_corners, corrected.transform_points(self.card.swatches[0].corners))
        self.assertFalse(np.allclose(witness.sampled_mean_bgr, witness.source_mean_bgr, atol=1e-4, rtol=0))
        self.assertNotEqual(source.channel_gains_bgr, corrected.channel_gains_bgr)
        raw = warp_perspective(image_to_tensor(self.image, CudaContext.resolve()),
            corrected.affine_matrix, corrected.corrected_bgr.shape[:2])
        np.testing.assert_array_equal(witness.sampled_mean_bgr,
            c._sample_quadrilateral_mean(raw, np.asarray(witness.sample_corners)))

    def check_test_out_of_image_corrected_samples_are_unavailable_without_clamping(self):
        tensor = image_to_tensor(self.image, CudaContext.resolve())
        matrix = np.eye(3); matrix[0, 2] = 1000
        with patch.object(c, '_sample_quadrilateral_mean', side_effect=AssertionError('edge-clamped sampling')):
            gains, facts = c._neutral_balance(self.card, replace(self.settings, neutral_sample_frame='corrected'), tensor, matrix)
        self.assertEqual(facts.state, 'unavailable')
        np.testing.assert_array_equal(gains, np.ones(3))

    def check_test_ruler_only_change_reuses_guarded_geometry_by_identity(self):
        with patch.object(c, 'detect_colour_card', return_value=self.card) as card, patch.object(c, 'warp_perspective', wraps=warp_perspective) as warp, patch.object(c, 'detect_ruler', side_effect=ruler) as ruler_call:
            old = c.calibrate_image(self.image, self.settings)
            new_settings = replace(self.settings, minor_tick_mm=2.)
            new = c.calibrate_image(self.image, new_settings, _previous_geometry=old)
        self.assertEqual((card.call_count, warp.call_count, ruler_call.call_count), (1, 1, 2))
        for name in ('corrected_bgr', 'gpu_corrected_bgr', 'affine_matrix', 'colour_card', 'neutral_balance', '_geometry'):
            self.assertIs(getattr(old, name), getattr(new, name))
        self.assertEqual((old.pixels_per_mm, new.pixels_per_mm), (4., 2.))
        self.assertEqual(new.calibration_settings, new_settings)
        self.assertEqual(old.calibration_settings, self.settings)

    def check_test_changed_source_controls_device_or_retained_products_invalidate_geometry(self):
        context = CudaContext.resolve()
        old = self.run_calibration()
        self.assertTrue(c.calibration_cache_matches(old, self.image, self.settings, context))
        self.assertFalse(c.calibration_cache_matches(old, self.image,
            replace(self.settings, neutral_patch_selection='1'), context))
        changed = self.image.copy(); changed[0, 0, 0] ^= 1
        self.assertFalse(c.calibration_cache_matches(old, changed, self.settings, context))
        self.assertFalse(c.calibration_cache_matches(old, self.image, self.settings, SimpleNamespace(device='cuda:99')))
        for mutate in (lambda r: r.corrected_bgr.__setitem__((0, 0, 0), 0),
                       lambda r: r.affine_matrix.__setitem__((0, 2), 5),
                       lambda r: r.gpu_corrected_bgr.add_(1)):
            old = self.run_calibration(); mutate(old)
            self.assertFalse(c.calibration_cache_matches(old, self.image, self.settings, context))
            new = self.run_calibration(_previous_geometry=old)
            self.assertIsNot(new._geometry, old._geometry)

    def check_test_generic_cuda_key_binds_the_current_concrete_device_and_input_contract(self):
        context = SimpleNamespace(device='cuda')
        old = self.run_calibration()
        index = old.gpu_corrected_bgr.device.index
        with patch('torch.cuda.current_device', return_value=index):
            key = c._calibration_source_key(self.image, context)
            self.assertEqual(key[1:], (str(old.gpu_corrected_bgr.device), 'source.uint8-bgr-hwc'))
            self.assertTrue(c.calibration_cache_matches(old, self.image, self.settings, context))
        # An identity-only witness needs no second physical GPU or allocation.
        with patch('torch.cuda.current_device', return_value=index+1):
            self.assertNotEqual(key, c._calibration_source_key(self.image, context))
            self.assertFalse(c.calibration_cache_matches(old, self.image, self.settings, context))

    def check_test_inference_products_remain_available_without_unsafe_cache_reuse(self):
        with torch.inference_mode():
            old = self.run_calibration()
        self.assertIsNone(old._geometry.product_key)
        self.assertGreater(old.corrected_bgr.size, 0)
        self.assertFalse(c.calibration_cache_matches(old, self.image, self.settings, CudaContext.resolve()))
        new = self.run_calibration(_previous_geometry=old)
        self.assertIsNot(new._geometry, old._geometry)
        self.assertIsNotNone(new._geometry.product_key)

    def check_test_same_object_reference_and_neutral_mutations_invalidate_geometry(self):
        for target_name, field, value in (('card', 'confidence', .4),
                ('swatch', 'mean_bgr', (100., 100., 100.)),
                ('neutral', 'reason', 'changed witness'),
                ('witness', 'sampled_mean_bgr', (100., 100., 100.))):
            with self.subTest(target=target_name):
                old = self.run_calibration()
                target = {'card': old.colour_card, 'swatch': old.colour_card.swatches[0],
                    'neutral': old.neutral_balance, 'witness': old.neutral_balance.patches[0]}[target_name]
                original = getattr(target, field)
                try:
                    object.__setattr__(target, field, value)
                    self.assertFalse(c.calibration_cache_matches(old, self.image, self.settings, CudaContext.resolve()))
                    self.assertIsNone(c._reusable_geometry(old, c._calibration_source_key(self.image, CudaContext.resolve()), self.settings))
                finally:
                    object.__setattr__(target, field, original)

    def check_test_same_object_ruler_mutation_refreshes_run_facts_and_reuses_geometry(self):
        old = self.run_calibration()
        object.__setattr__(old.ruler, 'matched_tick_count', 99)
        self.assertFalse(c.calibration_cache_matches(old, self.image, self.settings, CudaContext.resolve()))
        new = self.run_calibration(_previous_geometry=old)
        self.assertIs(new._geometry, old._geometry)
        self.assertEqual(new.ruler.matched_tick_count, 0)
        self.assertTrue(c.calibration_cache_matches(new, self.image, self.settings, CudaContext.resolve()))

    def check_test_invalid_projective_denominators_never_sample_or_clamp(self):
        tensor = image_to_tensor(self.image, CudaContext.resolve())
        for denominator in (0., -1., 1e10):
            with self.subTest(denominator=denominator):
                matrix = np.eye(3); matrix[2, 2] = denominator
                with patch.object(c, '_sample_quadrilateral_mean', side_effect=AssertionError('invalid projective sample')):
                    gains, facts = c._neutral_balance(self.card, replace(self.settings,
                        neutral_sample_frame='corrected'), tensor, matrix)
                self.assertEqual(facts.state, 'unavailable')
                np.testing.assert_array_equal(gains, np.ones(3))


    def test_workflow_lifecycle(self):
        """8 related NeutralCalibrationTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_default_source_mode_preserves_legacy_gain_and_pixel_arithmetic)
        self.scenario(self.check_test_manual_single_patch_changes_gains_and_pixels)
        self.scenario(self.check_test_automatic_threshold_and_fallback_record_the_actual_grid_ids)
        self.scenario(self.check_test_no_card_keeps_pixels_available_with_identity_gains)
        self.scenario(self.check_test_corrected_frame_samples_transformed_unbalanced_cuda_pixels)
        self.scenario(self.check_test_ruler_only_change_reuses_guarded_geometry_by_identity)
        self.scenario(self.check_test_generic_cuda_key_binds_the_current_concrete_device_and_input_contract)
        self.scenario(self.check_test_inference_products_remain_available_without_unsafe_cache_reuse)

    def test_validation_and_recovery(self):
        """7 related NeutralCalibrationTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_missing_selection_has_no_partial_or_automatic_fallback)
        self.scenario(self.check_test_disabled_balance_skips_sampling_even_for_missing_manual_patch)
        self.scenario(self.check_test_out_of_image_corrected_samples_are_unavailable_without_clamping)
        self.scenario(self.check_test_changed_source_controls_device_or_retained_products_invalidate_geometry)
        self.scenario(self.check_test_same_object_reference_and_neutral_mutations_invalidate_geometry)
        self.scenario(self.check_test_same_object_ruler_mutation_refreshes_run_facts_and_reuses_geometry)
        self.scenario(self.check_test_invalid_projective_denominators_never_sample_or_clamp)


@functional_units
class NeutralSettingsTests(unittest.TestCase):
    def check_test_selection_parser_is_shared_canonical_and_strict(self):
        from seedvision.calibration.neutral import normalize_neutral_patch_selection
        from seedvision.pipeline import build_default_pipeline
        node = build_default_pipeline().node('deskew_colour')
        for raw, expected in (('automatic', 'automatic'), (' 24,\t1, 9 ', '1,9,24'), ('1', '1')):
            self.assertEqual(normalize_neutral_patch_selection(raw), expected)
            self.assertEqual(node.validated_parameter_value('neutral_patch_selection', raw), expected)
            self.assertEqual(c.CalibrationSettings(neutral_patch_selection=raw).neutral_patch_selection, expected)
        for raw in ('', '1,', '1,1', '0', '25', '01', '+1', '１', '\u00a01', '1\n', None, [1], 1):
            for normalize in (normalize_neutral_patch_selection,
                              lambda v: node.validated_parameter_value('neutral_patch_selection', v),
                              lambda v: c.CalibrationSettings(neutral_patch_selection=v, apply_colour_balance=False)):
                with self.subTest(raw=raw), self.assertRaises(ValueError): normalize(raw)

    def check_test_version20_migration_preserves_all_old_flags_values_and_connections(self):
        from seedvision.persistence import (analysis_settings_profile_from_graph, analysis_settings_profile_to_payload,
            analysis_settings_profile_from_payload, apply_analysis_settings_profile, ANALYSIS_SETTINGS_VERSION)
        from seedvision.pipeline import build_default_pipeline
        graph = build_default_pipeline()
        graph.node('background_likelihood').enabled = False
        for node_id in graph.downstream('background_likelihood', recursive=True):
            graph.node(node_id).enabled = False
        graph.set_parameter('ruler_detection', 'minor_tick_mm', 2.)
        graph.disconnect(next(w for w in graph.connections if w.target == 'procedural_instances' and w.source == 'project'))
        before = analysis_settings_profile_to_payload(analysis_settings_profile_from_graph(graph))
        old = deepcopy(before); old['version'] = 20
        # A real v20 catalogue predates the opt-in trait classifier. Its absent
        # node/connections migrate additively, without altering old controls.
        old['nodes'] = [node for node in old['nodes'] if node['id'] != 'trait_classifier']
        old['connections'] = [wire for wire in old['connections'] if wire['target'] != 'trait_classifier']
        params = next(n for n in old['nodes'] if n['id'] == 'deskew_colour')['parameters']
        del params['neutral_patch_selection']; del params['neutral_sample_frame']
        target = build_default_pipeline()
        apply_analysis_settings_profile(target, analysis_settings_profile_from_payload(old))
        self.assertEqual(analysis_settings_profile_to_payload(analysis_settings_profile_from_graph(target)), before)
        self.assertEqual(before['version'], ANALYSIS_SETTINGS_VERSION)

    def check_test_current_and_version20_catalogues_remain_strict(self):
        from seedvision.persistence import (analysis_settings_profile_from_graph, analysis_settings_profile_to_payload,
            analysis_settings_profile_from_payload, apply_analysis_settings_profile, IncompatibleAnalysisSettingsProfile)
        from seedvision.pipeline import build_default_pipeline
        base = analysis_settings_profile_to_payload(analysis_settings_profile_from_graph(build_default_pipeline()))
        for version, mutate in ((21, lambda p: next(n for n in p['nodes'] if n['id']=='deskew_colour')['parameters'].pop('neutral_patch_selection')),
                                (20, lambda p: next(n for n in p['nodes'] if n['id']=='deskew_colour')['parameters'].pop('max_deskew_degrees')),
                                (20, lambda p: p['connections'].pop()),
                                (20, lambda p: p['nodes'].pop())):
            payload = deepcopy(base); payload['version'] = version; mutate(payload)
            with self.subTest(version=version), self.assertRaises(IncompatibleAnalysisSettingsProfile):
                apply_analysis_settings_profile(build_default_pipeline(), analysis_settings_profile_from_payload(payload))

    def check_test_inspector_disables_and_reenables_neutral_controls_without_rebinding(self):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        from PySide6.QtWidgets import QApplication, QCheckBox, QComboBox, QLineEdit
        from seedvision.pipeline import build_default_pipeline
        from seedvision.ui.pipeline_inspector import PipelineInspector
        app = QApplication.instance() or QApplication([])
        graph = build_default_pipeline(); widget = PipelineInspector()
        widget.parameter_changed.connect(graph.set_parameter)
        widget.set_node(graph.node('deskew_colour'))
        try:
            enabled = widget.findChild(QCheckBox, 'parameter_apply_colour_balance')
            controls = (widget.findChild(QLineEdit, 'parameter_neutral_patch_selection'),
                        widget.findChild(QComboBox, 'parameter_neutral_sample_frame'))
            for state in (False, True, False, True):
                enabled.setChecked(state); app.processEvents()
                self.assertTrue(all(control.isEnabled() == state for control in controls))
            graph.set_parameter('deskew_colour', 'apply_colour_balance', False); widget.refresh_status()
            self.assertTrue(all(not control.isEnabled() for control in controls))
        finally: widget.close()

    def check_test_new_controls_reach_calculation_settings_and_have_fixed_reference_disposition(self):
        from seedvision.pipeline import build_default_pipeline
        from seedvision.pipeline.settings import production_settings
        from seedvision.optimization.registry import capability
        graph = build_default_pipeline()
        graph.set_parameter('deskew_colour', 'neutral_patch_selection', '2,1')
        graph.set_parameter('deskew_colour', 'neutral_sample_frame', 'corrected')
        settings = production_settings(graph)['calibration_settings']
        self.assertEqual((settings.neutral_patch_selection, settings.neutral_sample_frame), ('1,2', 'corrected'))
        self.assertTrue({'neutral_patch_selection', 'neutral_sample_frame'} <= dict(capability(graph.node('deskew_colour')).fixed).keys())

    def check_test_raw_and_corrected_overlays_label_stable_grid_ids_and_used_patch(self):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        from PySide6.QtWidgets import QApplication
        from seedvision.ui.image_view import ImageView
        app = QApplication.instance() or QApplication([])
        fixture = NeutralCalibrationTests(); fixture.setUp()
        calibration = fixture.run_calibration(replace(fixture.settings, neutral_patch_selection='1'))
        for corrected in (False, True):
            view = ImageView()
            try:
                view._render_colour_reference(SimpleNamespace(calibration=calibration), corrected=corrected)
                labels = [item for item in view._overlay_items if item.data(0) == 'neutral_patch_id']
                self.assertEqual([item.data(1) for item in labels], [1, 2])
                self.assertEqual(labels[0].brush().color().name(), '#67f0a2')
                self.assertEqual(labels[1].brush().color().name(), '#ffdb55')
            finally: view.close(); app.processEvents()

    def test_workflow_lifecycle(self):
        """6 related NeutralSettingsTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_selection_parser_is_shared_canonical_and_strict)
        self.scenario(self.check_test_version20_migration_preserves_all_old_flags_values_and_connections)
        self.scenario(self.check_test_current_and_version20_catalogues_remain_strict)
        self.scenario(self.check_test_inspector_disables_and_reenables_neutral_controls_without_rebinding)
        self.scenario(self.check_test_new_controls_reach_calculation_settings_and_have_fixed_reference_disposition)
        self.scenario(self.check_test_raw_and_corrected_overlays_label_stable_grid_ids_and_used_patch)
