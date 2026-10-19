"""Archived model binding and shared native/replay resource admission."""

from tests.catalogue_fixtures import CatalogueTestCase

from tests.functional_units import functional_units
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from seedvision.benchmarking import native_model_replay as native
from seedvision.benchmarking import native_stage_replay as stages
from seedvision.cuda import plantcv_kmeans_predict as predict
from seedvision.model_io import plantcv_kmeans_model as centres
from seedvision.model_io import kmeans_model_artifact as artifact
from tests.test_kmeans_model_artifact import _conversion


def _tag(value):
    if isinstance(value, dict):
        return {'kind': 'mapping', 'items': [[key, _tag(item)] for key, item in value.items()]}
    if isinstance(value, (list, tuple)):
        return {'kind': 'tuple' if type(value) is tuple else 'list', 'items': [_tag(item) for item in value]}
    return value


def _pdf(names=('seed', 'background')):
    return ('header\n' + ''.join(name + '\t' + channel + '\t' + '\t'.join(['1'] * 256) + '\n'
        for name in names for channel in ('hue', 'saturation', 'value'))).encode('utf-8')


def _identity(model, kind):
    value = dict(path=r'C:\missing-model-origin\model.dat', source_sha256=model.source_sha256,
        source_bytes=model.source_bytes, reference_version='4.11.3',
        reference_source_sha256=native.SOURCE_HASHES[kind])
    if kind == 'bayes':
        value.update(class_names=model.class_names, tables_sha256=model.tables_sha256)
    else:
        value.update(centres_sha256=model.centres_sha256, patch_size=model.patch_size,
            channels=model.channels, cluster_count=len(model.centres), preprocessing=centres.PREPROCESSING)
    return value


def _asset(raw, kind):
    model = native.bayes.parse_plantcv_pdf(raw) if kind == 'bayes' else artifact.parse_kmeans_artifact(raw).model
    value = _identity(model, kind)
    selected = {name: value[name] for name in native.MODEL_FIELDS[kind]}
    binding = dict(kind='model_asset', identity=_tag(value), bytes=dict(kind='array', name='asset'))
    specs = {'asset': dict(name='asset', shape=[len(raw)], dtype='|u1', layout='C-contiguous',
        nbytes=len(raw), data_sha256=sha256(raw).hexdigest())}
    return model, value, selected, binding, specs


def _mask_bindings(names=('seed', 'background'), shape=(7, 11)):
    pairs = [[name, dict(kind='array', name=f'mask{index}')] for index, name in enumerate(names)]
    specs = {f'mask{index}': dict(shape=list(shape), dtype='|u1', layout='C-contiguous',
        nbytes=shape[0] * shape[1]) for index in range(len(names))}
    return _tag(names), dict(kind='mapping', items=pairs), specs


@functional_units
class ModelReplayContractTests(CatalogueTestCase):
    def check_test_bayes_archived_bytes_parse_with_all_origin_io_forbidden(self):
        raw = _pdf()
        model, value, selected, binding, specs = _asset(raw, 'bayes')
        with patch.object(Path, 'open', side_effect=AssertionError('origin must not open')), \
                patch.object(Path, 'resolve', side_effect=AssertionError('origin must not resolve')):
            bound = native.asset_binding(binding, 'bayes', specs, selected)
            actual = native.parse_asset(bound, np.frombuffer(raw, dtype=np.uint8))
        self.assertEqual(actual, model)
        self.assertEqual(dict(bound.identity), value)
        self.assertEqual(bound.array_name, 'asset')
        with self.assertRaises(TypeError):
            bound.identity['source_bytes'] = 0

    def check_test_long_source_class_names_keep_the_model_domain(self):
        raw = _pdf(('seed-' + 'x' * 4096, 'background'))
        model, _, selected, binding, specs = _asset(raw, 'bayes')
        bound = native.asset_binding(binding, 'bayes', specs, selected)
        self.assertEqual(native.parse_asset(bound, np.frombuffer(raw, dtype=np.uint8)), model)

    def check_test_both_kmeans_formats_bind_outer_bytes_and_numeric_identity(self):
        model = centres.model_from_centres(((0.0,), (127.5,), (255.0,)), 1, 1)
        for raw in (centres.encode_kmeans_model(model), artifact.encode_kmeans_artifact(model, _conversion(model))):
            with self.subTest(format=len(raw)):
                expected, _, selected, binding, specs = _asset(raw, 'kmeans')
                with patch.object(Path, 'open', side_effect=AssertionError('no path IO')), \
                        patch.object(Path, 'resolve', side_effect=AssertionError('no path resolution')):
                    bound = native.asset_binding(binding, 'kmeans', specs, selected)
                    actual = native.parse_asset(bound, np.frombuffer(raw, dtype=np.uint8))
                self.assertEqual(actual, expected)
                self.assertEqual(actual.centres, model.centres)
                self.assertEqual(actual.source_sha256, sha256(raw).hexdigest())

    def check_test_raw_bytes_are_rechecked_after_header_binding(self):
        raw = _pdf()
        _, _, selected, binding, specs = _asset(raw, 'bayes')
        bound = native.asset_binding(binding, 'bayes', specs, selected)
        changed = np.frombuffer(raw, dtype=np.uint8).copy()
        changed[-2] ^= 1
        with self.assertRaisesRegex(native.NativeModelReplayError, 'SHA-256'):
            native.parse_asset(bound, changed)
        for invalid in (changed.astype(np.int16), changed[:-1], changed[::-1]):
            with self.subTest(dtype=invalid.dtype, shape=invalid.shape), self.assertRaises(native.NativeModelReplayError):
                native.parse_asset(bound, invalid)

    def check_test_matching_raw_hash_does_not_allow_a_forged_table_identity(self):
        for kind, raw in (('bayes', _pdf()), ('kmeans', centres.encode_kmeans_model(
                centres.model_from_centres(((0.0,), (255.0,)), 1, 1)))):
            _, value, selected, binding, specs = _asset(raw, kind)
            field = 'tables_sha256' if kind == 'bayes' else 'centres_sha256'
            value[field] = selected[field] = '0' * 64
            binding['identity'] = _tag(value)
            bound = native.asset_binding(binding, kind, specs, selected)
            with self.subTest(kind=kind), self.assertRaisesRegex(native.NativeModelReplayError, 'numerical model'):
                native.parse_asset(bound, np.frombuffer(raw, dtype=np.uint8))

    def check_test_identity_selection_header_and_origin_tampering_rejects_before_parse(self):
        _, value, selected, binding, specs = _asset(_pdf(), 'bayes')
        mutations = (
            lambda v, s, b, h: v.update(source_bytes=True),
            lambda v, s, b, h: v.update(path='relative.tsv'),
            lambda v, s, b, h: v.update(path=r'C:\missing\..\model.tsv'),
            lambda v, s, b, h: v.update(reference_source_sha256='0' * 64),
            lambda v, s, b, h: v.update(extra='field'),
            lambda v, s, b, h: s.update(source_bytes=s['source_bytes'] + 1),
            lambda v, s, b, h: h['asset'].update(dtype='<i4'),
            lambda v, s, b, h: h['asset'].update(shape=None),
            lambda v, s, b, h: h['asset'].update(nbytes=h['asset']['nbytes'] + 1),
            lambda v, s, b, h: h['asset'].update(data_sha256='0' * 64),
        )
        with patch.object(native.bayes, 'parse_plantcv_pdf', side_effect=AssertionError('late parse')):
            for mutate in mutations:
                v, s, b, h = deepcopy((value, selected, binding, specs))
                mutate(v, s, b, h)
                b['identity'] = _tag(v)
                with self.subTest(mutate=mutate), self.assertRaises(native.NativeModelReplayError):
                    native.asset_binding(b, 'bayes', h, s)

    def check_test_primitive_decoder_rejects_types_duplicates_nonfinite_and_traversal(self):
        invalid = (
            dict(kind='record', type='Class', fields=[]),
            dict(kind='mapping', items=[['key', 1], ['key', 2]]),
            dict(kind='tuple', items=[float('nan')]),
            dict(kind='tuple', items=[2 ** 65]),
            dict(kind='tuple', items=[0] * native.MAX_METADATA_ITEMS),
            dict(kind=[], items=[]),
        )
        for value in invalid:
            with self.subTest(value=str(value)[:80]), self.assertRaises(native.NativeModelReplayError):
                native.decode(value)
        with patch.object(native, 'MAX_METADATA_BYTES', 10):
            with self.assertRaises(native.NativeModelReplayError):
                native.decode('x' * 11)
        with self.assertRaises(native.NativeModelReplayError):
            native.canonical('\ud800')

    def check_test_ordered_mask_binding_preserves_well_formed_product_differences(self):
        # A consistent renamed/reordered output remains a product to compare.
        for names in (('seed', 'background'), ('background', 'seed'), ('changed', 'background')):
            tagged_names, mapping, specs = _mask_bindings(names)
            actual_names, pairs = native.class_mask_bindings(tagged_names, mapping, specs, (7, 11), count=2, maximum=32)
            self.assertEqual(actual_names, names)
            self.assertEqual(pairs, tuple((name, f'mask{i}') for i, name in enumerate(names)))

    def check_test_missing_duplicate_inconsistent_or_malformed_mask_bindings_reject(self):
        original = _mask_bindings()
        mutations = (
            lambda n, m, s: n.update(kind='list'),
            lambda n, m, s: n['items'].__setitem__(1, 'seed'),
            lambda n, m, s: m['items'].reverse(),
            lambda n, m, s: m['items'][1].__setitem__(1, dict(kind='array', name='mask0')),
            lambda n, m, s: s.pop('mask1'),
            lambda n, m, s: s['mask1'].update(shape=[11, 7]),
            lambda n, m, s: s['mask1'].update(dtype='|b1'),
            lambda n, m, s: m['items'].pop(),
        )
        for mutate in mutations:
            n, m, s = deepcopy(original)
            mutate(n, m, s)
            with self.subTest(mutate=mutate), self.assertRaises(native.NativeModelReplayError):
                native.class_mask_bindings(n, m, s, (7, 11), count=2, maximum=32)

    def check_test_exact_category_controls_preserve_duplicate_and_absent_ids(self):
        self.assertEqual(native.mask_names({'cat_list': None}, 3), (('0', '1', '2'), 3))
        cats = [0, 0, 2, -1, 500, 10 ** 100]
        self.assertEqual(native.mask_names({'cat_list': cats}, 3), (('selected',), len(cats)))
        for parameters in ({}, {'cat_list': []}, {'cat_list': [True]}, {'cat_list': [0.0]}, {'cat_list': None, 'extra': 1}):
            with self.subTest(parameters=parameters), self.assertRaises(native.NativeModelReplayError):
                native.mask_names(parameters, 3)

    def test_workflow_lifecycle(self):
        """7 related ModelReplayContractTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_bayes_archived_bytes_parse_with_all_origin_io_forbidden)
        self.scenario(self.check_test_long_source_class_names_keep_the_model_domain)
        self.scenario(self.check_test_both_kmeans_formats_bind_outer_bytes_and_numeric_identity)
        self.scenario(self.check_test_raw_bytes_are_rechecked_after_header_binding)
        self.scenario(self.check_test_matching_raw_hash_does_not_allow_a_forged_table_identity)
        self.scenario(self.check_test_ordered_mask_binding_preserves_well_formed_product_differences)
        self.scenario(self.check_test_exact_category_controls_preserve_duplicate_and_absent_ids)

    def test_validation_and_recovery(self):
        """3 related ModelReplayContractTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_identity_selection_header_and_origin_tampering_rejects_before_parse)
        self.scenario(self.check_test_primitive_decoder_rejects_types_duplicates_nonfinite_and_traversal)
        self.scenario(self.check_test_missing_duplicate_inconsistent_or_malformed_mask_bindings_reject)


@functional_units
class ModelReplayPlanTests(CatalogueTestCase):
    def check_test_bayes_plan_charges_bgr_expected_masks_tables_scratch_comparison_and_audit(self):
        model = native.bayes.parse_plantcv_pdf(_pdf())
        value = _identity(model, 'bayes')
        shape, p, c = (129, 131), 129 * 131, 2
        plan = native.bayes_plan(shape, value)
        table, scratch = c * 3 * 256 * 8, 16384 * 1024
        self.assertEqual(plan.selected_bytes, (3 + c) * p + model.source_bytes)
        rounded=lambda size:((size+511)//512)*512
        comparison=rounded(p)+rounded(8*p)+512
        self.assertEqual(plan.cuda_live_bytes, 2*rounded(3*p)+c*rounded(p)+rounded(c*p)+rounded(table)+scratch+comparison+65536)
        self.assertEqual(plan.native_live_bytes, c * p + table + scratch)
        self.assertIsNone(plan.work)
        self.assertEqual(plan.work_details, (('class_pixel_visits', p * c),))

    def check_test_mask_plan_loads_no_model_asset_and_counts_all_selection_visits(self):
        model = centres.model_from_centres(((0.0,), (127.0,), (255.0,)), 1, 1)
        value = _identity(model, 'kmeans')
        for cats, count, scans in ((None, 3, 3), ([0, 0, -1, 500], 1, 4)):
            p = 19 * 23
            plan = native.masks_plan((19, 23), value, {'cat_list': cats})
            self.assertEqual(plan.selected_bytes, p * (1 + count))
            rounded=lambda size:((size+511)//512)*512
            self.assertEqual(plan.cuda_live_bytes, (1+2*count)*rounded(p)+2*rounded(p)+rounded(p)+rounded(8*p)+512+65536)
            self.assertEqual(plan.work_details, (('category_pixel_visits', p * scans),))

    def check_test_prediction_plan_uses_active_channel_bytes_and_shared_native_plan(self):
        for channels in (1, 3):
            for patch_size in (1, 2, 3):
                f = patch_size * patch_size * channels
                model = centres.model_from_centres(((0.0,) * f, (255.0,) * f), patch_size, channels)
                value = _identity(model, 'kmeans')
                shape, p = (7, 11), 77
                expected = predict.preflight_kmeans_prediction_shape(*shape, model)
                with self.subTest(channels=channels, patch=patch_size):
                    actual = native.prediction_plan(shape, value, model)
                    self.assertEqual(actual.selected_bytes, (channels + 1) * p + model.source_bytes)
                    self.assertEqual(actual.native_live_bytes, expected.live_bytes)
                    self.assertEqual(actual.work, expected.work)
                    rounded=lambda size:((size+511)//512)*512
                    self.assertEqual(actual.cuda_live_bytes, rounded(channels*p)+(rounded(3*p) if channels==3 else 0)
                                     +rounded(expected.gaussian_output_bytes)+2*rounded(p)+expected.scratch_bytes
                                     +rounded(p)+rounded(8*p)+512+65536)

    def check_test_exact_selected_and_cuda_cap_boundaries_reject_without_allocations(self):
        bayes_model = native.bayes.parse_plantcv_pdf(_pdf())
        model = centres.model_from_centres(((0.0,), (255.0,)), 1, 1)
        requests = (
            (native.bayes_plan, ((7, 11), _identity(bayes_model, 'bayes'))),
            (native.masks_plan, ((7, 11), _identity(model, 'kmeans'), {'cat_list': None})),
            (native.prediction_plan, ((7, 11), _identity(model, 'kmeans'), model)),
        )
        with patch.object(torch, 'empty', side_effect=AssertionError('unexpected CUDA allocation')), \
                patch.object(torch, 'tensor', side_effect=AssertionError('unexpected tensor allocation')):
            for function, args in requests:
                plan = function(*args)
                self.assertEqual(function(*args, selected_limit=plan.selected_bytes, cuda_limit=plan.cuda_live_bytes), plan)
                for caps in ({'selected_limit': plan.selected_bytes - 1}, {'cuda_limit': plan.cuda_live_bytes - 1}):
                    with self.subTest(function=function.__name__, caps=caps), self.assertRaises(native.NativeModelReplayUnavailable):
                        function(*args, **caps)

    def check_test_resource_and_structural_failures_remain_distinct(self):
        model = centres.model_from_centres(((0.0,),), 1, 1)
        value = _identity(model, 'kmeans')
        for shape in ((0, 1), (True, 11), (7,), (1.0, 11)):
            with self.subTest(shape=shape), self.assertRaises(native.NativeModelReplayError) as caught:
                native.prediction_plan(shape, value, model)
            self.assertNotIsInstance(caught.exception, native.NativeModelReplayUnavailable)
        with self.assertRaises(native.NativeModelReplayUnavailable):
            native.prediction_plan((16_000_001, 1), value, model)
        with self.assertRaises(native.NativeModelReplayUnavailable):
            native.prediction_plan((4000, 4000), value, model)
        with self.assertRaises(native.NativeModelReplayError):
            native.prediction_plan((7, 11), value, replace(model, source_sha256='0' * 64))
        with self.assertRaises(native.NativeModelReplayError) as caught:
            native.prediction_plan((7, 11), value, replace(model, centres=((1.0,),)))
        self.assertNotIsInstance(caught.exception, native.NativeModelReplayUnavailable)

    def check_test_verified_legacy_and_model_operation_contracts_survive_extensions(self):
        # Literal compatibility witnesses: later registry additions must not change
        # the previously verified input, output, or preview contracts.
        legacy_contracts = {
            'filter.median_blur': (('gray',), ('image',), 'image'),
            'threshold.binary': (('gray',), ('mask',), 'mask'),
            'threshold.mean': (('gray',), ('mask',), 'mask'),
            'threshold.otsu': (('gray',), ('mask',), 'mask'),
            'threshold.gaussian': (('gray',), ('mask',), 'mask'),
            'threshold.triangle': (('gray',), ('mask',), 'mask'),
            'threshold.texture': (('gray',), ('mask',), 'mask'),
            'threshold.dual_channels': (('rgb',), ('mask',), 'mask'),
            'threshold.custom_range': (('rgb',), ('mask', 'masked_rgb'), 'masked_rgb'),
            'morphology.fill': (('mask',), ('mask',), 'mask'),
            'morphology.erode': (('gray',), ('mask',), 'mask'),
            'morphology.dilate': (('gray',), ('mask',), 'mask'),
            'reference.internal.watershed.distance_map': (('mask',), ('distance',), 'distance'),
            'reference.internal.watershed.local_maxima': (
                ('distance', 'mask'), ('coordinates', 'peaks'), 'peaks'),
            'reference.internal.watershed.connected_markers': (('peaks',), ('markers',), 'markers'),
            'reference.internal.watershed.negate_distance': (
                ('distance',), ('negative_distance',), 'negative_distance'),
            'reference.internal.watershed.masked_flood': (
                ('negative_distance', 'markers', 'mask', 'rgb'), ('labels', 'flood_complete'), 'labels'),
            'reference.internal.watershed.marker_count_observation': (
                ('markers', 'flood_complete'), ('marker_count',), None),
            'native.watershed_debug_diagnostic': (
                ('image', 'labels'), ('diagnostic', 'metadata'), 'diagnostic'),
            'analysis.color': (('rgb', 'labels'), ('observations', 'histogram'), None),
            'analysis.size': (('image', 'labels'), ('observations', 'diagnostic'), 'diagnostic'),
            'calibrated_seed.size_units': (('observations', 'scale'), ('observations', 'metadata'), None),
        }
        model_contracts = {
            'analysis.naive_bayes_classifier': (
                ('rgb_image', 'model', 'model_identity'),
                ('class_masks', 'class_names', 'model_identity'), 'class_masks'),
            'analysis.kmeans_predict': (
                ('rgb_image', 'gray_image', 'model', 'model_identity'), ('labels', 'model_identity'), 'labels'),
            'analysis.kmeans_masks': (
                ('labels', 'model_identity'), ('class_masks', 'class_names', 'model_identity'), 'class_masks'),
        }
        self.assertEqual(len(legacy_contracts), 22)
        self.assertEqual(set(stages.MODEL_OPERATIONS), set(model_contracts))
        self.assertEqual(len(stages.MODEL_OPERATIONS), 3)
        for operation, contract in {**legacy_contracts, **model_contracts}.items():
            with self.subTest(operation=operation):
                self.assertIn(operation, stages.SUPPORTED_REPLAY_OPERATIONS)
                self.assertEqual(stages.SUPPORTED_REPLAY_OPERATIONS[operation], contract)
        self.assertEqual(
            stages.SUPPORTED_REPLAY_OPERATIONS['calibrated_seed.crop'],
            (('corrected_rgb', 'source_frame', 'geometry', 'bounds'),
             ('rgb_image', 'gray_image', 'metadata'), 'rgb_image'))

    def test_workflow_lifecycle(self):
        """4 related ModelReplayPlanTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_bayes_plan_charges_bgr_expected_masks_tables_scratch_comparison_and_audit)
        self.scenario(self.check_test_mask_plan_loads_no_model_asset_and_counts_all_selection_visits)
        self.scenario(self.check_test_prediction_plan_uses_active_channel_bytes_and_shared_native_plan)
        self.scenario(self.check_test_verified_legacy_and_model_operation_contracts_survive_extensions)

    def test_validation_and_recovery(self):
        """2 related ModelReplayPlanTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_exact_selected_and_cuda_cap_boundaries_reject_without_allocations)
        self.scenario(self.check_test_resource_and_structural_failures_remain_distinct)


class SharedKMeansShapePlanTests(CatalogueTestCase):
    def test_pure_shape_plan_preserves_unique_centres_and_pass_counts(self):
        for rows, passes in ((((0.0,), (255.0,), (0.0,)), 1), (((0.5,), (254.5,)), 2)):
            model = centres.model_from_centres(rows, 1, 1)
            plan = predict.preflight_kmeans_prediction_shape(7, 11, model, gaussian_tile_side=7, tile_pixels=5)
            gaussian_work = (7 * (11 + 40 * 2) + 7 * 11) * 41
            self.assertEqual(plan.work, gaussian_work + 2 + 77 * (1 + 2 * passes))
            self.assertEqual(plan.tile_pixels, 5)
        singleton = centres.model_from_centres(((0.5,), (0.5,)), 1, 1)
        plan = predict.preflight_kmeans_prediction_shape(1, 1, singleton)
        self.assertEqual(plan.work, (1 * (1 + 40) + 1) * 41 + 1 + 2)

    def test_shape_admission_checks_padded_domain_controls_and_classifier_work(self):
        model = centres.model_from_centres(((0.0,),), 1, 1)
        for kwargs in ({'gaussian_tile_side': 0}, {'gaussian_tile_side': True}, {'tile_pixels': 0}, {'tile_pixels': True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                predict.preflight_kmeans_prediction_shape(7, 11, model, **kwargs)
        with self.assertRaises(ValueError):
            predict.preflight_kmeans_prediction_shape(1, 1, replace(model, patch_size=True))
        wide = centres.model_from_centres(((0.0,) * 31 ** 2,), 31, 1)
        with self.assertRaises(ValueError):
            predict.preflight_kmeans_prediction_shape(3999, 4000, wide)
        costly = centres.model_from_centres(tuple((float(i),) * 9 for i in range(256)), 3, 1)
        with self.assertRaises(ValueError):
            predict.preflight_kmeans_prediction_shape(700, 700, costly)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class SharedKMeansCudaPlanTests(CatalogueTestCase):
    def test_shape_and_real_cuda_input_plans_are_identical_for_gray_rgb_odd_even_and_singleton(self):
        for channels in (1, 3):
            for patch_size in (1, 2, 3):
                f = channels * patch_size * patch_size
                model = centres.model_from_centres(((0.0,) * f, (255.0,) * f), patch_size, channels)
                for shape in ((1, 1), (1, 11), (7, 1), (7, 11)):
                    image_shape = shape if channels == 1 else (*shape, 3)
                    image = torch.zeros(image_shape, dtype=torch.uint8, device='cuda')
                    with self.subTest(channels=channels, patch=patch_size, shape=shape):
                        self.assertEqual(predict.preflight_kmeans_prediction(image, model),
                            predict.preflight_kmeans_prediction_shape(*shape, model))
