"""Software oracles for trait classification; all truth is explicitly synthetic."""

from tests.functional_units import functional_units
from dataclasses import replace
import json
import math
from unittest.mock import patch
import unittest

import numpy as np
import torch

from seedvision.learning.trait_features import (
    TRAIT_FEATURE_SCHEMA, TraitFeatureBatch, extract_trait_features,
)
from seedvision.learning.trait_classifier import (
    FrozenTraitModel, TraitDefinition, TraitExample, TraitTrainingEvidence,
    TraitVocabulary, evaluate_trait_model, predict_traits, train_trait_model,
)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for trait features and training')
@functional_units
class TraitClassifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device('cuda')
        cls.vocabulary = TraitVocabulary('synthetic-v1', 'synthetic definition fixture', 'a' * 64,
            (TraitDefinition('dark', 'Synthetic low first feature', 'synthetic-only'),
             TraitDefinition('light', 'Synthetic high first feature', 'synthetic-only')))
        cls.training = cls.evidence('train', [.1, .15, .2, .25, .75, .8, .85, .9])
        cls.calibration = cls.evidence('calibrate', [.12, .22, .78, .88])
        cls.model = train_trait_model(cls.training, cls.calibration, cls.vocabulary,
                                     iterations=120, calibration_iterations=31)

    @classmethod
    def evidence(cls, prefix, first_values, *, targets=None, permitted=None):
        values = torch.zeros((len(first_values), 20), dtype=torch.float64, device='cuda')
        values[:, 0] = torch.tensor(first_values, device='cuda')
        facts = tuple(TraitExample(f'{prefix}-{index}', f'{prefix}-physical-{index}',
            f'{prefix}-group-{index % 2}', targets[index] if targets else ('dark' if value < .5 else 'light'),
            'synthetic labelled fixture; no biological truth', True if permitted is None else permitted[index])
            for index, value in enumerate(first_values))
        return TraitTrainingEvidence(values, facts, f'synthetic {prefix} features')

    def batch(self, values=None):
        values = self.calibration.values if values is None else values
        return TraitFeatureBatch(values, tuple(range(1, len(values) + 1)), (),
                                 json.dumps({'source': 'synthetic fixture', 'frame': 'corrected-v1'}))

    def train(self, training=None, calibration=None, **kwargs):
        return train_trait_model(training or self.training, calibration or self.calibration,
                                 self.vocabulary, iterations=120, calibration_iterations=31, **kwargs)

    def check_test_synthetic_grouped_training_predicts_supplied_definition_ids(self):
        result = predict_traits(self.model, self.batch(), abstention_threshold=.7)
        self.assertEqual([x['predicted_trait_id'] for x in result['predictions']], ['dark', 'dark', 'light', 'light'])
        self.assertFalse(result['scientifically_validated'])
        self.assertIn('not confidence intervals', result['probability_semantics'])
        diagnostics = self.model.payload['development_diagnostics']
        self.assertLessEqual(diagnostics['nll'], diagnostics['uncalibrated_nll'] + 1e-12)
        self.assertEqual(sum(x['count'] for x in diagnostics['reliability_bins']), 4)
        self.assertFalse(diagnostics['scientifically_validated'])
        self.assertEqual(self.model.payload['standardization']['constant_columns'], list(range(1, 20)))

    def check_test_group_physical_and_example_overlap_reject_before_optimization(self):
        for key in ('group_id', 'physical_id', 'example_id'):
            facts = list(self.calibration.facts)
            facts[0] = replace(facts[0], **{key: getattr(self.training.facts[0], key)})
            evidence = replace(self.calibration, facts=tuple(facts))
            with self.subTest(key=key), patch('torch.optim.Adam', side_effect=AssertionError('optimization ran')):
                with self.assertRaisesRegex(ValueError, f'overlap by {key}'):
                    self.train(calibration=evidence)

    def check_test_denied_nonfinite_features_never_enter_statistics_fit_or_calibration(self):
        extended = self.evidence('denied-train', [.1, .15, .2, .25, .75, .8, .85, .9, float('nan')],
                                 permitted=[True] * 8 + [False])
        extended_calibration = self.evidence('denied-cal', [.12, .22, .78, .88, float('inf')],
                                             permitted=[True] * 4 + [False])
        candidate = self.train(extended, extended_calibration)
        for key in ('weights', 'bias', 'standardization', 'temperature_calibration'):
            self.assertEqual(candidate.payload[key], self.model.payload[key])
        self.assertEqual(len(candidate.payload['split_audit']['training']['excluded']), 1)
        self.assertFalse(candidate.payload['split_audit']['calibration']['excluded'][0]['permitted_use'])

    def check_test_calibration_changes_temperature_without_changing_training_statistics_or_weights(self):
        noisy = self.evidence('noisy-cal', [.12, .22, .78, .88], targets=['light', 'dark', 'light', 'dark'])
        model = self.train(calibration=noisy)
        for key in ('weights', 'bias', 'standardization'):
            self.assertEqual(model.payload[key], self.model.payload[key])
        self.assertNotEqual(model.payload['temperature_calibration']['temperature'],
                            self.model.payload['temperature_calibration']['temperature'])
        self.assertEqual(model.payload['evidence_sha256']['training'], self.model.payload['evidence_sha256']['training'])
        self.assertNotEqual(model.payload['evidence_sha256']['calibration'], self.model.payload['evidence_sha256']['calibration'])
        self.assertLessEqual(model.payload['development_diagnostics']['nll'],
                            model.payload['development_diagnostics']['uncalibrated_nll'] + 1e-12)

    def check_test_actual_threshold_no_model_schema_and_invalid_feature_abstention(self):
        values = self.calibration.values.clone()
        values[0, 0] = .5
        batch = self.batch(values)
        permissive = predict_traits(self.model, batch, abstention_threshold=0)
        strict = predict_traits(self.model, batch, abstention_threshold=1)
        self.assertFalse(permissive['predictions'][0]['abstained'])
        self.assertTrue(strict['predictions'][0]['abstained'])
        self.assertIsNotNone(strict['predictions'][0]['probabilities'])
        absent = predict_traits(None, batch)
        self.assertEqual(absent['predictions'][0]['abstention_reason'], 'no_frozen_model')
        incompatible = predict_traits(self.model, replace(batch, schema='other-v1'))
        self.assertEqual(incompatible['predictions'][0]['abstention_reason'], 'incompatible_feature_schema')
        values[0, 0] = float('nan')
        invalid = predict_traits(self.model, self.batch(values))
        self.assertEqual(invalid['predictions'][0]['abstention_reason'], 'invalid_features')
        self.assertIsNone(invalid['predictions'][0]['probabilities'])

    def check_test_frozen_json_roundtrip_hash_and_nested_detachment(self):
        restored = FrozenTraitModel.from_json(self.model.to_json())
        self.assertEqual(restored.model_sha256, self.model.model_sha256)
        self.assertEqual(predict_traits(restored, self.batch()), predict_traits(self.model, self.batch()))
        payload = restored.payload
        payload['weights'][0][0] += 10
        self.assertEqual(restored.model_sha256, self.model.model_sha256)
        envelope = json.loads(self.model.to_json())
        envelope['model']['weights'][0][0] += 1
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            FrozenTraitModel.from_json(json.dumps(envelope))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            FrozenTraitModel.from_json('{"model":{},"model":{},"model_sha256":"x"}')

    def check_test_frozen_model_rejects_invalid_numbers_dimensions_and_claims(self):
        alterations = [
            lambda p: p['weights'][0].pop(),
            lambda p: p['bias'].__setitem__(0, True),
            lambda p: p['standardization']['scale'].__setitem__(0, 0),
            lambda p: p['temperature_calibration'].__setitem__('temperature', float('inf')),
            lambda p: p.__setitem__('scientifically_validated', True),
            lambda p: p['vocabulary']['traits'][0].__setitem__('definition', ''),
            lambda p: p['split_audit']['training']['included'][0].__setitem__('permitted_use', False),
            lambda p: p['split_audit']['calibration']['included'][0].__setitem__('physical_id', 'train-physical-0'),
            lambda p: p['development_diagnostics'].__setitem__('coverage', True),
            lambda p: p['development_diagnostics']['reliability_bins'][0].__setitem__('count', 4),
        ]
        for alteration in alterations:
            with self.subTest(alteration=alteration):
                payload = self.model.payload
                alteration(payload)
                with self.assertRaises(ValueError):
                    FrozenTraitModel(json.dumps(payload))

    def check_test_disjoint_evaluation_never_refits_and_reports_zero_coverage_error_unavailable(self):
        evidence = self.evidence('heldout', [.5, .5], targets=['dark', 'light'])
        with patch('torch.optim.Adam', side_effect=AssertionError('evaluation trained')):
            result = evaluate_trait_model(self.model, evidence, abstention_threshold=1)
        self.assertEqual(result['coverage'], 0)
        self.assertIsNone(result['accepted_error'])
        self.assertEqual(result['model_sha256'], self.model.model_sha256)
        for prior in (self.training, self.calibration):
            for key in ('group_id', 'physical_id'):
                facts = list(evidence.facts)
                facts[0] = replace(facts[0], **{key: getattr(prior.facts[0], key)})
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'overlap'):
                    evaluate_trait_model(self.model, replace(evidence, facts=tuple(facts)))

    def check_test_cuda_probabilities_match_independent_numpy_softmax(self):
        payload = self.model.payload
        values = self.calibration.values.cpu().numpy()
        logits = ((values - np.array(payload['standardization']['mean'])) /
                  np.array(payload['standardization']['scale'])) @ np.array(payload['weights']).T + payload['bias']
        logits /= payload['temperature_calibration']['temperature']
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        reference = exp / exp.sum(axis=1, keepdims=True)
        result = predict_traits(self.model, self.batch(), abstention_threshold=0)
        actual = np.array([list(row['probabilities'].values()) for row in result['predictions']])
        np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-12)

    def check_test_training_controls_change_calculation_and_invalid_domains_reject(self):
        shorter = train_trait_model(self.training, self.calibration, self.vocabulary, iterations=2)
        self.assertNotEqual(shorter.payload['weights'], self.model.payload['weights'])
        regularized = self.train(l2=.5)
        self.assertNotEqual(regularized.payload['weights'], self.model.payload['weights'])
        for kwargs in ({'learning_rate': True}, {'l2': -1}, {'temperature_bounds': (1, 1)},
                       {'abstention_threshold': 1.1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError): self.train(**kwargs)
        denied = replace(self.training, facts=tuple(replace(x, permitted_use=False) for x in self.training.facts))
        with self.assertRaisesRegex(ValueError, 'no permitted'): self.train(training=denied)
        missing = replace(self.training, facts=tuple(replace(x, trait_id='dark') for x in self.training.facts))
        with self.assertRaisesRegex(ValueError, 'Every defined class'): self.train(training=missing)

    def check_test_extreme_finite_normalization_rejects_without_model(self):
        values = self.training.values.clone()
        values[:, 0] = torch.tensor([-1e308] * 4 + [1e308] * 4, device='cuda', dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, 'numeric range'):
            self.train(training=replace(self.training, values=values))

    def check_test_contradictory_physical_views_reject_and_repeated_fit_is_deterministic(self):
        facts = list(self.training.facts)
        facts[-1] = replace(facts[-1], physical_id=facts[0].physical_id)
        with patch('torch.optim.Adam', side_effect=AssertionError('optimization ran')):
            with self.assertRaisesRegex(ValueError, 'contradictory'):
                self.train(training=replace(self.training, facts=tuple(facts)))
        repeated = self.train()
        self.assertEqual(repeated.model_sha256, self.model.model_sha256)

    def check_test_calibrated_numeric_overflow_abstains_without_nonfinite_probabilities(self):
        payload = self.model.payload
        payload['temperature_calibration'].update(temperature=1e-300, bounds=[1e-300, 1.])
        payload['weights'][0][0] = 1e300
        model = FrozenTraitModel(json.dumps(payload))
        result = predict_traits(model, self.batch())
        self.assertTrue(all(x['abstained'] and x['probabilities'] is None for x in result['predictions']))
        json.dumps(result, allow_nan=False)
        with self.assertRaisesRegex(ValueError, 'calibrated numeric range'):
            evaluate_trait_model(model, self.evidence('overflow-heldout', [.12, .88]))

    def check_test_training_calibration_and_evaluation_schema_mismatch_reject_before_computation(self):
        with patch('torch.optim.Adam', side_effect=AssertionError('optimization ran')):
            for evidence in (self.training, self.calibration):
                with self.assertRaisesRegex(ValueError, 'pixel schema'):
                    replace(evidence, schema='different-feature-meaning-v1')
        # Exercise the public API guard against corrupted/deserialized evidence,
        # independently of the evidence constructor's equivalent invariant.
        corrupted = replace(self.calibration)
        object.__setattr__(corrupted, 'schema', 'different-feature-meaning-v1')
        with patch('torch.optim.Adam', side_effect=AssertionError('optimization ran')):
            with self.assertRaisesRegex(ValueError, 'feature schemas'):
                self.train(calibration=corrupted)
        with patch('seedvision.learning.trait_classifier._logits', side_effect=AssertionError('scoring ran')):
            with self.assertRaisesRegex(ValueError, 'feature schema'):
                evaluate_trait_model(self.model, corrupted)

    def extract(self, image, labels, requested=(1,), eligible=(1,), **kwargs):
        return extract_trait_features(image, labels, requested, eligible,
            encoding=kwargs.get('encoding', 'rgb_unit'), image_frame=kwargs.get('image_frame', 'corrected-A'),
            label_frame=kwargs.get('label_frame', 'corrected-A'))

    def check_test_uniform_square_and_multicolour_cpu_feature_oracles(self):
        image = torch.tensor([.2, .4, .8], device='cuda', dtype=torch.float64).expand(2, 2, 3)
        labels = torch.ones((2, 2), device='cuda', dtype=torch.int32)
        batch = self.extract(image, labels)
        expected = [.2, 0, .2, .2, .2, .4, 0, .4, .4, .4, .8, 0, .8, .8, .8,
                    math.log(4), 0, 1, 0, 4]
        np.testing.assert_allclose(batch.values.cpu().numpy()[0], expected, atol=1e-12)
        array = np.arange(18, dtype=np.float64).reshape(2, 3, 3) / 20
        colour = np.stack((array.mean(axis=(0, 1)), array.std(axis=(0, 1)),
                           *np.quantile(array.reshape(-1, 3), [.1, .5, .9], axis=0)), axis=1).ravel()
        batch = self.extract(torch.tensor(array, device='cuda'), torch.ones((2, 3), device='cuda', dtype=torch.int32))
        yy, xx = np.indices((2, 3)); xy = np.column_stack((yy.ravel(), xx.ravel()))
        moment = np.cov(xy.T, bias=True); eigen = np.linalg.eigvalsh(moment)
        geometry = [math.log(6), math.log(1.5), 1, (eigen[-1] - eigen[0]) / eigen.sum(), 10 / math.sqrt(6)]
        np.testing.assert_allclose(batch.values.cpu().numpy()[0], np.r_[colour, geometry], atol=1e-12)

    def check_test_feature_encoding_rotation_translation_and_other_pixels_invariance(self):
        byte = torch.arange(18, device='cuda', dtype=torch.uint8).reshape(2, 3, 3) * 10
        labels = torch.ones((2, 3), device='cuda', dtype=torch.int32)
        baseline = self.extract(byte, labels, encoding='rgb_byte')
        bgr = self.extract(byte.flip(-1), labels, encoding='bgr_byte')
        unit = self.extract(byte.double() / 255, labels)
        rotated = self.extract(torch.rot90(byte, 1, (0, 1)), torch.rot90(labels, 1), encoding='rgb_byte')
        for candidate in (bgr, unit, rotated):
            torch.testing.assert_close(candidate.values, baseline.values, rtol=1e-12, atol=1e-12)
        padded_image = torch.ones((5, 6, 3), device='cuda', dtype=torch.float64)
        padded_labels = torch.zeros((5, 6), device='cuda', dtype=torch.int32)
        padded_image[2:4, 2:5] = byte.double() / 255
        padded_labels[2:4, 2:5] = 1
        padded_labels[0, 0] = 2
        moved = self.extract(padded_image, padded_labels)
        torch.testing.assert_close(moved.values, baseline.values, rtol=1e-12, atol=1e-12)

    def check_test_feature_rejection_provenance_and_cuda_retention_without_raster_transfer(self):
        image = torch.full((2, 2, 3), .5, device='cuda')
        labels = torch.ones((2, 2), device='cuda', dtype=torch.int32)
        with patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('raster transferred')):
            batch = self.extract(image, labels, requested=(1, 2, 3), eligible=(1, 2))
        self.assertTrue(batch.values.is_cuda)
        self.assertEqual(batch.rejected, ((2, 'missing_instance'), (3, 'ineligible_instance')))
        self.assertEqual(self.extract(image, labels, label_frame='other').rejected, ((1, 'incompatible_frame'),))
        self.assertEqual(self.extract(image, labels, image_frame='').rejected, ((1, 'missing_frame_provenance'),))
        image[0, 0, 0] = float('nan')
        rejected = self.extract(image, labels)
        self.assertEqual(rejected.rejected, ((1, 'invalid_pixels'),))
        result = predict_traits(self.model, rejected)
        self.assertEqual(result['predictions'][0]['eligibility_reason'], 'invalid_pixels')
        with self.assertRaises(ValueError): self.extract(image, labels, requested=(1, 1))
        with self.assertRaises(ValueError): self.extract(image, labels, encoding='unknown')

    def test_workflow_lifecycle(self):
        """8 related TraitClassifierTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_synthetic_grouped_training_predicts_supplied_definition_ids)
        self.scenario(self.check_test_denied_nonfinite_features_never_enter_statistics_fit_or_calibration)
        self.scenario(self.check_test_calibration_changes_temperature_without_changing_training_statistics_or_weights)
        self.scenario(self.check_test_frozen_json_roundtrip_hash_and_nested_detachment)
        self.scenario(self.check_test_cuda_probabilities_match_independent_numpy_softmax)
        self.scenario(self.check_test_calibrated_numeric_overflow_abstains_without_nonfinite_probabilities)
        self.scenario(self.check_test_uniform_square_and_multicolour_cpu_feature_oracles)
        self.scenario(self.check_test_feature_encoding_rotation_translation_and_other_pixels_invariance)

    def test_validation_and_recovery(self):
        """9 related TraitClassifierTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_group_physical_and_example_overlap_reject_before_optimization)
        self.scenario(self.check_test_actual_threshold_no_model_schema_and_invalid_feature_abstention)
        self.scenario(self.check_test_frozen_model_rejects_invalid_numbers_dimensions_and_claims)
        self.scenario(self.check_test_disjoint_evaluation_never_refits_and_reports_zero_coverage_error_unavailable)
        self.scenario(self.check_test_training_controls_change_calculation_and_invalid_domains_reject)
        self.scenario(self.check_test_extreme_finite_normalization_rejects_without_model)
        self.scenario(self.check_test_contradictory_physical_views_reject_and_repeated_fit_is_deterministic)
        self.scenario(self.check_test_training_calibration_and_evaluation_schema_mismatch_reject_before_computation)
        self.scenario(self.check_test_feature_rejection_provenance_and_cuda_retention_without_raster_transfer)


if __name__ == '__main__':
    unittest.main()
