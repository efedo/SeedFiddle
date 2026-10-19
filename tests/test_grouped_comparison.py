"""Compact synthetic engineering witnesses, not biological validation."""

from tests.functional_units import functional_units
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from seedvision.benchmarking.checkpoints import CoordinateFrame
from seedvision.benchmarking.manifest import (
    BenchmarkManifest, BenchmarkSample, SplitRole, TruthIdentity, TruthState,
)
from seedvision.benchmarking import grouped_comparison as gc


def digest(text):
    return sha256(text.encode()).hexdigest()


def manifest(count=3):
    samples = []
    for index in range(count):
        identifier = f"image{index}"
        truth = TruthIdentity(TruthState.AVAILABLE, digest(identifier + 'annotations'),
            digest(identifier + 'regions'), digest(identifier + 'eligibility'),
            revision='review1', reviewer_ids=('expert1',))
        samples.append(BenchmarkSample(identifier, digest(identifier + 'source'),
            CoordinateFrame(identifier, (8, 12), 'top_left', 'x_right_y_down', 'pixel'),
            'soybean', SplitRole.LOCKED_TEST, truth,
            physical_specimen_id=f"seed{index}", capture_session_id=f"session{index}"))
    return BenchmarkManifest('synthetic_engineering', tuple(samples), True)


def plan(study, **kwargs):
    return gc.ComparisonPlan(study.manifest_sha256,
        gc.MethodBinding('reference', 'original_reference', digest('reference recipe')),
        gc.MethodBinding('candidate', 'native_compatible', digest('candidate recipe')),
        (gc.MetricGate('quality', 'higher', .0, .5),), repetitions=100, seed=17, **kwargs)


def records(study, specification, values=None, strata=()):
    result = []
    values = values or [(0.5, 0.7)] * len(study.samples)
    for sample, (reference, candidate) in zip(study.samples, values, strict=True):
        for method, score in ((specification.reference, reference), (specification.candidate, candidate)):
            result.append(gc.ComparisonRecord(sample.sample_id, sample.identity_sha256,
                gc.truth_identity_sha256(sample), method.method_id, method.identity_sha256,
                digest(sample.sample_id + method.method_id), 'available',
                ((specification.metrics[0].name, score),), strata,
                gallery_links=(f"gallery/{sample.sample_id}.png",)))
    return tuple(result)


def endpoint(report, scope=0):
    return report['strata'][scope]['metrics'][0]


@functional_units
class GroupedComparisonTests(unittest.TestCase):
    def check_test_constant_paired_shift_has_exact_bounds_and_never_certifies_science(self):
        study = manifest()
        specification = plan(study)
        report = gc.evaluate_grouped_comparison(study, records(study, specification), specification)
        metric = endpoint(report)
        self.assertEqual(report['state'], 'pass')
        self.assertAlmostEqual(metric['candidate_one_sided_bound'], .7)
        self.assertAlmostEqual(metric['difference_one_sided_bound'], .2)
        self.assertEqual(metric['independent_groups'], 3)
        self.assertFalse(report['scientifically_validated'])
        self.assertEqual(report['plan_sha256'], specification.plan_sha256)
        self.assertEqual(report['images'][0]['observations']['candidate']['gallery_links'], ['gallery/image0.png'])

    def check_test_seeded_bounds_match_independent_draw_matrix_oracle_and_rng_is_private(self):
        study = manifest()
        specification = plan(study)
        values = [(0.6, 0.8), (0.4, 0.3), (0.2, 0.5)]
        rows = records(study, specification, values)
        # Engine sorts content-derived group IDs; build the independently known
        # group order, then draw one matrix rather than its streaming reduction.
        order = sorted(range(3), key=lambda i: digest(json.dumps([f'image{i}'], separators=(',', ':'))))
        numeric = np.asarray([values[i] for i in order])
        generator = np.random.Generator(np.random.PCG64(17))
        draws = generator.integers(0, 3, size=(100, 3))
        absolute_draws = generator.integers(0, 3, size=(100, 3))
        candidates = numeric[absolute_draws, 1].mean(axis=1)
        differences = (numeric[draws, 1] - numeric[draws, 0]).mean(axis=1)
        before = np.random.get_state()
        first = gc.evaluate_grouped_comparison(study, rows, specification)
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        self.assertAlmostEqual(endpoint(first)['candidate_one_sided_bound'], np.quantile(candidates, .05), places=12)
        self.assertAlmostEqual(endpoint(first)['difference_one_sided_bound'], np.quantile(differences, .05), places=12)
        self.assertEqual(first, gc.evaluate_grouped_comparison(study, tuple(reversed(rows)), specification))

    def check_test_lower_error_uses_upper_bound_and_absolute_failure_is_not_equivalence(self):
        study = manifest()
        specification = replace(plan(study), metrics=(gc.MetricGate('error', 'lower', .15, .2),))
        report = gc.evaluate_grouped_comparison(study, records(study, specification, [(0.1, .2)] * 3), specification)
        metric = endpoint(report)
        self.assertAlmostEqual(metric['difference_one_sided_bound'], .1)
        self.assertTrue(metric['noninferiority_pass'])
        self.assertFalse(metric['absolute_pass'])
        self.assertEqual(report['state'], 'fail')

    def check_test_transitive_source_and_acquisition_links_never_resample_repeated_seeds(self):
        original = manifest()
        samples = list(original.samples)
        samples[1] = replace(samples[1], source_sha256=samples[0].source_sha256)
        samples[2] = replace(samples[2], capture_session_id=samples[1].capture_session_id)
        study = replace(original, samples=tuple(samples))
        specification = plan(study)
        result = gc.evaluate_grouped_comparison(study, records(study, specification), specification)
        self.assertEqual(len({i['group_id'] for i in result['images']}), 1)
        self.assertEqual(endpoint(result)['independent_groups'], 1)
        self.assertEqual(result['state'], 'inconclusive')
        self.assertIsNone(endpoint(result)['difference_one_sided_bound'])

    def check_test_group_and_image_aggregation_have_prespecified_distinct_estimands(self):
        original = manifest(5)
        samples = list(original.samples)
        for i in range(4):
            samples[i] = replace(samples[i], capture_session_id='same_session')
        study = replace(original, samples=tuple(samples))
        specification = plan(study)
        rows = records(study, specification, [(0., 0.)] * 4 + [(1., 1.)])
        grouped = endpoint(gc.evaluate_grouped_comparison(study, rows, specification))
        image = endpoint(gc.evaluate_grouped_comparison(study, rows, replace(specification, aggregation='image_mean')))
        self.assertAlmostEqual(grouped['candidate_estimate'], .5)
        self.assertAlmostEqual(image['candidate_estimate'], .2)
        self.assertEqual(grouped['independent_groups'], 2)

    def check_test_failed_abstained_and_missing_rows_stay_in_coverage_and_gallery(self):
        study = manifest()
        specification = plan(study)
        rows = list(records(study, specification))
        rows[1] = replace(rows[1], status='failed', metrics=(('quality', None),), reason='decoder failed')
        rows[3] = replace(rows[3], status='abstained', metrics=(('quality', None),), reason='insufficient confidence')
        rows.pop(5)
        result = gc.evaluate_grouped_comparison(study, rows, specification)
        self.assertEqual(endpoint(result)['paired_coverage'], 0.)
        states = [i['observations']['candidate']['status'] for i in result['images']]
        self.assertEqual(states, ['failed', 'abstained', 'not_recorded'])
        self.assertEqual(result['strata'][0]['outcome_counts']['candidate'],
            {'available': 0, 'failed': 1, 'abstained': 1, 'unavailable': 0, 'not_recorded': 1})
        self.assertEqual(sum(g['outcome_counts']['candidate']['failed'] for g in result['groups']), 1)
        self.assertEqual(result['state'], 'inconclusive')
        self.assertIsNone(endpoint(result)['candidate_estimate'])

    def check_test_missing_reference_cannot_remove_poor_candidate_from_absolute_fitness(self):
        study = manifest()
        specification = replace(plan(study, minimum_coverage=.6),
            metrics=(gc.MetricGate('quality', 'higher', .7, .1),))
        rows = list(records(study, specification, [(.8, .8), (.8, .8), (.8, .0)]))
        rows[4] = replace(rows[4], status='failed', metrics=(('quality', None),), reason='reference failed')
        result = gc.evaluate_grouped_comparison(study, rows, specification)
        metric = endpoint(result)
        self.assertAlmostEqual(metric['candidate_estimate'], .8)
        self.assertAlmostEqual(metric['candidate_absolute_estimate'], 1.6 / 3)
        self.assertAlmostEqual(metric['paired_coverage'], 2 / 3)
        self.assertEqual(metric['candidate_coverage'], 1)
        self.assertEqual(metric['paired_independent_groups'], 2)
        self.assertEqual(metric['candidate_independent_groups'], 3)
        self.assertTrue(metric['noninferiority_pass'])
        self.assertFalse(metric['absolute_pass'])
        self.assertEqual(result['state'], 'fail')

    def check_test_critical_missing_and_unknown_membership_are_inconclusive_not_dropped(self):
        study = manifest()
        absent = gc.CriticalStratum('lupin', (('species', 'lupin'),))
        specification = plan(study, critical_strata=(absent,))
        result = gc.evaluate_grouped_comparison(study, records(study, specification), specification)
        self.assertEqual(result['strata'][1]['total_images'], 0)
        self.assertEqual(result['state'], 'inconclusive')
        unknown = gc.CriticalStratum('contact', (('contact', 'touching'),))
        specification = replace(specification, critical_strata=(unknown,))
        result = gc.evaluate_grouped_comparison(study, records(study, specification), specification)
        self.assertEqual(result['unknown_stratum_samples'], ['image0', 'image1', 'image2'])
        self.assertIn('stratum_membership_not_recorded', endpoint(result, 1)['blockers'])

    def check_test_all_prespecified_critical_gates_are_required(self):
        study = manifest(4)
        specification = plan(study, critical_strata=(gc.CriticalStratum('difficult', (('contact', 'touching'),)),))
        rows = []
        for i, sample in enumerate(study.samples):
            singleton = replace(study, samples=(sample,))
            rows.extend(records(singleton, specification, [(.8, .9 if i < 2 else .1)],
                strata=(('contact', 'isolated' if i < 2 else 'touching'),)))
        result = gc.evaluate_grouped_comparison(study, rows, specification)
        self.assertEqual(result['state'], 'fail')
        self.assertFalse(endpoint(result, 1)['noninferiority_pass'])

    def check_test_identity_duplicate_endpoint_and_stratum_mismatches_reject(self):
        study = manifest()
        specification = plan(study)
        rows = records(study, specification)
        for field in ('sample_sha256', 'truth_sha256', 'method_sha256'):
            with self.subTest(field=field), self.assertRaises(gc.GroupedComparisonError):
                gc.evaluate_grouped_comparison(study, (replace(rows[0], **{field: digest('wrong')}), *rows[1:]), specification)
        with self.assertRaises(gc.GroupedComparisonError):
            gc.evaluate_grouped_comparison(study, (*rows, rows[0]), specification)
        with self.assertRaises(gc.GroupedComparisonError):
            gc.evaluate_grouped_comparison(study, (replace(rows[0], metrics=(('other', 1.),)), *rows[1:]), specification)
        with self.assertRaises(gc.GroupedComparisonError):
            gc.evaluate_grouped_comparison(study, (replace(rows[0], strata=(('contact', 'touching'),)), *rows[1:]), specification)
        with self.assertRaises(gc.GroupedComparisonError):
            gc.evaluate_grouped_comparison(study, rows, replace(specification, manifest_sha256=digest('wrong')))

    def check_test_unknown_truth_and_missing_access_never_authorize_comparison(self):
        study = manifest()
        samples = list(study.samples)
        samples[0] = replace(samples[0], truth=TruthIdentity(TruthState.UNAVAILABLE, unavailable_reason='not supplied'))
        study = replace(study, samples=tuple(samples))
        specification = plan(study)
        with self.assertRaisesRegex(gc.GroupedComparisonError, 'truth'):
            gc.evaluate_grouped_comparison(study, records(study, specification), specification)
        rows = list(records(study, specification))
        rows[0] = replace(rows[0], metrics=(('quality', None),), reason='truth absent')
        rows[1] = replace(rows[1], metrics=(('quality', None),), reason='truth absent')
        self.assertEqual(gc.evaluate_grouped_comparison(study, rows, specification)['state'], 'inconclusive')
        specification = replace(specification, reference=replace(specification.reference, training_access_ids=('unrecorded',)))
        with self.assertRaisesRegex(gc.GroupedComparisonError, 'access'):
            gc.evaluate_grouped_comparison(study, (), specification)

    def check_test_work_and_metric_budgets_reject_before_random_draws(self):
        study = manifest()
        specification = plan(study)
        with patch.object(gc, 'MAX_RESAMPLING_VISITS', 299), self.assertRaisesRegex(gc.GroupedComparisonError, 'work'):
            gc.evaluate_grouped_comparison(study, records(study, specification), specification)
        with patch.object(gc, 'MAX_METRIC_CELLS', 5), self.assertRaises(gc.GroupedComparisonError):
            gc.evaluate_grouped_comparison(study, (), specification)
        with patch.object(gc, 'MAX_RECORDS', 5), self.assertRaises(gc.GroupedComparisonError):
            gc.evaluate_grouped_comparison(study, records(study, specification), specification)

    def check_test_strict_input_roundtrip_and_atomic_export_preserve_prior_bytes(self):
        study = manifest()
        specification = plan(study)
        rows = records(study, specification)
        with tempfile.TemporaryDirectory() as temporary:
            source, destination = Path(temporary) / 'input.json', Path(temporary) / 'report.json'
            source.write_text(json.dumps({'manifest': study.to_dict(), 'plan': specification.to_dict(),
                'records': [r.to_dict() for r in rows]}), encoding='utf-8')
            loaded = gc.load_comparison_inputs(source)
            self.assertEqual(loaded, (study, rows, specification))
            report = gc.export_grouped_comparison(destination, *loaded)
            self.assertEqual(json.loads(destination.read_bytes()), report)
            old = destination.read_bytes()
            with patch.object(gc.os, 'replace', side_effect=OSError('disk failure')), self.assertRaises(OSError):
                gc.export_grouped_comparison(destination, *loaded)
            self.assertEqual(destination.read_bytes(), old)
            self.assertEqual(list(Path(temporary).glob('*.tmp')), [])
            source.write_bytes(b'{"manifest":{},"manifest":{},"plan":{},"records":[]}')
            with self.assertRaisesRegex(gc.GroupedComparisonError, 'Duplicate'):
                gc.load_comparison_inputs(source)

    def check_test_typed_malformed_nonfinite_and_boolean_inputs_reject(self):
        study = manifest()
        specification = plan(study)
        for replacement in ({'seed': True}, {'repetitions': 99}, {'minimum_coverage': float('nan')},
                            {'metrics': [specification.metrics[0]]}, {'minimum_independent_groups': 1}):
            with self.subTest(replacement=replacement), self.assertRaises(gc.GroupedComparisonError):
                replace(specification, **replacement)
        row = records(study, specification)[0]
        for replacement in ({'metrics': (('quality', float('inf')),)}, {'metrics': (('quality', True),)},
                            {'status': 'failed'}, {'output_revision': None}):
            with self.subTest(replacement=replacement), self.assertRaises(gc.GroupedComparisonError):
                replace(row, **replacement)
        payload = specification.to_dict()
        payload['unknown'] = 1
        with self.assertRaises(gc.GroupedComparisonError):
            gc.ComparisonPlan.from_dict(payload)

    def test_workflow_lifecycle(self):
        """6 related GroupedComparisonTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_constant_paired_shift_has_exact_bounds_and_never_certifies_science)
        self.scenario(self.check_test_seeded_bounds_match_independent_draw_matrix_oracle_and_rng_is_private)
        self.scenario(self.check_test_transitive_source_and_acquisition_links_never_resample_repeated_seeds)
        self.scenario(self.check_test_group_and_image_aggregation_have_prespecified_distinct_estimands)
        self.scenario(self.check_test_all_prespecified_critical_gates_are_required)
        self.scenario(self.check_test_strict_input_roundtrip_and_atomic_export_preserve_prior_bytes)

    def test_validation_and_recovery(self):
        """8 related GroupedComparisonTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_lower_error_uses_upper_bound_and_absolute_failure_is_not_equivalence)
        self.scenario(self.check_test_failed_abstained_and_missing_rows_stay_in_coverage_and_gallery)
        self.scenario(self.check_test_missing_reference_cannot_remove_poor_candidate_from_absolute_fitness)
        self.scenario(self.check_test_critical_missing_and_unknown_membership_are_inconclusive_not_dropped)
        self.scenario(self.check_test_identity_duplicate_endpoint_and_stratum_mismatches_reject)
        self.scenario(self.check_test_unknown_truth_and_missing_access_never_authorize_comparison)
        self.scenario(self.check_test_work_and_metric_budgets_reject_before_random_draws)
        self.scenario(self.check_test_typed_malformed_nonfinite_and_boolean_inputs_reject)
