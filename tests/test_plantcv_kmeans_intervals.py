"""Certified compact numeric decisions and actual pinned scalar-state parity."""
from dataclasses import asdict
from fractions import Fraction
import itertools
import math
import unittest
from unittest.mock import patch

from seedvision.cuda import plantcv_kmeans_intervals as native
from tests.test_plantcv_kmeans_control import _bits, _fixtures


class KMeansIntervalTests(unittest.TestCase):
    def test_point_sequences_preserve_actual_pinned_convergence_bits_and_branch_order(self):
        request, reference = _fixtures()
        for item, expected in zip(request['convergence'], reference['convergence']):
            point = native.F64Interval.point
            control = native.IntervalMiniBatchControl(item['batch'], tol=point(item['tol']),
                max_no_improvement=item['limit'],
                ewa_inertia=None if item.get('ewa') is None else point(item['ewa']),
                ewa_inertia_min=None if item.get('minimum') is None else point(item['minimum']),
                no_improvement=item.get('counter', 0))
            for step, ((inertia, diff), state) in enumerate(zip(item['values'], expected)):
                stop = control.convergence(step=step, n_steps=len(item['values']),
                    n_samples=item['samples'], batch_inertia=point(inertia), centres_squared_diff=point(diff))
                def bits(value):
                    self.assertTrue(value is None or value.is_point)
                    return None if value is None else _bits(value.lo)
                self.assertEqual([stop, bits(control.ewa_inertia), bits(control.ewa_inertia_min),
                    control.no_improvement], state)

    def test_strict_interval_improvement_equality_and_overlap_are_certified(self):
        interval, point = native.F64Interval, native.F64Interval.point
        self.assertTrue(native.strict_interval_less(interval(7, 8), interval(9, 10)))
        self.assertFalse(native.strict_interval_less(interval(10, 11), interval(9, 10)))
        self.assertFalse(native.strict_interval_less(point(10), point(10)))
        for left, right in ((interval(9, 11), point(10)), (interval(9, 10), interval(9, 10))):
            with self.assertRaises(native.KMeansNumericalAmbiguity):
                native.strict_interval_less(left, right)
        control = native.IntervalMiniBatchControl(1, ewa_inertia=point(10),
            ewa_inertia_min=point(10), no_improvement=7)
        before = asdict(control)
        with self.assertRaises(native.KMeansNumericalAmbiguity):
            control.convergence(step=1, n_steps=3, n_samples=1, batch_inertia=interval(9, 11))
        self.assertEqual(asdict(control), before)
        self.assertFalse(control.convergence(step=1, n_steps=3, n_samples=1, batch_inertia=interval(7, 8)))
        self.assertEqual(control.no_improvement, 0)
        self.assertLess(control.ewa_inertia.hi, 9)

    def test_interval_tolerance_stop_bypass_and_ambiguity_preserve_state_order(self):
        interval, point = native.F64Interval, native.F64Interval.point
        for tolerance, difference, expected in ((point(1), point(1), True),
                (interval(.9, 1.1), interval(.5, .8), True),
                (interval(.9, 1.1), interval(1.2, 2), False),
                (point(0), None, False), (interval(0, 1), point(2), False)):
            control = native.IntervalMiniBatchControl(1, tol=tolerance,
                ewa_inertia=point(3), ewa_inertia_min=point(2), no_improvement=7)
            stop = control.convergence(step=1, n_steps=3, n_samples=1,
                batch_inertia=point(4), centres_squared_diff=difference)
            self.assertEqual(stop, expected)
            self.assertEqual(control.ewa_inertia, point(4))
            self.assertEqual(control.no_improvement, 7 if expected else 8)
            self.assertEqual(control.ewa_inertia_min, point(2))
        for tolerance, difference in ((point(1), interval(1, 2)),
                (interval(0, 1), point(0)), (interval(.9, 1.1), point(1))):
            control = native.IntervalMiniBatchControl(1, tol=tolerance)
            before = asdict(control)
            with self.assertRaises(native.KMeansNumericalAmbiguity):
                control.convergence(step=1, n_steps=3, n_samples=1,
                    batch_inertia=point(4), centres_squared_diff=difference)
            self.assertEqual(asdict(control), before)

    def test_point_operations_match_binary64_and_nonpoint_endpoints_enclose_operations(self):
        interval, point = native.F64Interval, native.F64Interval.point
        self.assertEqual(_bits(native.interval_add(point(.1), point(.2)).lo), _bits(.1+.2))
        self.assertEqual(_bits(native.interval_scale(point(.1), .3).lo), _bits(.1*.3))
        self.assertEqual(_bits(native.interval_divide(point(.1), 7).lo), _bits(.1/7))
        for values in ((0., math.ulp(0.)), (.1, .3), (1e-300, 1e-299), (100., 101.)):
            source = interval(*values)
            for factor in (0., math.ulp(0.), .25, 1., 17.):
                result = native.interval_scale(source, factor)
                for value in values:
                    self.assertLessEqual(result.lo, value*factor)
                    self.assertGreaterEqual(result.hi, value*factor)
            result = native.interval_divide(source, 7)
            for value in values:
                self.assertLessEqual(result.lo, value/7)
                self.assertGreaterEqual(result.hi, value/7)
            result = native.interval_add(source, interval(.1, .2))
            for left in values:
                for right in (.1, .2):
                    self.assertLessEqual(result.lo, left+right)
                    self.assertGreaterEqual(result.hi, left+right)

    def test_directed_fraction_conversion_handles_inexact_and_subnormal_boundaries(self):
        for value in (Fraction(1, 10), Fraction(1, 2**1075), Fraction(1, 2**1074), Fraction(7, 3)):
            lower = native._directed_fraction(value, upper=False)
            upper = native._directed_fraction(value, upper=True)
            self.assertLessEqual(Fraction.from_float(lower), value)
            self.assertGreaterEqual(Fraction.from_float(upper), value)
            self.assertLessEqual(upper-lower, math.ulp(upper))

    def test_fractional_inertia_bounds_all_small_permutations_and_binary_groupings(self):
        values = [65025., 1e-12, 1e-12, 17.125, .5]
        serial = 0.
        for value in values:
            serial += value
        def sums(items):
            if len(items) == 1:
                return {items[0]}
            result = set()
            for split in range(1, len(items)):
                result.update(a+b for a in sums(items[:split]) for b in sums(items[split:]))
            return result
        for threads in (None, 1, 2, 4, 32):
            bound = native.source_inertia_interval(serial, row_count=len(values),
                feature_count=1, source_n_threads=threads, integral=False)
            self.assertFalse(bound.is_point)
            for order in itertools.permutations(values):
                for total in sums(order):
                    self.assertLessEqual(bound.lo, total)
                    self.assertGreaterEqual(bound.hi, total)
        # Identical thread provenance never turns fractional arithmetic into an
        # invented point value, including one row with zero observed loss.
        zero = native.source_inertia_interval(0., row_count=1, feature_count=1,
            source_n_threads=1, integral=False)
        self.assertEqual(zero.lo, 0.)
        self.assertGreater(zero.hi, 0.)

    def test_integral_inertia_is_exact_for_all_thread_counts_within_proved_domain(self):
        total = 1_000_000 * 4096 * 65025
        for threads in (None, 1, 2, 4, 1024):
            interval = native.source_inertia_interval(float(total), row_count=1_000_000,
                feature_count=4096, source_n_threads=threads, integral=True)
            self.assertEqual(interval, native.F64Interval.point(total))

    def test_all_reduction_profile_preserves_the_same_proof_without_inferred_provenance(self):
        self.assertEqual(native.source_reduction_profile(None), 'all-valid-openmp-reductions-v1')
        for threads in (1, 2, 4, 1024):
            self.assertEqual(native.source_reduction_profile(threads), 'recorded-openmp-threads-v1')
            for integral in (False, True):
                arguments = dict(row_count=17, feature_count=3, integral=integral)
                self.assertEqual(native.source_inertia_interval(123., source_n_threads=None, **arguments),
                    native.source_inertia_interval(123., source_n_threads=threads, **arguments))
        for value in (True, False, 0, -1, 1.5, '1'):
            with self.assertRaises(native.KMeansIntervalError):
                native.source_reduction_profile(value)

    def test_invalid_inputs_mutable_state_and_loaded_source_fail_explicitly(self):
        for values in ((-1, 0), (2, 1), (float('nan'), 1), (0, float('inf')), (True, 1)):
            with self.assertRaises(native.KMeansIntervalError):
                native.F64Interval(*values)
        for kwargs in (dict(row_count=0), dict(feature_count=4097), dict(source_n_threads=True),
                dict(source_n_threads=0), dict(integral=1), dict(serial_sum=65026.)):
            arguments = dict(serial_sum=1., row_count=1, feature_count=1, source_n_threads=1, integral=True)
            arguments.update(kwargs)
            with self.assertRaises(native.KMeansIntervalError):
                native.source_inertia_interval(**arguments)
        control = native.IntervalMiniBatchControl(1)
        self.assertFalse(control.convergence(step=0, n_steps=2, n_samples=1,
            batch_inertia=native.F64Interval(1, 2)))
        self.assertIsNone(control.ewa_inertia)
        control.no_improvement = -1
        before = asdict(control)
        with self.assertRaises(native.KMeansIntervalError):
            control.convergence(step=1, n_steps=2, n_samples=1, batch_inertia=native.F64Interval.point(2))
        self.assertEqual(asdict(control), before)
        from seedvision.benchmarking.native_recipe import _module_record
        self.assertTrue(_module_record(native)['loaded_code'])
        with patch.object(native, 'EXACT_INTEGER_LIMIT', 7):
            with self.assertRaises(ValueError):
                _module_record(native)
