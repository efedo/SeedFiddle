from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from seedvision.segmentation.procedural import ProceduralInstanceSettings
from seedvision.segmentation.procedural_fit import (
    ProceduralFitOptions,
    ProceduralFitParameter,
    fit_procedural_settings,
    score_procedural_instances,
)
import seedvision.segmentation.procedural_fit as procedural_fit_module


class ProceduralFitScoreTests(unittest.TestCase):
    @staticmethod
    def _single_seed() -> np.ndarray:
        labels = np.zeros((48, 48), np.uint16)
        cv2.rectangle(labels, (14, 14), (33, 33), 1, -1)
        return labels

    def test_perfect_relabelled_prediction_has_zero_loss(self) -> None:
        truth = self._single_seed()
        prediction = np.where(truth > 0, 71, 0).astype(np.uint16)
        score = score_procedural_instances(truth, prediction)
        self.assertTrue(score.perfect_fit)
        self.assertEqual(score.matched_instances, 1)
        self.assertEqual(score.false_positive_pixels, 0)
        self.assertEqual(score.distance_weighted_false_positive_pixels, 0.0)
        self.assertEqual(score.false_negative_pixels, 0)
        expected_diameter = 2.0 * math.sqrt(400.0 / math.pi)
        self.assertAlmostEqual(
            score.overreach_distance_scale_px,
            0.50 * expected_diameter,
        )

    def test_equal_false_positive_area_costs_more_than_false_negative_area(self) -> None:
        truth = self._single_seed()
        false_positive = truth.copy()
        false_positive[10:14, 14:24] = 1
        false_negative = truth.copy()
        false_negative[14:18, 14:24] = 0
        # A deliberately short distance scale makes this four-pixel-wide leak
        # a substantive overreach rather than a sub-pixel boundary tolerance.
        weighting = {"overreach_distance_scale_fraction": 0.05}
        extra_score = score_procedural_instances(
            truth, false_positive, **weighting
        )
        missed_score = score_procedural_instances(
            truth, false_negative, **weighting
        )
        self.assertEqual(extra_score.false_positive_pixels, 40)
        self.assertEqual(missed_score.false_negative_pixels, 40)
        self.assertGreater(extra_score.loss, missed_score.loss)

    def test_merged_adjacent_seed_is_penalized_as_leak_and_missed_instance(self) -> None:
        truth = np.zeros((50, 64), np.uint16)
        cv2.circle(truth, (22, 25), 10, 1, -1)
        cv2.circle(truth, (42, 25), 10, 2, -1)
        exact = score_procedural_instances(truth, truth)
        merged = np.where(truth > 0, 1, 0).astype(np.uint16)
        merged_score = score_procedural_instances(truth, merged)
        self.assertTrue(exact.perfect_fit)
        self.assertGreater(merged_score.false_positive_pixels, 0)
        self.assertEqual(merged_score.false_negative_instances, 1)
        self.assertGreater(merged_score.loss, exact.loss)

    def test_disjoint_predictions_are_optional_for_partial_annotation_sets(self) -> None:
        truth = self._single_seed()
        prediction = truth.copy()
        cv2.circle(prediction, (5, 5), 3, 2, -1)
        partial = score_procedural_instances(truth, prediction)
        complete = score_procedural_instances(
            truth, prediction, annotations_are_complete=True
        )
        self.assertTrue(partial.perfect_fit)
        self.assertEqual(partial.evaluated_predictions, 1)
        self.assertEqual(partial.false_positive_instances, 0)
        self.assertGreater(complete.false_positive_pixels, 0)
        self.assertEqual(complete.evaluated_predictions, 2)
        self.assertEqual(complete.false_positive_instances, 1)
        self.assertGreater(complete.loss, partial.loss)


class ProceduralFitDistanceWeightedOverreachTests(unittest.TestCase):
    """Lock the exponential, instance-aware overreach objective."""

    @staticmethod
    def _square_truth() -> np.ndarray:
        truth = np.zeros((96, 96), np.uint16)
        truth[40:50, 40:50] = 1
        return truth

    @staticmethod
    def _fraction_for_scale(truth: np.ndarray, scale_px: float) -> float:
        identifiers, areas = np.unique(truth[truth > 0], return_counts=True)
        if not len(identifiers):
            raise AssertionError("Test helper requires at least one target instance.")
        diameters = 2.0 * np.sqrt(np.asarray(areas, dtype=float) / math.pi)
        return float(scale_px) / max(1.0, float(np.median(diameters)))

    def _one_extra_pixel_score(
        self, distance_px: int, *, scale_px: float = 2.0
    ):
        truth = self._square_truth()
        prediction = truth.copy()
        # The target's rightmost pixel is x=49, so this sample is exactly the
        # requested centre-to-centre distance from the target raster.
        prediction[45, 49 + int(distance_px)] = 1
        fraction = self._fraction_for_scale(truth, scale_px)
        return score_procedural_instances(
            truth,
            prediction,
            overreach_distance_scale_fraction=fraction,
        )

    def test_equal_raw_overreach_costs_more_when_farther_from_target(self) -> None:
        adjacent = self._one_extra_pixel_score(1)
        far = self._one_extra_pixel_score(8)

        self.assertEqual(adjacent.false_positive_pixels, 1)
        self.assertEqual(far.false_positive_pixels, 1)
        self.assertEqual(adjacent.pixel_precision, far.pixel_precision)
        self.assertGreater(
            far.distance_weighted_false_positive_pixels,
            adjacent.distance_weighted_false_positive_pixels,
        )
        self.assertGreater(far.loss, adjacent.loss)
        self.assertAlmostEqual(adjacent.overreach_distance_scale_px, 2.0)
        self.assertAlmostEqual(far.overreach_distance_scale_px, 2.0)

    def test_one_scale_increments_follow_exponential_ratio_and_cap(self) -> None:
        weights = [
            self._one_extra_pixel_score(distance).distance_weighted_false_positive_pixels
            for distance in (2, 4, 6)
        ]
        np.testing.assert_allclose(weights, [1.0, 3.0, 7.0], rtol=1e-5, atol=1e-5)
        self.assertAlmostEqual((weights[1] + 1.0) / (weights[0] + 1.0), 2.0)
        self.assertAlmostEqual((weights[2] + 1.0) / (weights[1] + 1.0), 2.0)

        at_cap = self._one_extra_pixel_score(16)
        beyond_cap = self._one_extra_pixel_score(30)
        self.assertAlmostEqual(
            at_cap.distance_weighted_false_positive_pixels,
            math.expm1(math.log(2.0) * 8.0),
            places=4,
        )
        self.assertAlmostEqual(
            beyond_cap.distance_weighted_false_positive_pixels,
            at_cap.distance_weighted_false_positive_pixels,
            places=4,
        )

    def test_false_negative_objective_and_raw_metrics_ignore_distance_scale(self) -> None:
        truth = self._square_truth()
        prediction = truth.copy()
        prediction[40:44, 40:50] = 0

        short = score_procedural_instances(
            truth,
            prediction,
            overreach_distance_scale_fraction=0.05,
        )
        long = score_procedural_instances(
            truth,
            prediction,
            overreach_distance_scale_fraction=2.0,
        )
        self.assertEqual(short.false_positive_pixels, 0)
        self.assertEqual(short.distance_weighted_false_positive_pixels, 0.0)
        self.assertEqual(short.false_negative_pixels, 40)
        self.assertEqual(short.loss, long.loss)
        self.assertEqual(short.pixel_precision, long.pixel_precision)
        self.assertEqual(short.pixel_recall, long.pixel_recall)
        self.assertNotEqual(
            short.overreach_distance_scale_px,
            long.overreach_distance_scale_px,
        )

    def test_adjacent_seed_leak_uses_its_matched_target_not_target_union(self) -> None:
        truth = np.zeros((80, 100), np.uint16)
        truth[30:41, 10:21] = 11
        truth[30:41, 60:71] = 22
        prediction = np.zeros_like(truth)
        prediction[truth == 11] = 501
        prediction[truth == 22] = 902
        # Prediction 501 steals one pixel well inside seed 22. Its distance is
        # zero to the target union, but 45 px from its own matched seed 11.
        prediction[35, 65] = 501

        score = score_procedural_instances(
            truth,
            prediction,
            overreach_distance_scale_fraction=0.50,
        )
        self.assertEqual(score.matched_instances, 2)
        self.assertEqual(score.false_positive_instances, 0)
        self.assertEqual(score.false_negative_instances, 0)
        self.assertEqual(score.false_positive_pixels, 1)
        self.assertEqual(score.false_negative_pixels, 1)
        expected = math.expm1(
            math.log(2.0)
            * min(45.0 / score.overreach_distance_scale_px, 8.0)
        )
        self.assertAlmostEqual(
            score.distance_weighted_false_positive_pixels,
            expected,
            places=4,
        )
        self.assertGreater(score.distance_weighted_false_positive_pixels, 1.0)

    def test_disjoint_prediction_is_weighted_only_for_complete_annotations(self) -> None:
        truth = self._square_truth()
        prediction = truth.copy()
        prediction[5, 5] = 2

        partial = score_procedural_instances(truth, prediction)
        complete = score_procedural_instances(
            truth,
            prediction,
            annotations_are_complete=True,
        )
        self.assertTrue(partial.perfect_fit)
        self.assertEqual(partial.evaluated_predictions, 1)
        self.assertEqual(partial.false_positive_pixels, 0)
        self.assertEqual(partial.distance_weighted_false_positive_pixels, 0.0)
        self.assertEqual(complete.evaluated_predictions, 2)
        self.assertEqual(complete.false_positive_instances, 1)
        self.assertEqual(complete.false_positive_pixels, 1)
        self.assertGreater(complete.distance_weighted_false_positive_pixels, 1.0)
        self.assertGreater(complete.loss, partial.loss)

    def test_empty_prediction_and_wholly_disjoint_partial_prediction(self) -> None:
        truth = self._square_truth()
        empty = np.zeros_like(truth)
        empty_score = score_procedural_instances(
            truth,
            empty,
            overreach_distance_scale_fraction=0.05,
        )
        self.assertEqual(empty_score.matched_instances, 0)
        self.assertEqual(empty_score.false_positive_pixels, 0)
        self.assertEqual(empty_score.distance_weighted_false_positive_pixels, 0.0)
        self.assertEqual(empty_score.false_negative_pixels, 100)

        disjoint = np.zeros_like(truth)
        disjoint[4:7, 4:7] = 9
        partial = score_procedural_instances(truth, disjoint)
        complete = score_procedural_instances(
            truth,
            disjoint,
            annotations_are_complete=True,
        )
        self.assertEqual(partial.evaluated_predictions, 0)
        self.assertEqual(partial.false_positive_pixels, 0)
        self.assertEqual(partial.distance_weighted_false_positive_pixels, 0.0)
        self.assertEqual(partial.false_negative_pixels, 100)
        self.assertEqual(complete.evaluated_predictions, 1)
        self.assertEqual(complete.false_positive_pixels, 9)
        self.assertGreater(complete.distance_weighted_false_positive_pixels, 0.0)
        self.assertEqual(complete.false_negative_pixels, 100)

        with self.assertRaisesRegex(ValueError, "annotated seed"):
            score_procedural_instances(empty, disjoint)

    def test_distance_scale_fraction_is_validated_by_options_and_scorer(self) -> None:
        truth = self._square_truth()
        for invalid in (0.0, -0.1, float("nan"), float("inf")):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    ProceduralFitOptions(
                        overreach_distance_scale_fraction=invalid
                    )
                with self.assertRaises(ValueError):
                    score_procedural_instances(
                        truth,
                        truth,
                        overreach_distance_scale_fraction=invalid,
                    )
        for invalid_diameter in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(seed_diameter_px=invalid_diameter):
                with self.assertRaises(ValueError):
                    score_procedural_instances(
                        truth,
                        truth,
                        seed_diameter_px=invalid_diameter,
                    )

    def test_effective_distance_scale_has_a_one_pixel_floor(self) -> None:
        truth = np.zeros((12, 12), np.uint16)
        truth[6, 6] = 1
        prediction = truth.copy()
        prediction[6, 7] = 1
        score = score_procedural_instances(
            truth,
            prediction,
            overreach_distance_scale_fraction=0.01,
        )
        self.assertEqual(score.overreach_distance_scale_px, 1.0)
        self.assertAlmostEqual(
            score.distance_weighted_false_positive_pixels,
            1.0,
        )

    def test_fit_options_propagate_distance_scale_to_every_score(self) -> None:
        truth = self._square_truth()
        fraction = 0.25
        options = ProceduralFitOptions(
            maximum_evaluations=1,
            overreach_distance_scale_fraction=fraction,
        )
        result = fit_procedural_settings(
            truth,
            lambda _settings: truth,
            options=options,
            seed_diameter_px=20.0,
        )
        self.assertEqual(result.evaluations, 1)
        self.assertTrue(result.initial_score.perfect_fit)
        self.assertAlmostEqual(
            result.initial_score.overreach_distance_scale_px,
            fraction * 20.0,
        )
        self.assertEqual(
            result.initial_score.distance_weighted_false_positive_pixels,
            0.0,
        )


class ProceduralSettingsFitTests(unittest.TestCase):
    def test_annotation_distance_context_is_prepared_once_per_fit(self) -> None:
        truth = np.zeros((64, 64), np.uint16)
        cv2.circle(truth, (32, 32), 12, 1, -1)
        prediction = truth.copy()
        prediction[20:24, :] = 0
        options = ProceduralFitOptions(maximum_evaluations=4, passes=1)

        with patch(
            "seedvision.segmentation.procedural_fit._prepare_scoring_context",
            wraps=procedural_fit_module._prepare_scoring_context,
        ) as prepare:
            result = fit_procedural_settings(
                truth,
                lambda _settings: prediction,
                options=options,
                seed_diameter_px=24.0,
            )

        self.assertEqual(result.evaluations, 4)
        self.assertEqual(prepare.call_count, 1)

    def test_coordinate_search_improves_without_mutating_initial_settings(self) -> None:
        truth = np.zeros((64, 64), np.uint16)
        cv2.circle(truth, (32, 32), 12, 1, -1)
        initial = ProceduralInstanceSettings(foreground_threshold_scale=0.82)
        observed_values: list[float] = []

        def evaluate(settings: ProceduralInstanceSettings) -> np.ndarray:
            observed_values.append(settings.foreground_threshold_scale)
            # The next positive search coordinate (0.98) is the exact synthetic
            # optimum; lower thresholds over-include and are deliberately more
            # expensive than a comparably small miss.
            radius = max(
                2,
                int(round(12.0 + (0.98 - settings.foreground_threshold_scale) * 20.0)),
            )
            prediction = np.zeros_like(truth)
            cv2.circle(prediction, (32, 32), radius, 1, -1)
            return prediction

        options = ProceduralFitOptions(
            maximum_evaluations=3,
            passes=1,
            parameters=(
                ProceduralFitParameter(
                    "foreground_threshold_scale", 0.55, 1.25, 0.16
                ),
            ),
        )
        result = fit_procedural_settings(
            truth,
            evaluate,
            initial_settings=initial,
            options=options,
        )
        self.assertTrue(result.improved)
        self.assertTrue(result.proposed_score.perfect_fit)
        self.assertAlmostEqual(result.proposed_settings.foreground_threshold_scale, 0.98)
        self.assertAlmostEqual(initial.foreground_threshold_scale, 0.82)
        self.assertLessEqual(result.evaluations, options.maximum_evaluations)
        np.testing.assert_allclose(observed_values, [0.82, 0.66, 0.98])

    def test_fit_is_deterministic_and_honours_evaluation_budget(self) -> None:
        truth = np.zeros((40, 40), np.uint16)
        cv2.circle(truth, (20, 20), 8, 1, -1)

        def evaluate(settings: ProceduralInstanceSettings) -> np.ndarray:
            prediction = np.zeros_like(truth)
            radius = int(round(5 + settings.marker_count_multiplier * 3))
            cv2.circle(prediction, (20, 20), radius, 1, -1)
            return prediction

        options = ProceduralFitOptions(maximum_evaluations=5, passes=3)
        first = fit_procedural_settings(truth, evaluate, options=options)
        second = fit_procedural_settings(truth, evaluate, options=options)
        self.assertLessEqual(first.evaluations, 5)
        self.assertEqual(first.proposed_settings, second.proposed_settings)
        self.assertEqual(first.proposed_score, second.proposed_score)
        self.assertEqual(first.trials, second.trials)

    def test_invalid_weighting_cannot_make_false_positives_cheaper(self) -> None:
        with self.assertRaisesRegex(ValueError, "must exceed"):
            ProceduralFitOptions(
                false_positive_weight=1.0,
                false_negative_weight=1.0,
            )


if __name__ == "__main__":
    unittest.main()
