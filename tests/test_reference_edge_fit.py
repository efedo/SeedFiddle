from __future__ import annotations

import unittest

import numpy as np

from seedvision.segmentation.reference_edge_fit import (
    ReferenceEdgeFitParameter,
    ReferenceEdgeFitScore,
    fit_reference_edge_parameters,
    score_reference_edge_probabilities,
)
from seedvision.visualization import AnalysisLayerSettings


class ReferenceEdgeFitTests(unittest.TestCase):
    def test_balanced_score_rewards_correct_support_and_rejects_cross_matches(self) -> None:
        physical_target = np.asarray(((1, 1), (0, 0)), dtype=bool)
        nonphysical_target = ~physical_target
        good = score_reference_edge_probabilities(
            np.asarray(((0.95, 0.85), (0.05, 0.10)), np.float32),
            np.asarray(((0.08, 0.12), (0.90, 0.95)), np.float32),
            physical_target,
            nonphysical_target,
        )
        crossed = score_reference_edge_probabilities(
            np.asarray(((0.10, 0.15), (0.90, 0.95)), np.float32),
            np.asarray(((0.90, 0.85), (0.10, 0.05)), np.float32),
            physical_target,
            nonphysical_target,
        )

        self.assertLess(good.loss, crossed.loss)
        self.assertGreater(good.physical_recall, 0.85)
        self.assertGreater(good.nonphysical_recall, 0.85)
        self.assertLess(good.physical_cross_match, 0.15)
        self.assertLess(good.nonphysical_cross_match, 0.15)

    def test_bounded_coordinate_search_proposes_only_improving_edge_settings(self) -> None:
        initial = AnalysisLayerSettings(reference_edge_similarity_scale=1.0)

        def evaluate(settings: AnalysisLayerSettings) -> ReferenceEdgeFitScore:
            loss = (settings.reference_edge_similarity_scale - 1.6) ** 2
            return ReferenceEdgeFitScore(
                loss=loss,
                physical_recall=1.0 - min(loss, 1.0),
                nonphysical_recall=1.0 - min(loss, 1.0),
                physical_cross_match=min(loss, 1.0),
                nonphysical_cross_match=min(loss, 1.0),
                physical_samples=20,
                nonphysical_samples=20,
            )

        report = fit_reference_edge_parameters(
            initial,
            evaluate,
            maximum_evaluations=7,
            passes=2,
            parameters=(
                ReferenceEdgeFitParameter(
                    "reference_edge_similarity_scale", 0.35, 2.5, 0.60
                ),
            ),
        )

        self.assertTrue(report.improved)
        self.assertLess(report.proposed_score.loss, report.initial_score.loss)
        self.assertAlmostEqual(
            report.proposed_settings.reference_edge_similarity_scale, 1.6
        )
        self.assertLessEqual(report.evaluations, 7)

    def test_fit_refuses_targets_missing_either_edge_class(self) -> None:
        with self.assertRaisesRegex(ValueError, "both annotated physical"):
            score_reference_edge_probabilities(
                np.ones((3, 3), np.float32),
                np.zeros((3, 3), np.float32),
                np.ones((3, 3), bool),
                np.zeros((3, 3), bool),
            )


if __name__ == "__main__":
    unittest.main()
