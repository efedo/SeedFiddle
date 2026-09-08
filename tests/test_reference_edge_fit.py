from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from seedvision.segmentation.reference_edge_fit import (
    DEFAULT_REFERENCE_EDGE_FIT_PARAMETERS,
    ReferenceEdgeFitParameter,
    ReferenceEdgeFitScore,
    fit_reference_edge_parameters,
    evaluate_reference_edge_settings,
    score_reference_edge_probabilities,
)
from seedvision.visualization import AnalysisLayerSettings


class ReferenceEdgeFitTests(unittest.TestCase):
    def test_internal_prototype_sources_are_prominent_normal_maxima(self) -> None:
        import torch

        from seedvision.cuda.layers import _candidate_internal_edge_mask

        shape = (1, 1, 11, 13)
        edge = torch.zeros(shape, dtype=torch.float32)
        edge[:, :, :, 4] = 0.15
        edge[:, :, :, 5] = 0.55
        edge[:, :, :, 6] = 1.00
        edge[:, :, :, 7] = 0.55
        edge[:, :, :, 8] = 0.15
        ridge = torch.zeros_like(edge)
        # Simulate the wide low-valued halo produced by bilinear restoration of
        # a one-pixel ridge. Only its actual normal-direction maximum may train
        # an edge prototype.
        ridge[:, :, :, 4:9] = 0.20
        ridge[:, :, :, 6] = 1.00
        # A disconnected low ridge used to bypass the adaptive prominence
        # threshold merely because its value exceeded 0.05.
        ridge[:, :, :, 10] = 0.06
        tangent_x = torch.zeros_like(edge)
        tangent_y = torch.ones_like(edge)
        safe = torch.ones_like(edge, dtype=torch.bool)

        selected = _candidate_internal_edge_mask(
            edge,
            ridge,
            tangent_x,
            tangent_y,
            safe,
            ridge_weight=0.35,
        )[0, 0]

        self.assertTrue(bool(torch.all(selected[:, 6]).item()))
        self.assertEqual(int(selected.sum().item()), selected.shape[0])
        self.assertFalse(bool(selected[:, 10].any().item()))

    def test_default_fit_optimizes_edge_class_contrast(self) -> None:
        names = tuple(
            parameter.name for parameter in DEFAULT_REFERENCE_EDGE_FIT_PARAMETERS
        )
        self.assertIn("reference_edge_class_contrast", names)

        def evaluate(settings: AnalysisLayerSettings) -> ReferenceEdgeFitScore:
            loss = (settings.reference_edge_class_contrast - 5.0) ** 2
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
            AnalysisLayerSettings(), evaluate, maximum_evaluations=5, passes=1
        )
        self.assertTrue(report.improved)
        self.assertEqual(report.proposed_settings.reference_edge_class_contrast, 5.0)

    def test_evaluation_trains_and_scores_disjoint_instance_ids(self) -> None:
        import torch

        from seedvision.cuda import CudaContext

        annotations = np.zeros((6, 8), np.uint16)
        annotations[:3, :2] = 1
        annotations[:3, 2:4] = 2
        annotations[3:, :2] = 3
        annotations[3:, 2:4] = 4
        captured: dict[str, np.ndarray] = {}

        class Field:
            def __init__(self, value: float) -> None:
                self.tensor = torch.full((1, 1, 6, 8), value)

            def gpu_tensor(self, *, device=None, dtype=None):
                return self.tensor.to(device=device, dtype=dtype)

        def prototypes(*_args, seed_instance_annotations=None, **_kwargs):
            captured["training"] = np.asarray(seed_instance_annotations).copy()
            return SimpleNamespace(
                physical_edge_field=Field(0.8),
                non_edge_field=Field(0.2),
            )

        def references(held_out, *_args, **_kwargs):
            captured["holdout"] = np.asarray(held_out).copy()
            locations = np.argwhere(np.asarray(held_out) > 0)
            physical = np.zeros((6, 8), bool)
            nonphysical = np.zeros((6, 8), bool)
            physical[tuple(locations[0])] = True
            nonphysical[tuple(locations[-1])] = True
            return SimpleNamespace(
                physical_edge=physical,
                non_edge=nonphysical,
            )

        gradients = SimpleNamespace(strength=torch.ones((1, 1, 6, 8)))
        with patch(
            "seedvision.cuda.layers.reference_texture_probabilities",
            side_effect=prototypes,
        ), patch(
            "seedvision.annotation.instance_references."
            "instance_boundary_references_from_edge_evidence",
            side_effect=references,
        ):
            evaluate_reference_edge_settings(
                np.zeros((6, 8, 3), np.uint8),
                gradients,
                np.zeros((6, 8), np.uint8),
                None,
                12.0,
                annotations,
                AnalysisLayerSettings(),
                cuda_context=CudaContext.resolve(requested="cpu"),
            )

        self.assertEqual(set(np.unique(captured["training"])), {0, 1, 3})
        self.assertEqual(set(np.unique(captured["holdout"])), {0, 2, 4})
        self.assertFalse(
            np.any(
                (captured["training"] > 0)
                & (captured["holdout"] > 0)
            )
        )
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
