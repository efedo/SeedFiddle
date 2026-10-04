from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from seedvision.learning.metrics import _global_matches, evaluate_instances
from seedvision.segmentation import procedural_fit


class AssignmentBudgetTests(unittest.TestCase):
    def test_large_one_column_evaluator_component_is_rejected_before_matrix_allocation(self) -> None:
        truth = np.arange(1, 10_001, dtype=np.int32).reshape(100, 100)
        prediction = np.ones_like(truth)
        real_zeros = np.zeros

        def reject_dense_assignment(shape, *args, **kwargs):
            if isinstance(shape, tuple) and len(shape) == 2 and shape[0] >= 10_000:
                raise AssertionError(f"dense matrix allocated before budget rejection: {shape}")
            return real_zeros(shape, *args, **kwargs)

        # The one-to-many component has only 10,000 pair weights, but the
        # solver's 10,000 dummy columns would expand its cost matrix to about
        # 800 MB. The preflight must run before either matrix is materialized.
        with patch.object(np, "zeros", new=reject_dense_assignment):
            with self.assertRaisesRegex(MemoryError, "assignment budget"):
                evaluate_instances(truth, prediction, iou_threshold=0.00005)

    def test_direct_solver_preflights_dummy_columns_before_cost_allocation(self) -> None:
        weights = np.ones((10_000, 1), dtype=np.float64)
        def reject_cost_allocation(shape, *args, **kwargs):
            raise AssertionError("solver allocated costs before preflight")

        with patch.object(np, "zeros", new=reject_cost_allocation):
            with self.assertRaisesRegex(MemoryError, "assignment budget"):
                procedural_fit._maximum_weight_assignment(weights)

    def test_small_ambiguous_assignments_preserve_cardinality_then_weight(self) -> None:
        weights = np.array(
            [
                [4.0, 3.9],
                [3.9, 0.0],
            ],
            dtype=np.float64,
        )
        actual = set(procedural_fit._maximum_weight_assignment(weights))
        self.assertEqual(actual, {(0, 1), (1, 0)})

        # The evaluator still uses the same shared assignment for normal
        # partitions; label IDs are arbitrary and all two objects match.
        truth = np.array([[7, 7, 0, 11, 11]], dtype=np.int32)
        prediction = np.array([[900, 900, 0, 2, 2]], dtype=np.int32)
        metrics = evaluate_instances(truth, prediction)
        self.assertEqual(metrics.true_positives, 2)
        self.assertEqual(metrics.false_positives, 0)
        self.assertEqual(metrics.false_negatives, 0)
        self.assertEqual(metrics.panoptic_quality, 1.0)

    def test_global_matcher_uses_the_same_workspace_preflight(self) -> None:
        pairs = [(0.0001, identifier, 1, 1) for identifier in range(1, 10_001)]
        with self.assertRaisesRegex(MemoryError, "assignment budget"):
            _global_matches(pairs, threshold=0.00005)

    def test_near_budget_evaluator_writes_negative_costs_into_preallocated_matrix(self) -> None:
        pairs = [
            (0.75, true_id, predicted_id, 1)
            for true_id in range(1, 9)
            for predicted_id in range(1, 21)
        ]
        workspace = procedural_fit._preflight_maximum_weight_assignment(
            8,
            20,
            include_weights=True,
        )
        real_negative = np.negative
        calls = 0

        def require_output_buffer(values, *, out=None, **kwargs):
            nonlocal calls
            if out is None:
                raise AssertionError("negative assignment costs allocated a temporary matrix")
            calls += 1
            return real_negative(values, out=out, **kwargs)

        with (
            patch.object(procedural_fit, "MAX_ASSIGNMENT_WORKSPACE_BYTES", workspace),
            patch.object(np, "negative", new=require_output_buffer),
        ):
            matches = _global_matches(pairs, threshold=0.5)

        self.assertEqual(len(matches), 8)
        self.assertEqual(calls, 1)


if __name__ == "__main__":
    unittest.main()
