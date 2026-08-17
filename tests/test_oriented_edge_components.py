from __future__ import annotations

import unittest


class OrientedEdgeComponentTests(unittest.TestCase):
    @staticmethod
    def _labels(
        mask,
        tangent_x,
        tangent_y,
        *,
        maximum_gap: int,
        tolerance: float = 24.0,
        curvature_policy: str = "off",
        curvature_tolerance: float = 6.0,
    ):
        from seedvision.cuda.ops import oriented_connected_components

        return oriented_connected_components(
            mask[None, None],
            tangent_x[None, None],
            tangent_y[None, None],
            maximum_gap=maximum_gap,
            tangent_tolerance_degrees=tolerance,
            curvature_policy=curvature_policy,
            curvature_tolerance_degrees=curvature_tolerance,
        )[0, 0]

    def test_gap_four_reconnects_facing_fragments_without_gap_two(self) -> None:
        import torch

        mask = torch.zeros((10, 14), dtype=torch.bool)
        mask[4, 1:4] = True
        mask[4, 7:10] = True
        tangent_x = torch.ones(mask.shape, dtype=torch.float32)
        tangent_y = torch.zeros_like(tangent_x)

        short = self._labels(
            mask, tangent_x, tangent_y, maximum_gap=2
        )
        repaired = self._labels(
            mask, tangent_x, tangent_y, maximum_gap=4
        )

        self.assertNotEqual(int(short[4, 2]), int(short[4, 8]))
        self.assertEqual(int(repaired[4, 2]), int(repaired[4, 8]))

    def test_gap_four_does_not_merge_uninterrupted_parallel_edges(self) -> None:
        """Regress the gap-2-separate/gap-4-merged seed-rim failure."""

        import torch

        mask = torch.zeros((12, 16), dtype=torch.bool)
        mask[3, 1:14] = True
        mask[6, 1:14] = True
        tangent_x = torch.ones(mask.shape, dtype=torch.float32)
        tangent_y = torch.zeros_like(tangent_x)

        gap_two = self._labels(
            mask, tangent_x, tangent_y, maximum_gap=2
        )
        gap_four = self._labels(
            mask, tangent_x, tangent_y, maximum_gap=4
        )

        self.assertNotEqual(int(gap_two[3, 4]), int(gap_two[6, 4]))
        self.assertNotEqual(int(gap_four[3, 4]), int(gap_four[6, 4]))

    def test_gap_bridge_cannot_enter_interior_of_very_close_parallel_edge(self) -> None:
        import torch

        # A four-pixel diagonal displacement between these rows is only
        # fourteen degrees off their tangents. Chord alignment alone therefore
        # cannot distinguish it from a real break; endpoint gating must.
        mask = torch.zeros((10, 16), dtype=torch.bool)
        mask[4, 1:14] = True
        mask[5, 1:14] = True
        tangent_x = torch.ones(mask.shape, dtype=torch.float32)
        tangent_y = torch.zeros_like(tangent_x)

        labels = self._labels(
            mask, tangent_x, tangent_y, maximum_gap=4
        )

        self.assertNotEqual(int(labels[4, 4]), int(labels[5, 4]))

    def test_gap_bridge_preserves_directed_gradient_side(self) -> None:
        import torch

        mask = torch.zeros((10, 14), dtype=torch.bool)
        mask[4, 1:4] = True
        mask[5, 7:10] = True
        tangent_x = torch.ones(mask.shape, dtype=torch.float32)
        tangent_x[5, 7:10] = -1.0
        tangent_y = torch.zeros_like(tangent_x)

        labels = self._labels(
            mask, tangent_x, tangent_y, maximum_gap=4
        )

        self.assertNotEqual(int(labels[4, 2]), int(labels[5, 8]))

    def test_gap_bridge_accepts_gently_curved_same_side_fragments(self) -> None:
        import torch

        mask = torch.zeros((10, 14), dtype=torch.bool)
        mask[4, 1:4] = True
        mask[5, 7:10] = True
        tangent_x = torch.ones(mask.shape, dtype=torch.float32)
        tangent_y = torch.zeros_like(tangent_x)

        labels = self._labels(
            mask, tangent_x, tangent_y, maximum_gap=4
        )

        self.assertEqual(int(labels[4, 2]), int(labels[5, 8]))

    def test_wide_tangent_tolerance_does_not_enable_lateral_bridge(self) -> None:
        import torch

        mask = torch.zeros((12, 16), dtype=torch.bool)
        mask[3, 1:4] = True
        mask[6, 7:10] = True
        tangent_x = torch.ones(mask.shape, dtype=torch.float32)
        tangent_y = torch.zeros_like(tangent_x)

        labels = self._labels(
            mask,
            tangent_x,
            tangent_y,
            maximum_gap=4,
            tolerance=60.0,
        )

        self.assertNotEqual(int(labels[3, 2]), int(labels[6, 8]))

    def test_curvature_policy_rejects_s_bridge_but_off_preserves_it(self) -> None:
        import torch

        mask = torch.zeros((10, 14), dtype=torch.bool)
        mask[4, 1:4] = True
        mask[5, 7:10] = True
        tangent_x = torch.ones(mask.shape, dtype=torch.float32)
        tangent_y = torch.zeros_like(tangent_x)

        legacy = self._labels(
            mask,
            tangent_x,
            tangent_y,
            maximum_gap=4,
            curvature_policy="off",
        )
        preferred = self._labels(
            mask,
            tangent_x,
            tangent_y,
            maximum_gap=4,
            curvature_policy="prefer",
        )
        required = self._labels(
            mask,
            tangent_x,
            tangent_y,
            maximum_gap=4,
            curvature_policy="require",
        )

        self.assertEqual(int(legacy[4, 2]), int(legacy[5, 8]))
        self.assertNotEqual(int(preferred[4, 2]), int(preferred[5, 8]))
        self.assertNotEqual(int(required[4, 2]), int(required[5, 8]))

    def test_prefer_allows_ambiguous_bend_that_require_rejects(self) -> None:
        import torch

        # A one-pixel rise across a five-pixel gap bends about 11.3 degrees
        # away from equal endpoint tangents: resolved at the six-degree
        # strict tolerance, but still ambiguous to the doubled Prefer gate.
        mask = torch.zeros((10, 14), dtype=torch.bool)
        mask[5, 0:3] = True
        mask[4, 7:10] = True
        tangent_x = torch.ones(mask.shape, dtype=torch.float32)
        tangent_y = torch.zeros_like(tangent_x)

        preferred = self._labels(
            mask,
            tangent_x,
            tangent_y,
            maximum_gap=5,
            curvature_policy="prefer",
        )
        required = self._labels(
            mask,
            tangent_x,
            tangent_y,
            maximum_gap=5,
            curvature_policy="require",
        )

        self.assertEqual(int(preferred[5, 1]), int(preferred[4, 8]))
        self.assertNotEqual(int(required[5, 1]), int(required[4, 8]))

    def test_require_accepts_both_convex_windings(self) -> None:
        import torch

        for rise, start_degrees, end_degrees in (
            (-1, -20.0, -8.0),
            (1, 20.0, 8.0),
        ):
            with self.subTest(rise=rise):
                mask = torch.zeros((12, 14), dtype=torch.bool)
                first_y = 5
                second_y = first_y + rise
                mask[first_y, 1:4] = True
                mask[second_y, 7:10] = True
                tangent_x = torch.ones(mask.shape, dtype=torch.float32)
                tangent_y = torch.zeros_like(tangent_x)
                for y, x0, x1, degrees in (
                    (first_y, 1, 4, start_degrees),
                    (second_y, 7, 10, end_degrees),
                ):
                    radians = torch.deg2rad(torch.tensor(degrees))
                    tangent_x[y, x0:x1] = torch.cos(radians)
                    tangent_y[y, x0:x1] = torch.sin(radians)

                labels = self._labels(
                    mask,
                    tangent_x,
                    tangent_y,
                    maximum_gap=4,
                    curvature_policy="require",
                )

                self.assertEqual(
                    int(labels[first_y, 2]), int(labels[second_y, 8])
                )

    def test_curvature_validation_rejects_unknown_policy_and_tolerance(self) -> None:
        import torch

        mask = torch.ones((3, 3), dtype=torch.bool)
        tangent_x = torch.ones(mask.shape, dtype=torch.float32)
        tangent_y = torch.zeros_like(tangent_x)
        with self.assertRaises(ValueError):
            self._labels(
                mask,
                tangent_x,
                tangent_y,
                maximum_gap=2,
                curvature_policy="sometimes",
            )
        with self.assertRaises(ValueError):
            self._labels(
                mask,
                tangent_x,
                tangent_y,
                maximum_gap=2,
                curvature_tolerance=46.0,
            )


if __name__ == "__main__":
    unittest.main()
