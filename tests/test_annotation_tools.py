from __future__ import annotations

import unittest
from time import perf_counter

import cv2
import numpy as np

from seedvision.annotation import (
    EdgeTraceOptions,
    SmartFillOptions,
    smart_fill_instance,
    snap_edge_point,
    trace_edge_path,
)
from seedvision.annotation.tools import _smart_fill_extension_pressure


class AnnotationToolTests(unittest.TestCase):
    def test_edge_cursor_prefers_nearby_ridge_to_stronger_parallel_ridge(self) -> None:
        edge = np.zeros((40, 60), dtype=np.float32)
        edge[20, 5:55] = 0.72
        edge[24, 5:55] = 1.0

        self.assertTupleEqual(snap_edge_point((30, 20), edge, 12), (30, 20))

    def test_edge_trace_follows_a_curved_high_strength_ridge(self) -> None:
        edge = np.zeros((48, 64), dtype=np.uint8)
        curve = np.asarray(
            [(x, round(20 - 10 * np.sin((x - 5) / 50 * np.pi))) for x in range(5, 56)],
            dtype=np.int32,
        )
        cv2.polylines(edge, [curve], False, 255, 2)
        path = trace_edge_path(
            (5, 20),
            (55, 20),
            edge,
            EdgeTraceOptions(
                search_radius_px=16,
                edge_attraction=1.0,
                tangent_mode="off",
                tangent_weight=0.0,
                smoothing=1,
            ),
        )
        self.assertTupleEqual(tuple(path[0]), (5, 20))
        self.assertTupleEqual(tuple(path[-1]), (55, 20))
        self.assertLess(int(path[:, 1].min()), 14)
        self.assertGreater(float(np.mean(edge[path[:, 1], path[:, 0]] > 0)), 0.75)

    def test_edge_trace_long_preview_stays_interactive(self) -> None:
        edge = np.random.default_rng(4).integers(
            0, 256, (1400, 2400), dtype=np.uint8
        )
        started = perf_counter()
        path = trace_edge_path(
            (100, 100),
            (2100, 1100),
            edge,
            EdgeTraceOptions(search_radius_px=80, tangent_mode="off"),
        )
        elapsed = perf_counter() - started

        self.assertGreater(len(path), 500)
        self.assertLess(elapsed, 0.35)

    def test_edge_trace_never_switches_to_stronger_parallel_ridge(self) -> None:
        edge = np.zeros((44, 64), dtype=np.float32)
        edge[20, 5:56] = 0.72
        edge[24, 5:56] = 1.0

        path = trace_edge_path(
            (5, 20),
            (55, 20),
            edge,
            EdgeTraceOptions(
                search_radius_px=12,
                edge_attraction=1.0,
                tangent_mode="off",
                smoothing=0,
                edge_source="magnitude",
            ),
        )

        self.assertGreater(float(np.mean(path[:, 1] == 20)), 0.95)
        self.assertFalse(np.any(path[:, 1] == 24))

    def test_retired_plain_shape_snap_has_no_core_api(self) -> None:
        import seedvision.annotation as annotation
        import seedvision.annotation.tools as tools

        self.assertFalse(hasattr(annotation, "ShapeSnapOptions"))
        self.assertFalse(hasattr(annotation, "snap_shape_polygon"))
        self.assertFalse(hasattr(tools, "ShapeSnapOptions"))
        self.assertFalse(hasattr(tools, "snap_shape_polygon"))

    def test_smart_fill_adapts_to_touching_colours_and_stops_at_edge(self) -> None:
        height, width = 50, 70
        image = np.zeros((height, width, 3), dtype=np.uint8)
        for x in range(width):
            base = 60 + x * 2
            pattern = 5 if (x // 4) % 2 else 0
            image[:, x] = (base, base + pattern, base)
        edge = np.zeros((height, width), dtype=np.uint8)
        edge[:, 49] = 255
        labels = np.zeros((height, width), dtype=np.uint16)
        labels[23:28, 12:17] = 4
        filled, added = smart_fill_instance(
            labels,
            image,
            edge,
            (14, 25),
            4,
            SmartFillOptions(
                colour_tolerance_lab=10.0,
                edge_stop_threshold=0.45,
                maximum_distance_from_cursor_px=38,
                maximum_added_pixels=10_000,
            ),
        )
        self.assertGreater(added, 900)
        self.assertTrue(np.any(filled[:, 49] == 4))
        self.assertFalse(np.any(filled[:, 50:] == 4))

    def test_smart_fill_reaches_selected_thinned_ridge_not_broad_shoulder(self) -> None:
        height, width = 51, 72
        image = np.full((height, width, 3), 110, dtype=np.uint8)
        labels = np.zeros((height, width), dtype=np.uint16)
        ridge = np.zeros((height, width), dtype=np.uint8)
        ridge[:, 50] = 255
        coordinates = np.arange(width, dtype=np.float32)
        shoulder = np.exp(-0.5 * ((coordinates - 50.0) / 5.0) ** 2)
        magnitude = np.repeat(shoulder[None, :], height, axis=0)
        options = dict(
            colour_tolerance_lab=8.0,
            edge_stop_threshold=0.58,
            maximum_distance_from_cursor_px=38,
            maximum_added_pixels=10_000,
        )

        broad, _ = smart_fill_instance(
            labels,
            image,
            magnitude,
            (20, 25),
            3,
            SmartFillOptions(edge_source="magnitude", **options),
        )
        precise, _ = smart_fill_instance(
            labels,
            image,
            ridge,
            (20, 25),
            3,
            SmartFillOptions(edge_source="ridges", **options),
        )

        broad_frontier = int(np.max(np.nonzero(broad[25] == 3)))
        precise_frontier = int(np.max(np.nonzero(precise[25] == 3)))
        self.assertLessEqual(broad_frontier, 45)
        self.assertEqual(precise_frontier, 50)
        self.assertFalse(np.any(precise[:, 51:] == 3))

    def test_smart_fill_tunnelling_crosses_only_a_short_weak_barrier(self) -> None:
        image = np.full((31, 55, 3), 100, dtype=np.uint8)
        edge = np.zeros((31, 55), dtype=np.float32)
        edge[:, 26] = 0.30
        labels = np.zeros((31, 55), dtype=np.uint16)
        labels[13:18, 18:23] = 2
        base_options = dict(
            colour_tolerance_lab=8.0,
            edge_stop_threshold=0.20,
            maximum_distance_from_cursor_px=25,
            maximum_added_pixels=5_000,
        )
        stopped, _ = smart_fill_instance(
            labels,
            image,
            edge,
            (20, 15),
            2,
            SmartFillOptions(tunnel_strength=0.0, **base_options),
        )
        tunnelled, _ = smart_fill_instance(
            labels,
            image,
            edge,
            (20, 15),
            2,
            SmartFillOptions(tunnel_strength=0.5, **base_options),
        )
        self.assertFalse(np.any(stopped[:, 27:] == 2))
        self.assertTrue(np.any(tunnelled[:, 27:] == 2))

    def test_smart_fill_never_overwrites_another_seed_id(self) -> None:
        image = np.full((35, 50, 3), 120, dtype=np.uint8)
        edge = np.zeros((35, 50), dtype=np.uint8)
        labels = np.zeros((35, 50), dtype=np.uint16)
        labels[15:19, 8:12] = 1
        labels[10:25, 28:34] = 9
        filled, _ = smart_fill_instance(
            labels,
            image,
            edge,
            (10, 17),
            1,
            SmartFillOptions(
                maximum_distance_from_cursor_px=30,
                maximum_added_pixels=5_000,
            ),
        )
        self.assertTrue(np.all(filled[10:25, 28:34] == 9))

    def test_smart_fill_enforces_maximum_distance_from_cursor(self) -> None:
        image = np.full((81, 81, 3), 120, dtype=np.uint8)
        edge = np.zeros((81, 81), dtype=np.uint8)
        labels = np.zeros((81, 81), dtype=np.uint16)
        cursor = (40, 40)
        maximum_distance = 12

        filled, _added = smart_fill_instance(
            labels,
            image,
            edge,
            cursor,
            6,
            SmartFillOptions(
                maximum_distance_from_cursor_px=maximum_distance,
                maximum_added_pixels=5_000,
            ),
        )

        yy, xx = np.nonzero(filled == 6)
        distances = np.hypot(xx - cursor[0], yy - cursor[1])
        self.assertGreater(len(distances), 300)
        self.assertLessEqual(float(np.max(distances)), maximum_distance)
        self.assertEqual(int(filled[40, 52]), 6)
        self.assertEqual(int(filled[40, 53]), 0)

    def test_smart_fill_falloff_half_life_limits_repeated_colour_steps(self) -> None:
        self.assertAlmostEqual(
            float(_smart_fill_extension_pressure(10.0, 10.0)),
            0.5,
            places=7,
        )
        height = width = 61
        cursor = (20, 30)
        yy, xx = np.indices((height, width))
        manhattan_distance = np.abs(xx - cursor[0]) + np.abs(yy - cursor[1])
        corrected_lab = np.zeros((height, width, 3), dtype=np.uint8)
        corrected_lab[:, :, 0] = 100
        corrected_lab[:, :, 1:] = 128
        corrected_lab[:, :, 0][manhattan_distance >= 2] += 8
        corrected_lab[:, :, 0][manhattan_distance >= 10] += 8
        image = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)
        edge = np.zeros((height, width), dtype=np.float32)
        labels = np.zeros((height, width), dtype=np.uint16)
        common = dict(
            colour_tolerance_lab=10.0,
            edge_stop_threshold=0.58,
            maximum_distance_from_cursor_px=20,
            maximum_added_pixels=5_000,
            connectivity=4,
        )

        decayed, _ = smart_fill_instance(
            labels,
            image,
            edge,
            cursor,
            4,
            SmartFillOptions(falloff_half_life_px=10.0, **common),
            corrected_lab=corrected_lab,
        )
        nearly_constant, _ = smart_fill_instance(
            labels,
            image,
            edge,
            cursor,
            4,
            SmartFillOptions(falloff_half_life_px=4096.0, **common),
            corrected_lab=corrected_lab,
        )

        # The eight-unit step passes close to the cursor. At one half-life its
        # normalized size (0.8) exceeds the exact pressure (0.5), so the colour
        # transition itself and everything beyond it remain excluded. Unlike a
        # selected physical ridge, a colour-change barrier is not added as a
        # one-pixel boundary frontier.
        self.assertEqual(int(decayed[30, 22]), 4)
        self.assertEqual(int(decayed[30, 29]), 4)
        self.assertEqual(int(decayed[30, 30]), 0)
        self.assertEqual(int(decayed[30, 31]), 0)
        self.assertEqual(int(nearly_constant[30, 31]), 4)

    def test_smart_fill_recovers_one_seed_when_a_boundary_gap_leaks(self) -> None:
        image = np.full((130, 180, 3), 120, dtype=np.uint8)
        edge = np.zeros((130, 180), dtype=np.uint8)
        cv2.circle(edge, (70, 65), 20, 255, 1)
        cv2.circle(edge, (109, 65), 20, 255, 1)
        # Reproduce a small contact gap through which an ordinary flood would
        # merge both same-coloured seeds and reach its cursor-distance limit.
        edge[62:69, 89:92] = 0
        labels = np.zeros((130, 180), dtype=np.uint16)

        filled, added = smart_fill_instance(
            labels,
            image,
            edge,
            (70, 65),
            5,
            SmartFillOptions(
                edge_source="ridges",
                maximum_distance_from_cursor_px=48,
                maximum_added_pixels=10_000,
            ),
        )

        self.assertGreater(added, 1_000)
        self.assertLess(added, 1_600)
        self.assertEqual(int(filled[65, 70]), 5)
        self.assertEqual(int(filled[65, 109]), 0)

    def test_smart_fill_starts_an_empty_instance_and_is_preview_fast(self) -> None:
        image = np.full((900, 900, 3), 126, dtype=np.uint8)
        edge = np.zeros((900, 900), dtype=np.uint8)
        cv2.circle(edge, (450, 450), 120, 255, 2)
        labels = np.zeros((900, 900), dtype=np.uint16)
        started = perf_counter()
        filled, added = smart_fill_instance(
            labels,
            image,
            edge,
            (450, 450),
            7,
            SmartFillOptions(
                maximum_distance_from_cursor_px=150,
                maximum_added_pixels=100_000,
            ),
        )
        elapsed = perf_counter() - started

        self.assertGreater(added, 40_000)
        self.assertEqual(int(filled[450, 450]), 7)
        self.assertFalse(np.any(filled[330, :] == 7))
        self.assertLess(elapsed, 0.50)


if __name__ == "__main__":
    unittest.main()
