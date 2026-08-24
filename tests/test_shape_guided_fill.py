from __future__ import annotations

import unittest
from dataclasses import fields
from inspect import signature
from time import perf_counter
from unittest.mock import patch

import cv2
import numpy as np

from seedvision.annotation import (
    EllipseHypothesis,
    SHAPE_FILL_CONNECTIVITY,
    ShapeGuidedFillOptions,
    fit_rotated_edge_ellipse,
    shape_guided_fill_instance,
    shape_guided_fill_region,
    shape_outward_extension_pressure,
)
from seedvision.annotation.tools import SmartFillOptions, smart_fill_region


def _axial_angle_error(first: float, second: float) -> float:
    return abs((first - second + 90.0) % 180.0 - 90.0)


def _ellipse_fixture(
    shape: tuple[int, int],
    centre: tuple[int, int],
    axes: tuple[int, int],
    angle: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = shape
    image = np.full((height, width, 3), (218, 222, 226), np.uint8)
    seed_mask = np.zeros((height, width), np.uint8)
    cv2.ellipse(seed_mask, centre, axes, angle, 0, 360, 1, -1)
    image[seed_mask > 0] = (68, 102, 146)
    edge = np.zeros((height, width), np.uint8)
    cv2.ellipse(edge, centre, axes, angle, 0, 360, 255, 2)
    return image, edge, seed_mask.astype(bool)


class ShapeGuidedFillTests(unittest.TestCase):
    def test_shape_options_expose_only_shape_specific_growth_controls(self) -> None:
        options = ShapeGuidedFillOptions()
        field_names = {field.name for field in fields(options)}

        self.assertEqual(options.outward_penalty_half_life_fraction, 0.05)
        self.assertEqual(options.outward_hard_cutoff_fraction, 0.10)
        self.assertTrue(
            {
                "outward_penalty_half_life_fraction",
                "outward_hard_cutoff_fraction",
            }.issubset(field_names)
        )
        self.assertTrue(
            {
                "tangent_mode",
                "tangent_weight",
                "tunnel_strength",
                "maximum_distance_from_cursor_px",
                "falloff_half_life_px",
                "connectivity",
                "boundary_search_fraction",
                "shape_adherence",
            }.isdisjoint(field_names)
        )
        self.assertNotIn(
            "tangent_hue", signature(shape_guided_fill_region).parameters
        )
        self.assertNotIn(
            "tangent_hue", signature(shape_guided_fill_instance).parameters
        )

    def test_shape_fill_delegates_growth_to_smart_fill_with_shared_options(self) -> None:
        image, edge, _seed = _ellipse_fixture(
            (170, 200), (98, 84), (40, 25), 23.0
        )
        labels = np.zeros((170, 200), np.uint16)
        options = ShapeGuidedFillOptions(
            colour_tolerance_lab=23.0,
            edge_barrier_threshold=0.47,
            maximum_added_pixels=123_456,
        )

        with patch(
            "seedvision.annotation.shape_guided.smart_fill_region",
            wraps=smart_fill_region,
        ) as delegated:
            region = shape_guided_fill_region(
                labels, image, edge, (95, 85), 3, 80.0, options
            )

        self.assertTrue(region.accepted, region.reason)
        self.assertEqual(delegated.call_count, 1)
        for delegated_call in delegated.call_args_list:
            delegated_options = delegated_call.args[5]
            self.assertIsInstance(delegated_options, SmartFillOptions)
            self.assertEqual(delegated_options.colour_tolerance_lab, 23.0)
            self.assertEqual(delegated_options.click_colour_tolerance_lab, 360.0)
            self.assertEqual(delegated_options.edge_stop_threshold, 0.47)
            self.assertEqual(delegated_options.tunnel_strength, 0.0)
            self.assertGreater(delegated_options.maximum_distance_from_cursor_px, 51)
            self.assertEqual(delegated_options.falloff_half_life_px, 4096.0)
            self.assertEqual(delegated_options.maximum_added_pixels, 123_456)
            self.assertEqual(
                delegated_options.connectivity, SHAPE_FILL_CONNECTIVITY
            )
            self.assertIsNotNone(delegated_call.kwargs["allowed_mask"])
            self.assertIsNotNone(
                delegated_call.kwargs["extension_pressure_mask"]
            )

    def test_shape_pressure_starts_inside_and_is_cut_off_outward(self) -> None:
        pressure = shape_outward_extension_pressure(
            np.asarray((-40.0, 0.0, 5.0, 10.0, 10.01), np.float32),
            100.0,
        )
        np.testing.assert_allclose(
            pressure,
            np.asarray((1.0, 0.5, 0.25, 0.125, 0.0), np.float32),
            rtol=1.0e-5,
            atol=1.0e-5,
        )

    def test_tiny_shape_half_life_produces_a_genuinely_narrow_allowance(self) -> None:
        pressure = shape_outward_extension_pressure(
            np.asarray((-1.0, -0.5, 0.0, 0.5, 1.0), np.float32),
            100.0,
            ShapeGuidedFillOptions(
                penalty_start_inside_fraction=0.01,
                outward_penalty_half_life_fraction=0.005,
                outward_hard_cutoff_fraction=0.10,
            ),
        )
        np.testing.assert_allclose(
            pressure,
            np.asarray((1.0, 0.5, 0.25, 0.125, 0.0625), np.float32),
            rtol=1.0e-5,
            atol=1.0e-5,
        )

    def test_boundary_search_can_reach_a_real_edge_deep_inside_the_prior(self) -> None:
        height = width = 140
        centre = (70, 70)
        image, edge, _seed = _ellipse_fixture(
            (height, width), centre, (14, 14), 0.0
        )
        labels = np.zeros((height, width), np.uint16)
        angles = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
        prior_polygon = np.rint(
            np.column_stack(
                (
                    centre[0] + 40.0 * np.cos(angles),
                    centre[1] + 40.0 * np.sin(angles),
                )
            )
        ).astype(np.int32)
        fixed_prior = EllipseHypothesis(
            centre_xy=(float(centre[0]), float(centre[1])),
            axes_xy=(40.0, 40.0),
            angle_degrees=0.0,
            polygon=prior_polygon,
            score=1.0,
            edge_coverage=1.0,
        )

        with patch(
            "seedvision.annotation.shape_guided.fit_rotated_edge_ellipse",
            return_value=fixed_prior,
        ):
            region = shape_guided_fill_region(
                labels,
                image,
                edge,
                centre,
                2,
                80.0,
                ShapeGuidedFillOptions(shape="circle", lab_contrast_weight=0.0),
            )

        self.assertTrue(region.accepted, region.reason)
        assert region.boundary_polygon is not None
        radii = np.linalg.norm(
            region.boundary_polygon.astype(np.float32)
            - np.asarray(centre, np.float32),
            axis=1,
        )
        # The physical edge is 26 px inside the 40 px prior: far beyond the
        # 8 px (0.10D) outward allowance. A symmetric narrow search band would
        # miss it, while the one-sided design must retain the whole interior.
        self.assertAlmostEqual(float(np.median(radii)), 14.0, delta=2.5)

    def test_final_shape_mask_matches_canonical_smart_fill_at_supported_edge(self) -> None:
        image, edge, _seed = _ellipse_fixture(
            (170, 200), (98, 84), (40, 25), 23.0
        )
        labels = np.zeros((170, 200), np.uint16)
        shape_options = ShapeGuidedFillOptions()
        with patch(
            "seedvision.annotation.shape_guided.smart_fill_region",
            wraps=smart_fill_region,
        ) as delegated:
            guided = shape_guided_fill_region(
                labels, image, edge, (98, 84), 3, 80.0, shape_options
            )
        self.assertTrue(guided.accepted, guided.reason)
        canonical = smart_fill_region(
            *delegated.call_args.args, **delegated.call_args.kwargs
        )
        guided_full = np.zeros(labels.shape, dtype=bool)
        guided_full[
            guided.y : guided.y + guided.mask.shape[0],
            guided.x : guided.x + guided.mask.shape[1],
        ] = guided.mask
        canonical_full = np.zeros(labels.shape, dtype=bool)
        canonical_full[
            canonical.y : canonical.y + canonical.mask.shape[0],
            canonical.x : canonical.x + canonical.mask.shape[1],
        ] = canonical.mask

        self.assertTrue(np.array_equal(guided_full, canonical_full))

    def test_preferred_scale_selects_the_matching_concentric_edge(self) -> None:
        edge = np.zeros((170, 170), np.uint8)
        cv2.circle(edge, (85, 85), 30, 255, 2)
        cv2.circle(edge, (85, 85), 45, 255, 2)

        normal = fit_rotated_edge_ellipse(
            edge,
            (85, 85),
            60.0,
            ShapeGuidedFillOptions(shape="circle"),
        )
        larger = fit_rotated_edge_ellipse(
            edge,
            (85, 85),
            60.0,
            ShapeGuidedFillOptions(
                shape="circle", preferred_scale=1.50
            ),
        )

        self.assertIsNotNone(normal)
        self.assertIsNotNone(larger)
        assert normal is not None and larger is not None
        self.assertAlmostEqual(normal.axes_xy[0], 30.0, delta=3.0)
        self.assertAlmostEqual(larger.axes_xy[0], 45.0, delta=3.0)

    def test_auto_search_recovers_a_rotated_ellipse(self) -> None:
        image, edge, _seed = _ellipse_fixture(
            (170, 190), (98, 82), (39, 21), 37.0
        )
        hypothesis = fit_rotated_edge_ellipse(
            edge,
            (95, 84),
            78.0,
            ShapeGuidedFillOptions(),
        )

        self.assertIsNotNone(hypothesis)
        assert hypothesis is not None
        self.assertLess(_axial_angle_error(hypothesis.angle_degrees, 37.0), 6.0)
        self.assertAlmostEqual(hypothesis.centre_xy[0], 98.0, delta=3.0)
        self.assertAlmostEqual(hypothesis.centre_xy[1], 82.0, delta=3.0)
        self.assertGreater(hypothesis.edge_coverage, 0.45)
        self.assertEqual(image.shape[:2], edge.shape)

    def test_refined_boundary_follows_non_elliptical_real_edge(self) -> None:
        height, width = 190, 220
        centre = np.asarray((111.0, 92.0), np.float32)
        angles = np.linspace(0.0, 2.0 * np.pi, 160, endpoint=False)
        rotation = np.deg2rad(28.0)
        radius_modulation = 1.0 + 0.10 * np.sin(3.0 * angles + 0.4)
        local_x = 43.0 * radius_modulation * np.cos(angles)
        local_y = 25.0 * radius_modulation * np.sin(angles)
        polygon = np.rint(
            np.column_stack(
                (
                    centre[0] + local_x * np.cos(rotation) - local_y * np.sin(rotation),
                    centre[1] + local_x * np.sin(rotation) + local_y * np.cos(rotation),
                )
            )
        ).astype(np.int32)
        edge = np.zeros((height, width), np.uint8)
        cv2.polylines(edge, [polygon], True, 255, 2, lineType=cv2.LINE_8)
        seed = np.zeros((height, width), np.uint8)
        cv2.fillPoly(seed, [polygon], 1)
        image = np.full((height, width, 3), (224, 224, 224), np.uint8)
        image[seed > 0] = (62, 105, 151)
        labels = np.zeros((height, width), np.uint16)

        region = shape_guided_fill_region(
            labels,
            image,
            edge,
            tuple(centre),
            4,
            86.0,
            ShapeGuidedFillOptions(),
        )

        self.assertTrue(region.accepted, region.reason)
        self.assertIsNotNone(region.boundary_polygon)
        self.assertIsNotNone(region.prior_polygon)
        assert region.boundary_polygon is not None
        assert region.prior_polygon is not None
        distance = cv2.distanceTransform((edge == 0).astype(np.uint8), cv2.DIST_L2, 3)
        boundary = region.boundary_polygon
        boundary_distance = distance[boundary[:, 1], boundary[:, 0]]
        self.assertLess(float(np.median(boundary_distance)), 1.25)
        self.assertGreater(
            float(
                np.mean(
                    np.linalg.norm(
                        boundary.astype(np.float32)
                        - region.prior_polygon.astype(np.float32),
                        axis=1,
                    )
                )
            ),
            0.70,
        )
        self.assertGreater(region.edge_coverage, 0.55)

    def test_empty_or_spatially_unrelated_evidence_refuses_to_paint(self) -> None:
        image = np.full((150, 190, 3), 120, np.uint8)
        labels = np.zeros((150, 190), np.uint16)
        empty = np.zeros((150, 190), np.uint8)
        empty_result = shape_guided_fill_region(
            labels, image, empty, (95, 75), 2, 70.0
        )
        self.assertFalse(empty_result.accepted)
        self.assertEqual(empty_result.added_count, 0)
        self.assertFalse(np.any(empty_result.mask))

        unrelated = np.zeros_like(empty)
        unrelated[28:30, 25:165] = 255
        unrelated_result = shape_guided_fill_region(
            labels,
            image,
            unrelated,
            (95, 82),
            2,
            70.0,
            ShapeGuidedFillOptions(),
        )
        self.assertFalse(unrelated_result.accepted)
        self.assertEqual(unrelated_result.added_count, 0)
        self.assertFalse(np.any(unrelated_result.mask))

    def test_apply_preserves_pixels_owned_by_another_seed(self) -> None:
        image, edge, _seed = _ellipse_fixture(
            (170, 200), (98, 84), (40, 25), 23.0
        )
        labels = np.zeros((170, 200), np.uint16)
        labels[76:82, 115:121] = 9

        region = shape_guided_fill_region(
            labels,
            image,
            edge,
            (95, 85),
            3,
            80.0,
            ShapeGuidedFillOptions(),
        )
        filled, added = shape_guided_fill_instance(
            labels,
            image,
            edge,
            (95, 85),
            3,
            80.0,
            ShapeGuidedFillOptions(),
        )

        self.assertTrue(region.accepted, region.reason)
        self.assertGreater(added, 2_000)
        self.assertTrue(np.all(filled[76:82, 115:121] == 9))
        local_other = labels[
            region.y : region.y + region.mask.shape[0],
            region.x : region.x + region.mask.shape[1],
        ] == 9
        self.assertFalse(np.any(region.mask & local_other))

    def test_large_image_cost_depends_on_bounded_seed_region(self) -> None:
        image, edge, _seed = _ellipse_fixture(
            (1000, 1200), (604, 497), (118, 68), 61.0
        )
        labels = np.zeros((1000, 1200), np.uint16)
        started = perf_counter()
        region = shape_guided_fill_region(
            labels,
            image,
            edge,
            (600, 500),
            7,
            236.0,
            ShapeGuidedFillOptions(maximum_added_pixels=100_000),
        )
        elapsed = perf_counter() - started

        self.assertTrue(region.accepted, region.reason)
        self.assertLessEqual(max(region.mask.shape), 400)
        self.assertLess(elapsed, 1.25)


if __name__ == "__main__":
    unittest.main()
