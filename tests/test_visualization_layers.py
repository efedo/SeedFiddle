from __future__ import annotations

import unittest


class AnalysisLayerTests(unittest.TestCase):
    def test_colour_distribution_uses_painted_mode_frequencies(self) -> None:
        import torch

        from seedvision.cuda import lab_colour_distribution

        lab = torch.tensor(
            [[[80.0, 128.0, 128.0], [165.0, 151.0, 96.0], [20.0, 210.0, 210.0]]],
            dtype=torch.float32,
        )
        common = torch.tensor([[80.0, 128.0, 128.0]]).repeat(256, 1)
        rare = torch.tensor([[165.0, 151.0, 96.0]]).repeat(64, 1)
        probability, _, _, weights, _, _ = lab_colour_distribution(
            lab,
            torch.cat((common, rare), dim=0),
            torch.ones((1, 3), dtype=torch.bool),
            maximum_components=2,
            refinement_iterations=0,
            frequency_weight_power=1.0,
        )

        self.assertAlmostEqual(float(weights.max()), 0.8, delta=0.02)
        self.assertAlmostEqual(float(weights.min()), 0.2, delta=0.02)
        self.assertGreater(float(probability[0, 0]), 0.98)
        self.assertGreater(float(probability[0, 1]), 0.20)
        self.assertLess(float(probability[0, 1]), 0.30)
        self.assertLess(float(probability[0, 2]), 0.01)

    @classmethod
    def setUpClass(cls) -> None:
        try:
            import cv2  # noqa: F401
            import numpy  # noqa: F401
        except ImportError as error:
            raise unittest.SkipTest(f"Layer dependencies unavailable: {error}")

    def test_background_match_is_displayed_darker(self) -> None:
        import cv2
        import numpy as np

        from seedvision.visualization import build_analysis_layers

        image = np.full((96, 96, 3), 225, dtype=np.uint8)
        cv2.circle(image, (48, 48), 15, (25, 70, 190), -1)
        valid = np.full((96, 96), 255, dtype=np.uint8)
        layers = build_analysis_layers(
            image,
            valid,
            np.asarray(((48, 48),), dtype=np.float32),
            np.asarray((15,), dtype=np.float32),
            30.0,
            offset_x=0,
            offset_y=0,
        )

        self.assertGreater(
            int(layers.background_likelihood[10, 10]),
            int(layers.background_likelihood[48, 48]),
        )
        displayed = layers.background_rgba()
        self.assertLess(int(displayed[10, 10, 0]), int(displayed[48, 48, 0]))

    def test_refined_background_uses_local_noise_frequency_profile(self) -> None:
        import cv2
        import numpy as np

        from seedvision.visualization import build_analysis_layers

        rng = np.random.default_rng(20260807)
        noise = rng.integers(-12, 13, size=(128, 128, 1), dtype=np.int16)
        image = np.clip(210 + noise, 0, 255).astype(np.uint8)
        image = np.repeat(image, 3, axis=2)
        cv2.circle(image, (64, 64), 22, (35, 75, 185), -1)
        valid = np.full((128, 128), 255, dtype=np.uint8)
        layers = build_analysis_layers(
            image,
            valid,
            np.asarray(((64, 64),), dtype=np.float32),
            np.asarray((22,), dtype=np.float32),
            44.0,
            offset_x=0,
            offset_y=0,
        )

        background_score = float(
            np.mean(layers.refined_background_likelihood[16:40, 16:40])
        )
        nonbackground_score = float(
            np.mean(layers.refined_background_likelihood[56:72, 56:72])
        )
        self.assertGreater(background_score, nonbackground_score + 40.0)
        profile = layers.noise_frequency_profile
        self.assertEqual(len(profile.band_scales_px), 3)
        self.assertGreater(profile.background_sample_count, 128)
        self.assertGreater(profile.nonbackground_sample_count, 128)
        self.assertGreater(profile.separation, 0.5)
        displayed = layers.refined_background_rgba()
        self.assertLess(
            float(np.mean(displayed[16:40, 16:40, 0])),
            float(np.mean(displayed[56:72, 56:72, 0])),
        )

    def test_refined_background_exposes_24_directed_rays_and_colour_range(self) -> None:
        import cv2
        import numpy as np

        from seedvision.visualization import build_analysis_layers

        image = np.full((96, 96, 3), (220, 225, 230), dtype=np.uint8)
        cv2.circle(image, (48, 48), 20, (45, 75, 155), -1)
        valid = np.full((96, 96), 255, np.uint8)
        layers = build_analysis_layers(
            image,
            valid,
            np.asarray(((48, 48),), np.float32),
            np.asarray((20,), np.float32),
            40.0,
            offset_x=0,
            offset_y=0,
        )
        self.assertEqual(len(layers.directional_background_likelihoods), 24)
        self.assertEqual(
            layers.directional_background_angles_degrees,
            tuple(float(angle) for angle in range(0, 360, 15)),
        )
        self.assertIsNotNone(layers.background_colour_profile)
        self.assertGreater(layers.background_colour_profile.sample_count, 31)
        self.assertEqual(layers.directional_background_rgba(0).shape, (96, 96, 4))

    def test_manual_reference_points_override_automatic_background_colour(self) -> None:
        import cv2
        import numpy as np

        from seedvision.visualization import build_analysis_layers

        image = np.full((96, 96, 3), 225, dtype=np.uint8)
        cv2.circle(image, (48, 48), 18, (25, 70, 190), -1)
        valid = np.full((96, 96), 255, dtype=np.uint8)
        layers = build_analysis_layers(
            image,
            valid,
            np.asarray(((48, 48),), dtype=np.float32),
            np.asarray((18,), dtype=np.float32),
            36.0,
            offset_x=0,
            offset_y=0,
            background_reference_points=((48.0, 48.0),),
        )

        self.assertEqual(layers.background_mode, "manual")
        self.assertEqual(layers.background_reference_count, 1)
        self.assertGreater(
            int(layers.background_likelihood[48, 48]),
            int(layers.background_likelihood[8, 8]) + 100,
        )

    def test_painted_background_fits_multimodal_distribution_and_refines(self) -> None:
        import cv2
        import numpy as np

        from seedvision.visualization import AnalysisLayerSettings, build_analysis_layers

        image = np.full((96, 96, 3), (218, 224, 230), np.uint8)
        image[:, 48:] = (185, 215, 225)
        cv2.circle(image, (48, 56), 18, (28, 64, 145), -1)
        valid = np.full((96, 96), 255, np.uint8)
        layers = build_analysis_layers(
            image,
            valid,
            np.asarray(((48, 56),), np.float32),
            np.asarray((18,), np.float32),
            40.0,
            offset_x=0,
            offset_y=0,
            background_reference_points=((18.0, 18.0), (78.0, 18.0)),
            settings=AnalysisLayerSettings(
                background_colour_components=4,
                background_refinement_iterations=2,
                background_refinement_min_probability=0.80,
            ),
        )

        profile = layers.background_colour_profile
        self.assertIsNotNone(profile)
        self.assertGreaterEqual(len(profile.component_centres_lab), 2)
        self.assertEqual(profile.refinement_iterations, 2)
        self.assertGreater(int(layers.background_likelihood[18, 18]), 220)
        self.assertGreater(int(layers.background_likelihood[18, 78]), 220)
        self.assertLess(int(layers.background_likelihood[56, 48]), 80)

    def test_disabled_background_analysis_keeps_other_layers_available(self) -> None:
        import cv2
        import numpy as np

        from seedvision.visualization import build_analysis_layers

        image = np.full((96, 96, 3), 225, dtype=np.uint8)
        cv2.circle(image, (48, 48), 16, (25, 70, 190), -1)
        valid = np.full((96, 96), 255, dtype=np.uint8)
        layers = build_analysis_layers(
            image,
            valid,
            np.asarray(((48, 48),), dtype=np.float32),
            np.asarray((16,), dtype=np.float32),
            32.0,
            offset_x=0,
            offset_y=0,
            background_colour_enabled=False,
        )

        self.assertEqual(layers.background_mode, "disabled")
        self.assertFalse(np.any(layers.background_likelihood))
        self.assertFalse(np.any(layers.refined_background_likelihood))
        self.assertEqual(layers.noise_frequency_profile.background_sample_count, 0)
        self.assertGreater(np.count_nonzero(layers.instance_labels == 1), 0)
        self.assertGreater(int(layers.edge_likelihood.max()), 0)

    def test_manual_background_and_foreground_regions_are_hard_constraints(self) -> None:
        import cv2
        import numpy as np

        from seedvision.visualization import build_analysis_layers

        image = np.full((96, 96, 3), 225, dtype=np.uint8)
        cv2.circle(image, (48, 48), 18, (25, 70, 190), -1)
        valid = np.full((96, 96), 255, dtype=np.uint8)
        background_mask = np.zeros((96, 96), dtype=bool)
        background_mask[10:15, 10:15] = True
        foreground_mask = np.zeros((96, 96), dtype=bool)
        foreground_mask[46:51, 46:51] = True
        layers = build_analysis_layers(
            image,
            valid,
            np.asarray(((48, 48),), dtype=np.float32),
            np.asarray((18,), dtype=np.float32),
            36.0,
            offset_x=0,
            offset_y=0,
            background_reference_mask=background_mask,
            foreground_reference_mask=foreground_mask,
        )

        self.assertEqual(layers.background_reference_count, 25)
        for x, y, expected in ((12, 12, 255), (10, 14, 255), (48, 48, 0), (50, 46, 0)):
            self.assertEqual(int(layers.background_likelihood[y, x]), expected)
            self.assertEqual(int(layers.refined_background_likelihood[y, x]), expected)

    def test_vertical_edge_encodes_a_directed_vertical_tangent(self) -> None:
        import numpy as np

        from seedvision.visualization.layers import _directional_edges

        dark_to_bright = np.zeros((96, 96, 3), dtype=np.uint8)
        dark_to_bright[:, 48:] = 255
        valid = np.full((96, 96), 255, dtype=np.uint8)
        likelihood, directed_hue, undirected_hue = _directional_edges(
            dark_to_bright, valid
        )
        strong = likelihood >= 230

        self.assertTrue(np.any(strong))
        self.assertEqual(int(likelihood.max()), 255)
        # A bright right side produces an upward (270-degree) tangent, encoded
        # as hue 135 on OpenCV's half-degree 0..179 hue scale.
        self.assertGreaterEqual(float(np.median(directed_hue[strong])), 132.0)
        self.assertLessEqual(float(np.median(directed_hue[strong])), 138.0)
        self.assertGreaterEqual(float(np.median(undirected_hue[strong])), 87.0)
        self.assertLessEqual(float(np.median(undirected_hue[strong])), 93.0)

        bright_to_dark = 255 - dark_to_bright
        reverse_likelihood, reverse_directed_hue, reverse_undirected_hue = (
            _directional_edges(bright_to_dark, valid)
        )
        reverse_strong = reverse_likelihood >= 230
        # Reversing light/dark polarity reverses the directed tangent by 180°,
        # which is 90 units on OpenCV's circular hue scale.
        circular_difference = np.abs(
            (
                (
                    directed_hue[strong].astype(int)
                    - reverse_directed_hue[reverse_strong].astype(int)
                    + 90
                )
                % 180
            )
            - 90
        )
        self.assertGreaterEqual(float(np.median(circular_difference)), 87.0)
        np.testing.assert_allclose(
            undirected_hue[strong],
            reverse_undirected_hue[reverse_strong],
            atol=1,
        )

    def test_directed_and_undirected_outputs_share_one_gradient_computation(self) -> None:
        from unittest.mock import patch

        import numpy as np

        import seedvision.cuda.layers as cuda_layers
        from seedvision.visualization import build_analysis_layers

        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[:, 32:] = (210, 210, 210)
        valid = np.full((64, 64), 255, dtype=np.uint8)
        with patch.object(
            cuda_layers,
            "directional_edges",
            wraps=cuda_layers.directional_edges,
        ) as gradients:
            build_analysis_layers(
                image,
                valid,
                np.empty((0, 2), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                24.0,
                offset_x=0,
                offset_y=0,
            )
        self.assertEqual(gradients.call_count, 1)

    def test_user_edge_gamma_setting_changes_edge_likelihood(self) -> None:
        import numpy as np

        from seedvision.visualization import AnalysisLayerSettings
        from seedvision.visualization.layers import _directional_edges

        image = np.zeros((96, 96, 3), dtype=np.uint8)
        image[:, 48:] = 255
        valid = np.full((96, 96), 255, dtype=np.uint8)
        brightened, _, _ = _directional_edges(
            image, valid, AnalysisLayerSettings(edge_strength_gamma=0.30)
        )
        suppressed, _, _ = _directional_edges(
            image, valid, AnalysisLayerSettings(edge_strength_gamma=1.50)
        )
        self.assertGreater(float(np.mean(brightened)), float(np.mean(suppressed)))

    def test_seed_edge_curve_uses_curvature_and_optional_lightness_boost(self) -> None:
        import numpy as np

        from seedvision.visualization.layers import (
            _directional_edges,
            _seed_edge_curve_likelihood,
        )

        height = width = 160
        yy, xx = np.indices((height, width))
        radial_distance = np.hypot(xx - 80, yy - 80)
        inside = radial_distance <= 34
        valid = np.full((height, width), 255, dtype=np.uint8)

        dark_rim = np.full((height, width, 3), 230, dtype=np.uint8)
        dark_rim_values = np.clip(
            205 - radial_distance / 34.0 * 115, 0, 255
        ).astype(np.uint8)
        dark_rim[inside] = np.repeat(
            dark_rim_values[:, :, None], 3, axis=2
        )[inside]
        edge, directed_hue, _ = _directional_edges(dark_rim, valid)
        curve, selected_radius = _seed_edge_curve_likelihood(
            dark_rim, valid, edge, directed_hue, 68.0
        )
        boundary = (radial_distance >= 31) & (radial_distance <= 36)

        self.assertGreater(float(np.mean(curve[boundary])), 80.0)
        self.assertGreater(int(curve.max()), 230)
        self.assertAlmostEqual(
            float(np.median(selected_radius[curve >= 128])), 34.0, delta=0.5
        )

        bright_rim = np.full((height, width, 3), 230, dtype=np.uint8)
        bright_rim_values = np.clip(
            70 + radial_distance / 34.0 * 130, 0, 255
        ).astype(np.uint8)
        bright_rim[inside] = np.repeat(
            bright_rim_values[:, :, None], 3, axis=2
        )[inside]
        reverse_edge, reverse_hue, _ = _directional_edges(bright_rim, valid)
        reverse_curve, _ = _seed_edge_curve_likelihood(
            bright_rim, valid, reverse_edge, reverse_hue, 68.0
        )
        # Reversed radial lightness no longer gates an otherwise valid arc.
        self.assertGreater(int(reverse_curve.max()), 200)

        from seedvision.visualization import AnalysisLayerSettings

        unboosted_curve, _ = _seed_edge_curve_likelihood(
            dark_rim,
            valid,
            edge,
            directed_hue,
            68.0,
            AnalysisLayerSettings(curve_lightness_boost=0.0),
        )
        self.assertGreaterEqual(
            float(np.mean(curve[boundary])),
            float(np.mean(unboosted_curve[boundary])),
        )

        straight = np.full((height, width, 3), 230, dtype=np.uint8)
        x_values = np.clip(200 - (80 - np.arange(width)) * 4, 80, 200)
        straight[:, :80] = x_values[:80][None, :, None]
        straight_edge, straight_hue, _ = _directional_edges(straight, valid)
        straight_curve, _ = _seed_edge_curve_likelihood(
            straight, valid, straight_edge, straight_hue, 68.0
        )
        self.assertLess(int(straight_curve.max()), 10)

    def test_instance_labels_and_rgb_values_are_unique(self) -> None:
        import cv2
        import numpy as np

        from seedvision.visualization import build_analysis_layers

        image = np.full((120, 120, 3), 230, dtype=np.uint8)
        centers = np.asarray(((30, 30), (76, 30), (30, 76), (76, 76)), np.float32)
        for index, center in enumerate(centers):
            cv2.circle(
                image,
                tuple(np.rint(center).astype(int)),
                14,
                (25 + index * 35, 70, 180 - index * 25),
                -1,
            )
        valid = np.full((120, 120), 255, dtype=np.uint8)
        layers = build_analysis_layers(
            image,
            valid,
            centers,
            np.full(4, 14, dtype=np.float32),
            30.0,
            offset_x=7,
            offset_y=9,
        )

        colours = {tuple(colour) for colour in layers.instance_colours[1:]}
        self.assertEqual(len(colours), 4)
        self.assertEqual(set(np.unique(layers.instance_labels)), {0, 1, 2, 3, 4})
        for label in range(1, 5):
            self.assertGreater(np.count_nonzero(layers.instance_labels == label), 0)

    def test_boundary_troubleshooting_products_stay_lazy_until_viewed(self) -> None:
        import cv2
        import numpy as np

        from seedvision.cuda import GpuRaster
        from seedvision.visualization import build_analysis_layers

        image = np.full((128, 128, 3), 228, dtype=np.uint8)
        cv2.ellipse(image, (64, 64), (27, 20), 18, 0, 360, (35, 78, 165), -1)
        valid = np.full((128, 128), 255, dtype=np.uint8)
        layers = build_analysis_layers(
            image,
            valid,
            np.asarray(((64, 64),), np.float32),
            np.asarray((24,), np.float32),
            48.0,
            offset_x=0,
            offset_y=0,
        )
        rasters = (
            layers.background_likelihood,
            layers.refined_background_likelihood,
            layers.instance_labels,
            layers.edge_likelihood,
            layers.directed_edge_hue,
            layers.edge_ridges,
            layers.edge_trace_labels,
            layers.edge_trace_continuity,
            layers.edge_trace_gap_confidence,
            layers.edge_radius_confidence,
            layers.edge_circle_confidence,
            layers.edge_ellipse_confidence,
            layers.edge_fit_residual,
            layers.edge_centre_votes,
            layers.edge_semantic_sides,
            layers.edge_rejection_strength,
            layers.seed_edge_curve_likelihood,
        )
        self.assertTrue(all(isinstance(raster, GpuRaster) for raster in rasters))
        self.assertTrue(all(not raster.is_materialized for raster in rasters))

        rgba = layers.edge_radius_confirmation_rgba()
        self.assertEqual(rgba.shape, (128, 128, 4))
        self.assertTrue(layers.edge_radius_confidence.is_materialized)
        self.assertTrue(layers.valid_mask.is_materialized)
        self.assertFalse(layers.edge_circle_confidence.is_materialized)
        self.assertFalse(layers.edge_centre_votes.is_materialized)

    def test_boundary_node_cache_reuses_gpu_ridges_and_traces(self) -> None:
        import cv2
        import numpy as np

        from seedvision.visualization import AnalysisLayerSettings, build_analysis_layers

        image = np.full((112, 112, 3), 225, dtype=np.uint8)
        cv2.circle(image, (56, 56), 23, (30, 75, 170), -1)
        valid = np.full((112, 112), 255, dtype=np.uint8)
        centers = np.asarray(((56, 56),), np.float32)
        radii = np.asarray((23,), np.float32)
        cache: dict[str, object] = {}

        build_analysis_layers(
            image, valid, centers, radii, 46.0,
            offset_x=0, offset_y=0, cache_values=cache,
        )
        first = cache["layer.seed_edge_curves"]
        first_gradients = cache["layer.edge_gradients"]

        build_analysis_layers(
            image,
            valid,
            centers,
            radii,
            46.0,
            offset_x=0,
            offset_y=0,
            cache_values=cache,
            dirty_nodes={"seed_edge_curves"},
            settings=AnalysisLayerSettings(boundary_semantic_weight=0.45),
        )
        final_only = cache["layer.seed_edge_curves"]
        self.assertIs(final_only.ridge_state, first.ridge_state)
        self.assertIs(final_only.trace_state, first.trace_state)
        self.assertIs(cache["layer.edge_gradients"], first_gradients)

        build_analysis_layers(
            image,
            valid,
            centers,
            radii,
            46.0,
            offset_x=0,
            offset_y=0,
            cache_values=cache,
            dirty_nodes={"edge_traces", "seed_edge_curves"},
            settings=AnalysisLayerSettings(trace_tangent_tolerance_degrees=30.0),
        )
        trace_changed = cache["layer.seed_edge_curves"]
        self.assertIs(trace_changed.ridge_state, final_only.ridge_state)
        self.assertIsNot(trace_changed.trace_state, final_only.trace_state)

        build_analysis_layers(
            image,
            valid,
            centers,
            radii,
            46.0,
            offset_x=0,
            offset_y=0,
            cache_values=cache,
            dirty_nodes={"edge_ridges", "edge_traces", "seed_edge_curves"},
            settings=AnalysisLayerSettings(ridge_high_threshold=0.30),
        )
        ridge_changed = cache["layer.seed_edge_curves"]
        self.assertIsNot(ridge_changed.ridge_state, trace_changed.ridge_state)
        self.assertIsNot(ridge_changed.trace_state, trace_changed.trace_state)


if __name__ == "__main__":
    unittest.main()
