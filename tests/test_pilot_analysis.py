from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PilotAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import cv2  # noqa: F401
            import numpy  # noqa: F401
        except ImportError as error:
            raise unittest.SkipTest(f"OpenCV baseline dependencies unavailable: {error}")

    def test_sparse_pilot_generates_sixteen_proposals(self) -> None:
        import numpy as np

        from seedvision.segmentation.baseline import BaselineSettings, analyze_path

        path = ROOT / "images" / "IMG_9670c.JPG"
        if not path.exists():
            self.skipTest("Pilot image is not present")
        progress_events = []
        result = analyze_path(
            path,
            progress_callback=lambda node_id, state: progress_events.append(
                (node_id, state)
            ),
        )
        self.assertEqual(result.count, 16)
        self.assertEqual(result.crowding, "low")
        self.assertTrue(result.approximate)
        self.assertGreater(result.dish.confidence, 0.5)
        self.assertEqual(result.dish.vessel_type, "petri_dish")
        self.assertTrue(result.dish.rim_pair_detected)
        self.assertLess(result.dish.inner_radius, result.dish.outer_radius)
        effective_analysis_radius = (
            np.sqrt(result.analysis_region_pixel_count / np.pi)
            / BaselineSettings().inner_radius_fraction
        )
        self.assertAlmostEqual(
            effective_analysis_radius,
            result.dish.outer_radius,
            delta=2.0,
        )
        band = result.perimeter_background_band
        self.assertIsNotNone(band)
        self.assertTrue(band.outside_vessel)
        self.assertGreater(band.inner_radius_px, result.dish.outer_radius)
        self.assertGreater(band.outer_radius_px, band.inner_radius_px)
        self.assertAlmostEqual(band.buffer_cm, 0.35)
        self.assertAlmostEqual(band.thickness_cm, 0.50)
        self.assertAlmostEqual(
            band.inner_radius_px - result.dish.outer_radius,
            result.pixels_per_mm * 3.5,
            delta=0.5,
        )
        self.assertAlmostEqual(
            band.outer_radius_px - band.inner_radius_px,
            result.pixels_per_mm * 5.0,
            delta=0.5,
        )
        self.assertGreater(band.sample_count, 16)
        self.assertIsNotNone(result.layers.surrounding_noise_likelihood)
        self.assertIsNotNone(result.layers.surrounding_noise_valid_mask)
        self.assertFalse(result.layers.surrounding_noise_likelihood.is_materialized)
        surrounding_valid = np.asarray(result.layers.surrounding_noise_valid_mask)
        surrounding_noise = np.asarray(result.layers.surrounding_noise_likelihood)
        self.assertEqual(surrounding_noise.shape, surrounding_valid.shape)
        self.assertEqual(int(np.count_nonzero(surrounding_valid)), band.sample_count)
        self.assertGreater(
            surrounding_valid.shape[0], result.layers.valid_mask.shape[0]
        )
        self.assertTrue(
            all(
                value > 0.0
                for value in result.layers.noise_frequency_profile.background_log_scale
            )
        )
        self.assertIsNotNone(result.calibration.colour_card)
        self.assertIsNotNone(result.calibration.ruler)
        self.assertIsNotNone(result.pixels_per_mm)
        self.assertGreater(result.calibration.colour_card.detected_swatch_count, 16)
        self.assertGreaterEqual(result.foreground_threshold, 8.0)
        self.assertGreater(result.foreground_pixel_count, 0)
        self.assertGreater(result.analysis_region_pixel_count, result.foreground_pixel_count)
        self.assertGreater(result.distance_candidate_count, 0)
        self.assertGreater(result.circle_candidate_count, 0)
        labels = result.layers.instance_labels
        colours = result.layers.instance_colours[1:]
        self.assertEqual(int(labels.max()), result.count)
        self.assertEqual(len({tuple(colour) for colour in colours}), result.count)
        for identifier in range(1, result.count + 1):
            self.assertTrue((labels == identifier).any())
        for node_id in (
            "raw_images",
            "colour_reference",
            "deskew_colour",
            "layout_detection",
            "perimeter_background_reference",
            "foreground_segmentation",
            "edge_gradients",
            "edge_ridges",
            "edge_traces",
            "seed_edge_curves",
            "seed_interior",
            "colour_probabilities",
        ):
            self.assertIn(node_id, result.node_timings_seconds)
            self.assertGreater(result.node_timings_seconds[node_id], 0.0)
            self.assertIn((node_id, "started"), progress_events)
            self.assertIn((node_id, "completed"), progress_events)

    def test_dense_lupin_uses_outer_background_prior_and_soft_foreground(self) -> None:
        import numpy as np

        from seedvision.segmentation import PipelineAnalysisCache, analyze_path

        path = ROOT / "images" / "IMG_0002c.JPG"
        if not path.exists():
            self.skipTest("Dense lupin regression image is not present")
        cache = PipelineAnalysisCache()
        automatic = analyze_path(path, node_cache=cache)

        self.assertIsNotNone(automatic.perimeter_background_lab)
        self.assertLess(automatic.background_prior_deviation, 30.0)
        self.assertGreater(len(np.unique(automatic.foreground_probability)), 32)
        self.assertGreater(int(automatic.foreground_probability.max()), 245)
        self.assertLess(
            automatic.foreground_pixel_count / automatic.analysis_region_pixel_count,
            0.75,
        )

        reference = (
            float(automatic.dish.center_x),
            float(automatic.dish.center_y),
        )
        manual = analyze_path(
            path,
            node_cache=cache,
            foreground_reference_points=(reference,),
            dirty_nodes={"foreground_segmentation"},
        )
        difference = np.abs(
            manual.foreground_probability.astype(np.int16)
            - automatic.foreground_probability.astype(np.int16)
        )
        local_x = reference[0] - manual.crop_offset[0]
        local_y = reference[1] - manual.crop_offset[1]
        yy, xx = np.ogrid[: difference.shape[0], : difference.shape[1]]
        outside_reference = (xx - local_x) ** 2 + (yy - local_y) ** 2 > 40**2
        self.assertEqual(manual.foreground_reference_count, 1)
        painted_probability = int(
            manual.foreground_probability[round(local_y), round(local_x)]
        )
        self.assertGreater(painted_probability, 240)
        self.assertLess(painted_probability, 255)
        self.assertGreater(
            int(np.count_nonzero((difference > 2) & outside_reference)),
            10_000,
        )

    def test_perimeter_background_band_buffer_and_thickness_are_independent(self) -> None:
        from seedvision.calibration.geometry import DishCircle
        from seedvision.segmentation.baseline import _background_band_radii

        dish = DishCircle(320, 240, 192, 1.0)
        inner, outer = _background_band_radii(
            dish,
            4.0,
            buffer_cm=0.35,
            thickness_cm=0.50,
        )
        self.assertAlmostEqual(inner, 206.0)
        self.assertAlmostEqual(outer, 226.0)

        moved_inner, moved_outer = _background_band_radii(
            dish,
            4.0,
            buffer_cm=0.60,
            thickness_cm=0.20,
        )
        self.assertAlmostEqual(moved_inner, 216.0)
        self.assertAlmostEqual(moved_outer - moved_inner, 8.0)

        fallback_inner, fallback_outer = _background_band_radii(
            dish,
            None,
            buffer_cm=0.35,
            thickness_cm=0.50,
        )
        self.assertAlmostEqual(fallback_inner, inner)
        self.assertAlmostEqual(fallback_outer, outer)

    def test_painted_background_samples_remain_multimodal_in_foreground_branch(self) -> None:
        import cv2
        import numpy as np

        from seedvision.cuda import CudaContext
        from seedvision.segmentation.baseline import BaselineSettings, _foreground_feature

        crop = np.full((96, 96, 3), (220, 220, 220), np.uint8)
        crop[:, 48:] = (180, 210, 230)
        cv2.circle(crop, (48, 55), 15, (30, 70, 150), -1)
        samples = np.concatenate(
            (
                np.full((160, 3), (220, 220, 220), np.uint8),
                np.full((160, 3), (180, 210, 230), np.uint8),
            ),
            axis=0,
        )
        feature, *_ = _foreground_feature(
            crop,
            BaselineSettings(
                foreground_refinement_iterations=1,
                foreground_reference_components=4,
            ),
            CudaContext.resolve(requested="cpu"),
            background_reference_samples=samples,
            seed_diameter=28.0,
        )

        self.assertLess(float(feature[12, 12]), 3.0)
        self.assertLess(float(feature[12, 82]), 3.0)
        self.assertGreater(float(feature[55, 48]), 30.0)

    def test_foreground_mask_colour_distribution_propagates_beyond_paint(self) -> None:
        import numpy as np

        from seedvision.cuda import CudaContext
        from seedvision.segmentation.baseline import BaselineSettings, _foreground_feature

        crop = np.full((128, 128, 3), (220, 220, 220), np.uint8)
        # Two spatially separate regions have the same subtle seed colour. Only
        # the left one is painted, so any change on the right comes from the
        # learned colour distribution rather than a hard mask override.
        crop[40:60, 20:40] = (211, 214, 218)
        crop[40:60, 88:108] = (211, 214, 218)
        reference_mask = np.zeros((128, 128), dtype=bool)
        reference_mask[44:56, 24:36] = True
        settings = BaselineSettings(
            foreground_reference_weight=0.90,
            foreground_refinement_iterations=0,
        )
        context = CudaContext.resolve(requested="cpu")
        unreferenced = _foreground_feature(
            crop, settings, context, seed_diameter=28.0
        )[1]
        referenced_result = _foreground_feature(
            crop,
            settings,
            context,
            foreground_reference_mask=reference_mask,
            seed_diameter=28.0,
        )
        referenced = referenced_result[1]
        profile = referenced_result[-1]

        # A painted pixel is not hard-forced to one; it receives the same
        # colour-derived evidence as an unpainted matching pixel.
        self.assertGreater(int(referenced[50, 30]), 240)
        self.assertLess(int(referenced[50, 30]), 255)
        self.assertEqual(int(referenced[50, 30]), int(referenced[50, 98]))
        self.assertGreater(
            int(referenced[50, 98]), int(unreferenced[50, 98]) + 120
        )
        self.assertEqual(int(referenced[10, 10]), int(unreferenced[10, 10]))
        self.assertIsNotNone(profile)
        self.assertGreaterEqual(len(profile.component_centres_lab), 1)
        self.assertAlmostEqual(sum(profile.component_weights), 1.0, places=4)

    def test_isolated_reference_seed_colours_drive_automatic_foreground(self) -> None:
        import numpy as np

        from seedvision.cuda import CudaContext
        from seedvision.segmentation.baseline import (
            BaselineSettings,
            _bgr_samples_to_lab,
            _foreground_feature,
        )

        crop = np.full((128, 128, 3), (180, 190, 200), np.uint8)
        crop[36:92, 18:52] = (55, 65, 75)
        context = CudaContext.resolve(requested="cpu")
        settings = BaselineSettings(
            foreground_reference_weight=0.90,
            foreground_refinement_iterations=0,
        )
        reference_bgr = np.full((512, 3), (180, 190, 200), np.uint8)
        reference_lab = _bgr_samples_to_lab(reference_bgr, context)
        background_lab = _bgr_samples_to_lab(
            np.full((512, 3), (220, 220, 220), np.uint8), context
        )

        unreferenced = _foreground_feature(
            crop,
            settings,
            context,
            perimeter_background_samples_lab=background_lab,
            seed_diameter=28.0,
        )[1]
        automatic_result = _foreground_feature(
            crop,
            settings,
            context,
            perimeter_background_samples_lab=background_lab,
            automatic_foreground_samples_lab=reference_lab,
            seed_diameter=28.0,
        )
        automatic = automatic_result[1]
        profile = automatic_result[-1]

        matching_loss = int(unreferenced[50, 96]) - int(automatic[50, 96])
        distractor_loss = int(unreferenced[60, 32]) - int(automatic[60, 32])
        self.assertLess(matching_loss, 30)
        self.assertGreater(distractor_loss, matching_loss + 25)
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.source, "isolated_reference_seeds")
        self.assertEqual(profile.source_sample_count, 512)
        self.assertGreaterEqual(len(profile.component_centres_lab), 1)

    def test_zero_reference_weight_never_forces_painted_foreground(self) -> None:
        import numpy as np

        from seedvision.cuda import CudaContext
        from seedvision.segmentation.baseline import BaselineSettings, _foreground_feature

        crop = np.full((96, 96, 3), (220, 220, 220), np.uint8)
        crop[34:62, 34:62] = (211, 214, 218)
        reference_mask = np.zeros((96, 96), dtype=bool)
        reference_mask[42:54, 42:54] = True
        settings = BaselineSettings(
            foreground_reference_weight=0.0,
            foreground_refinement_iterations=0,
        )
        context = CudaContext.resolve(requested="cpu")
        automatic = _foreground_feature(crop, settings, context, seed_diameter=24.0)[1]
        referenced = _foreground_feature(
            crop,
            settings,
            context,
            foreground_reference_mask=reference_mask,
            seed_diameter=24.0,
        )[1]

        self.assertEqual(int(referenced[48, 48]), int(automatic[48, 48]))
        self.assertLess(int(referenced[48, 48]), 255)

    def test_painted_pattern_colours_remain_individual_frequency_bins(self) -> None:
        import numpy as np

        from seedvision.cuda import CudaContext
        from seedvision.segmentation.baseline import BaselineSettings, _foreground_feature

        crop = np.full((128, 160, 3), (225, 225, 225), np.uint8)
        dark = (35, 48, 72)
        light = (175, 188, 204)
        crop[38:58, 20:40] = dark
        crop[58:78, 20:40] = light
        crop[38:58, 118:138] = dark
        crop[58:78, 118:138] = light
        reference_mask = np.zeros(crop.shape[:2], dtype=bool)
        reference_mask[40:56, 22:38] = True
        reference_mask[60:76, 22:38] = True
        probability, profile = (
            lambda result: (result[1], result[-1])
        )(
            _foreground_feature(
                crop,
                BaselineSettings(
                    foreground_reference_weight=0.9,
                    foreground_refinement_iterations=0,
                    foreground_reference_components=64,
                ),
                CudaContext.resolve(requested="cpu"),
                foreground_reference_mask=reference_mask,
                seed_diameter=26.0,
            )
        )

        self.assertGreater(int(probability[48, 128]), 220)
        self.assertGreater(int(probability[68, 128]), 220)
        self.assertGreaterEqual(len(profile.component_centres_lab), 2)

    def test_all_proposals_are_inside_analysis_region(self) -> None:
        from seedvision.segmentation.baseline import analyze_path

        path = ROOT / "images" / "IMG_9670c.JPG"
        if not path.exists():
            self.skipTest("Pilot image is not present")
        result = analyze_path(path)
        for proposal in result.proposals:
            distance = (
                (proposal.center_x - result.dish.center_x) ** 2
                + (proposal.center_y - result.dish.center_y) ** 2
            ) ** 0.5
            self.assertLessEqual(distance, result.dish.outer_radius * 0.91)

    def test_default_disabled_branches_do_not_execute_candidate_or_instance_math(self) -> None:
        from seedvision.pipeline import build_default_pipeline
        from seedvision.segmentation.baseline import analyze_path

        path = ROOT / "images" / "IMG_9670c.JPG"
        if not path.exists():
            self.skipTest("Pilot image is not present")
        graph = build_default_pipeline()
        enabled = frozenset(
            node.identifier for node in graph.nodes.values() if node.enabled
        )
        result = analyze_path(path, enabled_nodes=enabled)

        self.assertEqual(result.count, 0)
        self.assertEqual(result.distance_candidate_count, 0)
        self.assertEqual(result.circle_candidate_count, 0)
        self.assertEqual(int(result.layers.instance_labels.max()), 0)
        self.assertIn("foreground_noise_likelihood", result.node_timings_seconds)
        disabled = {
            node.identifier for node in graph.nodes.values() if not node.enabled
        }
        self.assertTrue(disabled.isdisjoint(result.node_timings_seconds))
        self.assertTrue(
            set(graph.unused_nodes).isdisjoint(result.node_timings_seconds)
        )

    def test_pipeline_setting_changes_proposal_sensitivity(self) -> None:
        from seedvision.segmentation import BaselineSettings, analyze_path

        path = ROOT / "images" / "IMG_9670c.JPG"
        if not path.exists():
            self.skipTest("Pilot image is not present")
        strict = analyze_path(
            path, BaselineSettings(circle_accumulator_threshold=40)
        )
        sensitive = analyze_path(
            path, BaselineSettings(circle_accumulator_threshold=10)
        )
        self.assertLessEqual(strict.count, sensitive.count)
        self.assertLess(
            strict.circle_candidate_count, sensitive.circle_candidate_count
        )

    def test_circle_bank_consumes_edge_noise_and_local_lighting_evidence(self) -> None:
        import cv2
        import numpy as np

        from seedvision.cuda import CudaContext, image_to_tensor
        from seedvision.segmentation.baseline import (
            BaselineSettings,
            _circle_candidates,
        )

        context = CudaContext.resolve(requested="cpu")
        size = 160
        crop = np.zeros((size, size, 3), np.uint8)
        edge = np.zeros((size, size), np.float32)
        noise = np.zeros((size, size), np.float32)
        cv2.circle(edge, (80, 80), 18, 1.0, 2)
        cv2.circle(noise, (80, 80), 18, 1.0, -1)
        source = image_to_tensor(crop, context)
        common = dict(
            circle_accumulator_threshold=10,
            circle_edge_threshold=10,
            circle_min_radius_fraction=0.40,
            circle_max_radius_fraction=0.50,
            circle_min_distance_fraction=0.50,
            circle_working_maximum_dimension=512,
        )

        weight_keys = (
            "circle_edge_magnitude_weight",
            "circle_sensor_noise_weight",
            "circle_flattened_grayscale_weight",
            "circle_shadow_weight",
            "circle_highlight_weight",
        )
        tensor_keys = (
            "edge_magnitude_tensor",
            "sensor_noise_tensor",
            "flattened_grayscale_tensor",
            "shadow_likelihood_tensor",
            "highlight_likelihood_tensor",
        )

        def candidates(weight_key: str, tensor_key: str):
            weights = {key: 0.0 for key in weight_keys}
            weights[weight_key] = 1.0
            tensors = {key: None for key in tensor_keys}
            tensors["edge_magnitude_tensor"] = image_to_tensor(
                edge if tensor_key == "edge_magnitude_tensor" else edge * 0.0,
                context,
            )
            tensors[tensor_key] = image_to_tensor(
                edge if tensor_key == "edge_magnitude_tensor" else noise,
                context,
            )
            return _circle_candidates(
                crop,
                40.0,
                BaselineSettings(
                    **common,
                    **weights,
                ),
                context,
                source_tensor=source,
                analysis_radius=75.0,
                **tensors,
            )

        results = (
            candidates(weight_key, tensor_key)
            for weight_key, tensor_key in zip(
                weight_keys, tensor_keys, strict=True
            )
        )
        for result in results:
            nearest = min(
                result,
                key=lambda item: (item.x - 80.0) ** 2 + (item.y - 80.0) ** 2,
            )
            self.assertAlmostEqual(nearest.x, 80.0, delta=1.0)
            self.assertAlmostEqual(nearest.y, 80.0, delta=1.0)
            self.assertAlmostEqual(nearest.radius, 18.0, delta=2.0)

    def test_node_cache_reuses_upstream_stages_for_edge_change(self) -> None:
        from seedvision.segmentation import (
            AdvancedAnalysisSettings,
            AnalysisLayerSettings,
            PipelineAnalysisCache,
            analyze_path,
        )

        path = ROOT / "images" / "IMG_9670c.JPG"
        if not path.exists():
            self.skipTest("Pilot image is not present")
        cache = PipelineAnalysisCache()
        first = analyze_path(path, node_cache=cache)
        calibration = first.calibration
        background = first.layers.background_likelihood
        shared_gradients = cache.values["layer.edge_gradients"]
        layout_timing = first.node_timings_seconds["layout_detection"]
        second = analyze_path(
            path,
            node_cache=cache,
            dirty_nodes={"directed_edges", "seed_edge_curves"},
        )
        self.assertIs(second.calibration, calibration)
        self.assertIs(second.layers.background_likelihood, background)
        self.assertIn("deskew_colour", cache.last_reused_nodes)
        self.assertIn("background_likelihood", cache.last_reused_nodes)
        self.assertIn("directed_edges", cache.last_computed_nodes)
        self.assertIn("seed_edge_curves", cache.last_computed_nodes)
        self.assertIn("edge_gradients", cache.last_reused_nodes)
        self.assertNotIn("undirected_edges", cache.last_computed_nodes)
        self.assertIs(cache.values["layer.edge_gradients"], shared_gradients)
        self.assertEqual(
            second.node_timings_seconds["layout_detection"], layout_timing
        )
        self.assertGreater(second.node_timings_seconds["directed_edges"], 0.0)
        colour_probabilities = second.advanced.colour_probabilities
        third = analyze_path(
            path,
            node_cache=cache,
            dirty_nodes={"wrinkling"},
            advanced_settings=AdvancedAnalysisSettings(
                wrinkle_scale_fraction=0.08
            ),
        )
        self.assertIs(third.calibration, calibration)
        self.assertIs(third.advanced.colour_probabilities, colour_probabilities)
        self.assertIn("wrinkling", cache.last_computed_nodes)
        self.assertNotIn("illumination_decomposition", cache.last_computed_nodes)
        analyze_path(
            path,
            node_cache=cache,
            dirty_nodes={"edge_gradients"},
        )
        self.assertIn("edge_gradients", cache.last_computed_nodes)
        self.assertIn("directed_edges", cache.last_computed_nodes)
        self.assertIn("undirected_edges", cache.last_computed_nodes)
        self.assertIn("seed_edge_curves", cache.last_computed_nodes)
        adjusted = analyze_path(
            path,
            node_cache=cache,
            layer_settings=AnalysisLayerSettings(
                perimeter_background_buffer_cm=0.60,
                perimeter_background_band_thickness_cm=0.20,
            ),
            dirty_nodes={"perimeter_background_reference"},
        )
        adjusted_band = adjusted.perimeter_background_band
        self.assertAlmostEqual(adjusted_band.buffer_cm, 0.60)
        self.assertAlmostEqual(adjusted_band.thickness_cm, 0.20)
        self.assertAlmostEqual(
            adjusted_band.inner_radius_px - adjusted.dish.outer_radius,
            adjusted.pixels_per_mm * 6.0,
            delta=0.5,
        )
        self.assertAlmostEqual(
            adjusted_band.outer_radius_px - adjusted_band.inner_radius_px,
            adjusted.pixels_per_mm * 2.0,
            delta=0.5,
        )
        self.assertIs(adjusted.foreground_probability, third.foreground_probability)
        self.assertIn(
            "perimeter_background_reference", cache.last_computed_nodes
        )
        self.assertIn("background_likelihood", cache.last_computed_nodes)
        self.assertNotIn("foreground_segmentation", cache.last_computed_nodes)


if __name__ == "__main__":
    unittest.main()
