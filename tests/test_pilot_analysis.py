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

    def test_sparse_pilot_runs_diagnostics_without_foreground_references(self) -> None:
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
        self.assertGreater(result.count, 0)
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
        # Every probability layer now shares the crop that reaches the outer
        # edge of the perimeter Background band; the annulus is no longer a
        # larger one-off texture-only raster.
        self.assertEqual(surrounding_valid.shape, result.layers.valid_mask.shape)
        self.assertEqual(
            result.foreground_colour_probability.shape,
            result.layers.valid_mask.shape,
        )
        expected_outer_area = np.pi * band.outer_radius_px**2
        self.assertAlmostEqual(
            int(np.count_nonzero(np.asarray(result.layers.valid_mask))),
            expected_outer_area,
            delta=expected_outer_area * 0.01,
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
        self.assertEqual(result.foreground_threshold, 0.0)
        self.assertEqual(result.foreground_pixel_count, 0)
        self.assertFalse(np.any(result.foreground_colour_probability))
        self.assertGreater(result.analysis_region_pixel_count, result.foreground_pixel_count)
        self.assertEqual(result.distance_candidate_count, 0)
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

    def test_annotated_seed_scale_uses_largest_complete_fraction(self) -> None:
        import numpy as np

        from seedvision.segmentation.baseline import (
            _annotated_seed_diameter_measurements,
        )

        labels = np.zeros((140, 430), dtype=np.uint16)
        widths = (20, 28, 36, 44, 52, 60, 68, 76)
        for identifier, width in enumerate(widths, start=1):
            row = (identifier - 1) // 4
            column = (identifier - 1) % 4
            y0 = 16 + row * 58
            x0 = 12 + column * 100
            labels[y0 : y0 + 4, x0 : x0 + width] = identifier
        # A wide but image-cutoff annotation must remain visible diagnostically
        # without entering the selected top fraction.
        labels[120:140, 0:90] = 99

        measurements = _annotated_seed_diameter_measurements(
            labels, top_fraction=0.25
        )

        selected = sorted(
            item.diameter_px for item in measurements if item.selected
        )
        self.assertEqual(len(selected), 2)
        self.assertAlmostEqual(selected[0], 67.0, delta=1.0)
        self.assertAlmostEqual(selected[1], 75.0, delta=1.0)
        cutoff = next(item for item in measurements if item.identifier == 99)
        self.assertFalse(cutoff.complete)
        self.assertFalse(cutoff.selected)

    def test_dense_lupin_requires_user_authored_foreground_reference(self) -> None:
        import numpy as np

        from seedvision.segmentation import PipelineAnalysisCache, analyze_path

        path = ROOT / "images" / "IMG_0002c.JPG"
        if not path.exists():
            self.skipTest("Dense lupin regression image is not present")
        cache = PipelineAnalysisCache()
        automatic = analyze_path(path, node_cache=cache)

        self.assertIsNotNone(automatic.perimeter_background_lab)
        self.assertLess(automatic.background_prior_deviation, 30.0)
        self.assertFalse(np.any(automatic.foreground_colour_probability))
        self.assertEqual(automatic.foreground_pixel_count, 0)

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

    def test_perimeter_background_band_rejects_colour_card_contamination(self) -> None:
        import cv2
        import numpy as np

        from seedvision.calibration.geometry import DishCircle
        from seedvision.cuda import CudaContext
        from seedvision.segmentation.baseline import _dish_perimeter_background_lab

        height = width = 160
        image = np.full((height, width, 3), (218, 222, 226), np.uint8)
        yy, xx = np.indices((height, width))
        radius = np.sqrt((xx - 80.0) ** 2 + (yy - 80.0) ** 2)
        ring = (radius >= 50.0) & (radius <= 65.0)
        colour_card = ring & (xx >= 80) & (yy <= 80)
        image[colour_card] = (25, 45, 220)

        median_lab, band, samples = _dish_perimeter_background_lab(
            image,
            DishCircle(80, 80, 40, 1.0),
            None,
            CudaContext.resolve(requested="cpu"),
            colour_tolerance=10.0,
            band_radii_px=(50.0, 65.0),
        )

        self.assertIsNotNone(band.accepted_sample_mask)
        accepted = np.asarray(band.accepted_sample_mask) > 0
        self.assertGreater(int(np.count_nonzero(accepted & ring & ~colour_card)), 100)
        self.assertEqual(int(np.count_nonzero(accepted & colour_card)), 0)
        self.assertEqual(band.sample_count, int(np.count_nonzero(accepted)))
        self.assertEqual(int(samples.shape[0]), band.sample_count)
        expected_lab = cv2.cvtColor(
            np.asarray((218, 222, 226), np.uint8).reshape(1, 1, 3),
            cv2.COLOR_BGR2LAB,
        )[0, 0]
        np.testing.assert_allclose(median_lab, expected_lab, atol=2.0)

    def test_background_samples_do_not_synthesize_foreground_evidence(self) -> None:
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

        self.assertFalse(np.any(feature))

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
        profile = referenced_result[-2]

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

    def test_annotated_foreground_source_is_a_per_id_safely_inset_interior(self) -> None:
        import cv2
        import numpy as np

        from seedvision.annotation.instance_references import (
            instance_boundary_references,
        )
        from seedvision.cuda import CudaContext
        from seedvision.segmentation.baseline import (
            _annotated_foreground_reference_source_tensor,
        )

        labels = np.zeros((64, 72), np.uint16)
        labels[12:52, 8:36] = 1
        labels[12:52, 36:64] = 2
        context = CudaContext.resolve(requested="cpu")

        actual = (
            _annotated_foreground_reference_source_tensor(
                labels,
                20.0,
                context,
                inset_fraction=0.15,
            )
            .cpu()
            .numpy()
        )
        contour = instance_boundary_references(
            labels,
            20.0,
            interior_buffer_fraction=0.15,
        ).physical_edge
        # The contour has distance zero. A three-pixel inset therefore removes
        # the contour plus two further pixels on every side, including both
        # labelled sides of the contact between IDs.
        expected = (labels > 0) & ~(
            cv2.dilate(
                contour.astype(np.uint8),
                np.ones((5, 5), np.uint8),
            )
            > 0
        )
        np.testing.assert_array_equal(actual, expected)
        self.assertTrue(actual[32, 20])
        self.assertTrue(actual[32, 52])
        self.assertFalse(actual[32, 35])
        self.assertFalse(actual[32, 36])

        tiny = np.zeros((28, 28), np.uint16)
        tiny[10:15, 10:15] = 7
        self.assertFalse(
            bool(
                _annotated_foreground_reference_source_tensor(
                    tiny,
                    30.0,
                    context,
                    inset_fraction=0.20,
                ).any()
            )
        )

    def test_annotated_instances_are_opt_in_foreground_colour_references(self) -> None:
        import cv2
        import numpy as np

        from seedvision.cuda import CudaContext
        from seedvision.segmentation.baseline import (
            BaselineSettings,
            _annotated_foreground_reference_source_tensor,
            _foreground_feature,
        )

        image = np.full((128, 128, 3), (220, 220, 220), np.uint8)
        labels = np.zeros(image.shape[:2], np.uint16)
        cv2.circle(labels, (38, 64), 20, 1, -1)
        seed_colour = (211, 214, 218)
        image[labels > 0] = seed_colour
        # This separate patch has exactly the annotated seed colour. A change
        # here proves that annotations train a colour model rather than merely
        # overwriting their own coordinates.
        image[45:84, 86:112] = seed_colour
        context = CudaContext.resolve(requested="cpu")
        seed_diameter = 40.0
        disabled = _foreground_feature(
            image,
            BaselineSettings(
                foreground_include_annotated_seed_instances=False,
                foreground_reference_weight=0.90,
                foreground_refinement_iterations=0,
            ),
            context,
            seed_instance_annotations=labels,
            seed_diameter=seed_diameter,
        )
        enabled = _foreground_feature(
            image,
            BaselineSettings(
                foreground_include_annotated_seed_instances=True,
                foreground_reference_weight=0.90,
                foreground_refinement_iterations=0,
            ),
            context,
            seed_instance_annotations=labels,
            seed_diameter=seed_diameter,
        )

        self.assertIsNone(disabled[-1])
        source = enabled[-1]
        self.assertIsNotNone(source)
        assert source is not None
        expected_source = _annotated_foreground_reference_source_tensor(
            labels, seed_diameter, context
        )
        np.testing.assert_array_equal(
            np.asarray(source) > 0,
            expected_source.cpu().numpy(),
        )
        profile = enabled[-2]
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.source, "annotated_instances")
        self.assertEqual(
            profile.source_sample_count,
            int(expected_source.sum().item()),
        )
        disabled_probability = np.asarray(disabled[1])
        enabled_probability = np.asarray(enabled[1])
        matching = np.s_[55:75, 90:108]
        self.assertGreater(
            float(enabled_probability[matching].mean()),
            float(disabled_probability[matching].mean()) + 150.0,
        )
        self.assertAlmostEqual(
            float(enabled_probability[matching].mean()),
            float(enabled_probability[labels > 0].mean()),
            delta=2.0,
        )

    def test_automatic_foreground_source_obeys_painted_class_precedence(self) -> None:
        import cv2
        import numpy as np

        from seedvision.cuda import CudaContext
        from seedvision.segmentation.baseline import (
            BaselineSettings,
            _annotated_foreground_reference_source_tensor,
            _foreground_feature,
        )

        image = np.full((96, 96, 3), (220, 220, 220), np.uint8)
        labels = np.zeros(image.shape[:2], np.uint16)
        cv2.circle(labels, (48, 48), 26, 3, -1)
        image[labels > 0] = (70, 110, 170)
        painted_foreground = np.zeros(labels.shape, bool)
        painted_background = np.zeros(labels.shape, bool)
        other = np.zeros(labels.shape, bool)
        painted_foreground[40:45, 40:45] = True
        painted_background[50:55, 40:45] = True
        other[40:45, 50:55] = True
        context = CudaContext.resolve(requested="cpu")
        seed_diameter = 52.0
        common = dict(
            foreground_reference_mask=painted_foreground,
            background_reference_mask=painted_background,
            foreground_exclusion_mask=other,
            seed_instance_annotations=labels,
            seed_diameter=seed_diameter,
            # Make the focused synthetic raster entirely valid, so the expected
            # source is governed only by material-reference precedence.
            valid_radius=10_000.0,
        )

        enabled = _foreground_feature(
            image,
            BaselineSettings(
                foreground_include_annotated_seed_instances=True,
                foreground_refinement_iterations=0,
            ),
            context,
            **common,
        )
        raw_source = _annotated_foreground_reference_source_tensor(
            labels, seed_diameter, context
        ).cpu().numpy()
        expected_source = raw_source.copy()
        expected_source &= ~painted_foreground
        expected_source &= ~painted_background
        expected_source &= ~other
        np.testing.assert_array_equal(
            np.asarray(enabled[-1]) > 0,
            expected_source,
        )
        profile = enabled[-2]
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.source, "painted_and_annotated_instances")
        self.assertEqual(
            profile.source_sample_count,
            int(expected_source.sum()) + int(painted_foreground.sum()),
        )

        disabled = _foreground_feature(
            image,
            BaselineSettings(
                foreground_include_annotated_seed_instances=False,
                foreground_refinement_iterations=0,
            ),
            context,
            **common,
        )
        self.assertIsNone(disabled[-1])
        disabled_profile = disabled[-2]
        self.assertIsNotNone(disabled_profile)
        assert disabled_profile is not None
        self.assertEqual(disabled_profile.source, "painted")
        self.assertEqual(
            disabled_profile.source_sample_count,
            int(painted_foreground.sum()),
        )

    def test_isolated_scale_reference_colours_do_not_become_foreground(self) -> None:
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
        profile = automatic_result[-2]

        np.testing.assert_array_equal(automatic, unreferenced)
        self.assertFalse(np.any(automatic))
        self.assertIsNone(profile)

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
            lambda result: (result[1], result[-2])
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

    def test_candidate_fusion_keeps_primary_marker_as_suppression_anchor(self) -> None:
        from seedvision.segmentation.baseline import (
            BaselineSettings,
            _Candidate,
            _fuse_candidates,
        )

        primary = [_Candidate(0.0, 0.0, 20.0, 0.48, "distance")]
        # Both circles describe the same primary marker. Fusing the positive
        # observation first moves the weighted centre far enough that a mutable-
        # centre suppression pass would incorrectly admit the negative one.
        secondary = [
            _Candidate(45.0, 0.0, 20.0, 0.62, "circle"),
            _Candidate(-45.0, 0.0, 20.0, 0.62, "circle"),
        ]

        fused = _fuse_candidates(
            primary,
            secondary,
            seed_diameter=100.0,
            settings=BaselineSettings(),
        )

        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0].source, "fused")

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
        import numpy as np

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
            dirty_nodes={"edge_gradients", "seed_edge_curves"},
        )
        self.assertIs(second.calibration, calibration)
        self.assertIs(second.layers.background_likelihood, background)
        self.assertIn("deskew_colour", cache.last_reused_nodes)
        self.assertIn("background_likelihood", cache.last_reused_nodes)
        self.assertIn("edge_gradients", cache.last_computed_nodes)
        self.assertIn("seed_edge_curves", cache.last_computed_nodes)
        self.assertIsNot(cache.values["layer.edge_gradients"], shared_gradients)
        self.assertEqual(
            second.node_timings_seconds["layout_detection"], layout_timing
        )
        self.assertGreater(second.node_timings_seconds["edge_gradients"], 0.0)
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
        self.assertIsNot(
            adjusted.foreground_colour_probability,
            third.foreground_colour_probability,
        )
        self.assertEqual(
            adjusted.foreground_colour_probability.shape,
            adjusted.layers.valid_mask.shape,
        )
        self.assertFalse(np.any(adjusted.foreground_colour_probability))
        self.assertIsNot(
            adjusted.foreground_probability,
            third.foreground_probability,
        )
        self.assertIn(
            "material_evidence_decision", cache.last_computed_nodes
        )
        self.assertIn(
            "perimeter_background_reference", cache.last_computed_nodes
        )
        self.assertIn("background_likelihood", cache.last_computed_nodes)
        self.assertIn("foreground_segmentation", cache.last_computed_nodes)


if __name__ == "__main__":
    unittest.main()
