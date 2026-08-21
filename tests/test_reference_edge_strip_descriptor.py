from __future__ import annotations

import unittest


class ReferenceEdgeStripDescriptorTests(unittest.TestCase):
    """Scientific invariants for annotation-derived edge strip features."""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            import numpy  # noqa: F401
            import torch  # noqa: F401
        except ImportError as error:
            raise unittest.SkipTest(f"Strip descriptor dependencies unavailable: {error}")

    def test_adaptive_edge_shape_resolves_small_seeds_without_exceeding_cap(self) -> None:
        from seedvision.cuda.layers import _adaptive_edge_working_shape

        scale, height, width = _adaptive_edge_working_shape(
            200,
            400,
            50.0,
            material_maximum_dimension=128,
            edge_maximum_dimension=320,
            minimum_working_seed_diameter_px=32.0,
        )

        self.assertAlmostEqual(scale, 0.64)
        self.assertEqual((height, width), (128, 256))
        self.assertAlmostEqual(50.0 * scale, 32.0)

        capped_scale, capped_height, capped_width = _adaptive_edge_working_shape(
            200,
            400,
            20.0,
            material_maximum_dimension=128,
            edge_maximum_dimension=320,
            minimum_working_seed_diameter_px=32.0,
        )

        self.assertAlmostEqual(capped_scale, 0.80)
        self.assertEqual((capped_height, capped_width), (160, 320))
        self.assertLess(20.0 * capped_scale, 32.0)
        self.assertEqual(max(capped_height, capped_width), 320)

    def test_all_strip_geometry_settings_participate_in_cache_identity(self) -> None:
        from dataclasses import replace
        from unittest.mock import patch

        import numpy as np

        from seedvision.cuda.layers import reference_texture_probabilities
        from seedvision.visualization import AnalysisLayerSettings, build_analysis_layers

        image = np.full((48, 64, 3), (90, 125, 165), np.uint8)
        valid = np.full((48, 64), 255, np.uint8)
        cache: dict[str, object] = {}

        def calculate(settings: AnalysisLayerSettings) -> None:
            build_analysis_layers(
                image,
                valid,
                np.empty((0, 2), np.float32),
                np.empty((0,), np.float32),
                18.0,
                offset_x=0,
                offset_y=0,
                settings=settings,
                cache_values=cache,
                instance_masks_enabled=False,
                seed_edge_curves_enabled=False,
            )

        settings = AnalysisLayerSettings()
        with patch(
            "seedvision.cuda.layers.reference_texture_probabilities",
            wraps=reference_texture_probabilities,
        ) as fit:
            calculate(settings)
            calculate(settings)
            self.assertEqual(fit.call_count, 1)
            settings = replace(
                settings,
                reference_edge_strip_normal_offset_fraction=0.06,
            )
            calculate(settings)
            self.assertEqual(fit.call_count, 2)
            settings = replace(
                settings,
                reference_edge_strip_tangent_half_length_fraction=0.09,
            )
            calculate(settings)
            self.assertEqual(fit.call_count, 3)
            settings = replace(
                settings,
                reference_texture_edge_working_maximum_dimension=2176,
            )
            calculate(settings)
            self.assertEqual(fit.call_count, 4)
            settings = replace(
                settings,
                reference_edge_minimum_working_seed_diameter_px=30.0,
            )
            calculate(settings)
            self.assertEqual(fit.call_count, 5)

    def test_instance_normals_preserve_both_sides_of_a_seed_contact(self) -> None:
        import numpy as np

        from seedvision.cuda.layers import _instance_outward_normals

        labels = np.zeros((18, 24), np.uint16)
        labels[3:15, 2:12] = 1
        labels[3:15, 12:22] = 2

        contour, normal_x, normal_y = _instance_outward_normals(labels)

        self.assertTrue(np.all(contour[4:14, 11:13]))
        np.testing.assert_allclose(normal_x[4:14, 11], 1.0, atol=1e-6)
        np.testing.assert_allclose(normal_x[4:14, 12], -1.0, atol=1e-6)
        np.testing.assert_allclose(normal_y[4:14, 11:13], 0.0, atol=1e-6)
        self.assertFalse(contour[8, 7])
        self.assertEqual(float(normal_x[8, 7]), 0.0)
        self.assertEqual(float(normal_y[8, 7]), 0.0)

    def test_physical_training_strip_uses_annotation_axis_not_wrong_image_tangent(self) -> None:
        from unittest.mock import patch

        import numpy as np

        from seedvision.cuda.layers import (
            _edge_strip_feature_maps,
            _instance_outward_normals,
            directional_edges,
        )
        from seedvision.visualization import AnalysisLayerSettings, build_analysis_layers

        height = width = 64
        # A pure left-to-right ramp supplies a vertical image-gradient tangent
        # everywhere. It is deliberately wrong along the horizontal sides of
        # the reviewed square instance.
        ramp = np.linspace(25, 225, width, dtype=np.uint8)
        image = np.repeat(ramp[None, :, None], height, axis=0)
        image = np.repeat(image, 3, axis=2)
        labels = np.zeros((height, width), np.uint16)
        labels[16:48, 16:48] = 1
        observed: list[tuple[np.ndarray, np.ndarray]] = []

        def capture(*args, **kwargs):
            observed.append(
                (
                    args[3].detach().cpu().numpy()[0, 0].copy(),
                    args[4].detach().cpu().numpy()[0, 0].copy(),
                )
            )
            return _edge_strip_feature_maps(*args, **kwargs)

        settings = AnalysisLayerSettings(
            reference_texture_material_prototypes_per_class=8,
            reference_texture_edge_prototypes_per_class=8,
            reference_texture_minimum_samples_per_prototype=4,
            reference_texture_fit_iterations=1,
            reference_texture_working_maximum_dimension=256,
            reference_texture_edge_working_maximum_dimension=512,
        )
        with patch(
            "seedvision.cuda.layers._edge_strip_feature_maps",
            side_effect=capture,
        ):
            build_analysis_layers(
                image,
                np.full((height, width), 255, np.uint8),
                np.empty((0, 2), np.float32),
                np.empty((0,), np.float32),
                32.0,
                offset_x=0,
                offset_y=0,
                settings=settings,
                seed_instance_annotations=labels,
                instance_masks_enabled=False,
                seed_edge_curves_enabled=False,
            )

        self.assertEqual(len(observed), 2)
        training_tangent_x, training_tangent_y = observed[0]
        query_tangent_x, query_tangent_y = observed[1]
        contour, outward_x, outward_y = _instance_outward_normals(labels)
        self.assertTrue(contour[16, 24])
        np.testing.assert_allclose(
            training_tangent_x[contour],
            outward_y[contour],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            training_tangent_y[contour],
            -outward_x[contour],
            atol=1e-6,
        )
        # The image's vertical tangent would be (0, +/-1) here. The captured
        # training tangent instead follows the reviewed top edge horizontally.
        self.assertAlmostEqual(abs(float(training_tangent_x[16, 24])), 1.0)
        self.assertAlmostEqual(float(training_tangent_y[16, 24]), 0.0)

        gradients = directional_edges(
            image,
            np.full((height, width), 255, np.uint8),
            settings,
        )
        expected_query_x = gradients.tangent_x.detach().cpu().numpy()[0, 0]
        expected_query_y = gradients.tangent_y.detach().cpu().numpy()[0, 0]
        np.testing.assert_allclose(query_tangent_x, expected_query_x, atol=1e-6)
        np.testing.assert_allclose(query_tangent_y, expected_query_y, atol=1e-6)
        self.assertAlmostEqual(float(query_tangent_x[16, 24]), 0.0, places=5)
        self.assertAlmostEqual(abs(float(query_tangent_y[16, 24])), 1.0, places=5)

    def test_strip_polarity_reversal_swaps_only_inside_and_outside(self) -> None:
        import torch

        from seedvision.cuda.layers import _edge_strip_feature_maps

        height, width = 25, 31
        x = torch.arange(width, dtype=torch.float32)[None, None, None, :]
        lightness = x.expand(1, 1, height, width) * 5.0 + 40.0
        lab = torch.cat(
            (
                lightness,
                torch.full_like(lightness, 120.0),
                torch.full_like(lightness, 140.0),
            ),
            dim=1,
        )
        edge = torch.full((1, 1, height, width), 0.45)
        ridge = torch.full((1, 1, height, width), 0.25)
        tangent_x = torch.zeros_like(edge)
        tangent_y = torch.ones_like(edge)
        valid = torch.ones_like(edge, dtype=torch.bool)

        forward, reverse, strip_valid, normal_x, normal_y = (
            _edge_strip_feature_maps(
                lab,
                edge,
                ridge,
                tangent_x,
                tangent_y,
                valid,
                20.0,
                normal_offset_fraction=0.10,
                tangent_half_length_fraction=0.08,
            )
        )

        self.assertEqual(tuple(forward.shape), (1, 26, height, width))
        self.assertEqual(tuple(reverse.shape), tuple(forward.shape))
        torch.testing.assert_close(reverse[:, :7], forward[:, 14:21])
        torch.testing.assert_close(reverse[:, 7:14], forward[:, 7:14])
        torch.testing.assert_close(reverse[:, 14:21], forward[:, :7])
        torch.testing.assert_close(reverse[:, 21:24], -forward[:, 21:24])
        torch.testing.assert_close(reverse[:, 24:], forward[:, 24:])
        self.assertTrue(bool(strip_valid[0, 0, 12, 15]))
        torch.testing.assert_close(normal_x, -torch.ones_like(normal_x))
        torch.testing.assert_close(normal_y, torch.zeros_like(normal_y))
        self.assertTrue(bool(torch.isfinite(forward).all()))
        self.assertTrue(bool(torch.isfinite(reverse).all()))

    def test_strip_descriptor_rotates_with_the_edge_frame(self) -> None:
        import torch

        from seedvision.cuda.layers import _edge_strip_feature_maps

        size = 31
        x = torch.arange(size, dtype=torch.float32)[None, None, None, :]
        lightness = x.expand(1, 1, size, size) * 4.0 + 50.0
        lab = torch.cat(
            (
                lightness,
                torch.full_like(lightness, 118.0),
                torch.full_like(lightness, 143.0),
            ),
            dim=1,
        )
        edge = torch.linspace(0.1, 0.9, size)[None, None, None, :].expand(
            1, 1, size, size
        )
        ridge = (edge > 0.5).float()
        tangent_x = torch.zeros_like(edge)
        tangent_y = torch.ones_like(edge)
        valid = torch.ones_like(edge, dtype=torch.bool)
        arguments = dict(
            seed_diameter=20.0,
            normal_offset_fraction=0.10,
            tangent_half_length_fraction=0.08,
        )

        forward, reverse, strip_valid, _nx, _ny = _edge_strip_feature_maps(
            lab,
            edge,
            ridge,
            tangent_x,
            tangent_y,
            valid,
            **arguments,
        )
        rotated_lab = torch.rot90(lab, 1, (-2, -1))
        rotated_edge = torch.rot90(edge, 1, (-2, -1))
        rotated_ridge = torch.rot90(ridge, 1, (-2, -1))
        rotated_valid = torch.rot90(valid, 1, (-2, -1))
        rotated_tangent_x = torch.ones_like(rotated_edge)
        rotated_tangent_y = torch.zeros_like(rotated_edge)
        (
            rotated_forward,
            rotated_reverse,
            rotated_strip_valid,
            _rotated_nx,
            _rotated_ny,
        ) = _edge_strip_feature_maps(
            rotated_lab,
            rotated_edge,
            rotated_ridge,
            rotated_tangent_x,
            rotated_tangent_y,
            rotated_valid,
            **arguments,
        )

        torch.testing.assert_close(
            rotated_forward,
            torch.rot90(forward, 1, (-2, -1)),
            atol=2e-5,
            rtol=2e-5,
        )
        torch.testing.assert_close(
            rotated_reverse,
            torch.rot90(reverse, 1, (-2, -1)),
            atol=2e-5,
            rtol=2e-5,
        )
        torch.testing.assert_close(
            rotated_strip_valid,
            torch.rot90(strip_valid, 1, (-2, -1)),
        )

    def test_strip_validity_normalizes_partial_support_and_rejects_empty_sides(self) -> None:
        import torch

        from seedvision.cuda.layers import _edge_strip_feature_maps

        height = width = 29
        lab = torch.zeros((1, 3, height, width), dtype=torch.float32)
        lab[:, 0] = 100.0
        lab[:, 1] = 120.0
        lab[:, 2] = 140.0
        edge = torch.full((1, 1, height, width), 0.4)
        ridge = torch.full((1, 1, height, width), 0.2)
        tangent_x = torch.zeros_like(edge)
        tangent_y = torch.ones_like(edge)
        valid = torch.ones_like(edge, dtype=torch.bool)
        valid[:, :, :, :12] = False

        forward, reverse, strip_valid, _normal_x, _normal_y = (
            _edge_strip_feature_maps(
                lab,
                edge,
                ridge,
                tangent_x,
                tangent_y,
                valid,
                20.0,
                normal_offset_fraction=0.10,
                tangent_half_length_fraction=0.10,
            )
        )

        # The central strip has complete support, while a strip straddling the
        # invalid half-plane lacks one of its explicit normal-side regions.
        self.assertTrue(bool(strip_valid[0, 0, 14, 20]))
        self.assertFalse(bool(strip_valid[0, 0, 14, 12]))
        self.assertLess(float(forward[0, -1, 14, 12]), 0.60)
        self.assertTrue(bool(torch.isfinite(forward).all()))
        self.assertTrue(bool(torch.isfinite(reverse).all()))

    def test_public_edge_classifier_uses_the_independent_high_resolution_scale(self) -> None:
        import cv2
        import numpy as np

        from seedvision.visualization import AnalysisLayerSettings, build_analysis_layers

        height, width = 160, 640
        image = np.full((height, width, 3), (218, 222, 226), np.uint8)
        labels = np.zeros((height, width), np.uint16)
        centre = (320, 80)
        cv2.circle(image, centre, 20, (55, 92, 145), -1)
        cv2.line(image, (320, 65), (320, 95), (135, 165, 195), 3)
        cv2.circle(labels, centre, 20, 1, -1)
        settings = AnalysisLayerSettings(
            reference_texture_material_prototypes_per_class=8,
            reference_texture_edge_prototypes_per_class=8,
            reference_texture_minimum_samples_per_prototype=4,
            reference_texture_fit_iterations=1,
            reference_texture_working_maximum_dimension=256,
            reference_texture_edge_working_maximum_dimension=512,
            reference_edge_minimum_working_seed_diameter_px=28.0,
        )

        layers = build_analysis_layers(
            image,
            np.full((height, width), 255, np.uint8),
            np.empty((0, 2), np.float32),
            np.empty((0,), np.float32),
            40.0,
            offset_x=0,
            offset_y=0,
            settings=settings,
            seed_instance_annotations=labels,
            instance_masks_enabled=False,
            seed_edge_curves_enabled=False,
        )

        profile = layers.reference_texture_profile
        self.assertIsNotNone(profile)
        self.assertAlmostEqual(profile.working_scale, 0.4)
        self.assertAlmostEqual(profile.edge_working_scale, 0.7)
        self.assertAlmostEqual(profile.edge_working_seed_diameter_px, 28.0)
        self.assertAlmostEqual(profile.edge_strip_normal_offset_px, 2.0)
        self.assertAlmostEqual(profile.edge_strip_tangent_half_length_px, 3.2)
        self.assertGreater(profile.edge_working_scale, profile.working_scale)
        self.assertEqual(len(profile.edge_feature_names), 26)
        self.assertTrue(
            any(name.startswith("interior strip") for name in profile.edge_feature_names)
        )
        self.assertTrue(
            any(name.startswith("centre/edge strip") for name in profile.edge_feature_names)
        )
        self.assertTrue(
            any(name.startswith("exterior strip") for name in profile.edge_feature_names)
        )
        self.assertTrue(
            any(name.startswith("signed cross-edge") for name in profile.edge_feature_names)
        )
        edge_prototypes = tuple(
            prototype
            for prototype in profile.prototypes
            if prototype.class_name in {"physical_edge", "non_edge"}
        )
        self.assertGreater(len(edge_prototypes), 0)
        self.assertTrue(
            all(
                0.0 <= prototype.centre_xy[0] < width
                and 0.0 <= prototype.centre_xy[1] < height
                for prototype in edge_prototypes
            )
        )
        physical = np.asarray(layers.physical_edge_probability)
        non_edge = np.asarray(layers.non_edge_probability)
        self.assertEqual(physical.shape, (height, width))
        self.assertEqual(non_edge.shape, (height, width))
        self.assertTrue(np.isfinite(physical).all())
        self.assertTrue(np.isfinite(non_edge).all())
        self.assertGreater(int(physical.max()), 0)

    def test_strip_classifier_prefers_a_matching_seed_rim_over_internal_pattern(self) -> None:
        import cv2
        import numpy as np

        from seedvision.visualization import AnalysisLayerSettings, build_analysis_layers

        height, width = 180, 360
        image = np.full((height, width, 3), (220, 224, 228), np.uint8)
        labels = np.zeros((height, width), np.uint16)
        yy, xx = np.indices((height, width))
        radius = 38
        centres = ((95, 90), (265, 90))
        dark_seed = np.asarray((42, 74, 118), np.uint8)
        light_seed = np.asarray((145, 165, 188), np.uint8)
        for centre_x, centre_y in centres:
            seed = (
                (xx - centre_x) ** 2 + (yy - centre_y) ** 2
            ) <= radius**2
            image[seed & (xx < centre_x)] = dark_seed
            image[seed & (xx >= centre_x)] = light_seed
        cv2.circle(labels, centres[0], radius, 1, -1)

        layers = build_analysis_layers(
            image,
            np.full((height, width), 255, np.uint8),
            np.empty((0, 2), np.float32),
            np.empty((0,), np.float32),
            float(radius * 2),
            offset_x=0,
            offset_y=0,
            settings=AnalysisLayerSettings(
                reference_texture_material_prototypes_per_class=8,
                reference_texture_edge_prototypes_per_class=24,
                reference_texture_minimum_samples_per_prototype=4,
                reference_texture_fit_iterations=2,
                reference_texture_working_maximum_dimension=512,
                reference_texture_edge_working_maximum_dimension=512,
            ),
            seed_instance_annotations=labels,
            instance_masks_enabled=False,
            seed_edge_curves_enabled=False,
        )

        physical = np.asarray(layers.physical_edge_probability, dtype=np.float32)
        internal = np.asarray(layers.non_edge_probability, dtype=np.float32)
        query_rim = np.zeros((height, width), np.uint8)
        cv2.circle(query_rim, centres[1], radius, 255, 2)
        query_rim = query_rim > 0
        query_pattern = np.zeros((height, width), bool)
        query_pattern[
            centres[1][1] - radius + 8 : centres[1][1] + radius - 7,
            centres[1][0] - 1 : centres[1][0] + 2,
        ] = True
        rim_margin = physical[query_rim] - internal[query_rim]
        pattern_margin = physical[query_pattern] - internal[query_pattern]

        # The query seed is not annotated or overwritten. Its geometrically
        # matching outer rim should nevertheless receive a more physical class
        # margin than its equally prominent two-tone coat transition.
        self.assertGreater(float(np.median(rim_margin)), 0.0)
        self.assertGreater(float(np.median(internal[query_pattern])), 1.0)
        self.assertLess(
            float(np.median(physical[query_pattern])),
            float(np.median(physical[query_rim])) * 0.30,
        )
        self.assertGreater(
            float(np.median(rim_margin)),
            float(np.median(pattern_margin)) + 20.0,
        )

    def test_material_only_references_skip_both_edge_strip_passes(self) -> None:
        from unittest.mock import patch

        import numpy as np

        from seedvision.visualization import AnalysisLayerSettings, build_analysis_layers

        height, width = 96, 320
        image = np.full((height, width, 3), (205, 215, 225), np.uint8)
        image[24:72, 110:210] = (55, 90, 145)
        foreground = np.zeros((height, width), bool)
        foreground[32:64, 130:190] = True

        with patch("seedvision.cuda.layers._edge_strip_feature_maps") as strips:
            layers = build_analysis_layers(
                image,
                np.full((height, width), 255, np.uint8),
                np.empty((0, 2), np.float32),
                np.empty((0,), np.float32),
                24.0,
                offset_x=0,
                offset_y=0,
                settings=AnalysisLayerSettings(
                    reference_texture_material_prototypes_per_class=8,
                    reference_texture_edge_prototypes_per_class=8,
                    reference_texture_minimum_samples_per_prototype=4,
                    reference_texture_fit_iterations=1,
                    reference_texture_working_maximum_dimension=256,
                    reference_texture_edge_working_maximum_dimension=512,
                ),
                foreground_reference_mask=foreground,
                instance_masks_enabled=False,
                seed_edge_curves_enabled=False,
            )

        strips.assert_not_called()
        profile = layers.reference_texture_profile
        self.assertGreater(profile.count_for("foreground"), 0)
        self.assertEqual(profile.count_for("physical_edge"), 0)
        self.assertEqual(profile.count_for("non_edge"), 0)
        self.assertEqual(
            profile.sample_count_unit_for("foreground"),
            "source reference pixels",
        )
        self.assertEqual(
            profile.sample_count_unit_for("physical_edge"),
            "edge-working-resolution strip samples",
        )
        self.assertGreater(
            int(np.asarray(layers.reference_seed_surface_probability).max()), 0
        )
        self.assertEqual(int(np.asarray(layers.physical_edge_probability).max()), 0)
        self.assertEqual(int(np.asarray(layers.non_edge_probability).max()), 0)

    def test_instance_boundaries_are_derived_once_at_edge_working_resolution(self) -> None:
        from unittest.mock import patch

        import cv2
        import numpy as np

        from seedvision.annotation.instance_references import (
            instance_boundary_references,
        )
        from seedvision.visualization import AnalysisLayerSettings, build_analysis_layers

        height, width = 160, 640
        image = np.full((height, width, 3), (218, 222, 226), np.uint8)
        labels = np.zeros((height, width), np.uint16)
        cv2.circle(image, (320, 80), 20, (55, 92, 145), -1)
        cv2.circle(labels, (320, 80), 20, 1, -1)
        with patch(
            "seedvision.annotation.instance_references.instance_boundary_references",
            wraps=instance_boundary_references,
        ) as derive:
            layers = build_analysis_layers(
                image,
                np.full((height, width), 255, np.uint8),
                np.empty((0, 2), np.float32),
                np.empty((0,), np.float32),
                40.0,
                offset_x=0,
                offset_y=0,
                settings=AnalysisLayerSettings(
                    reference_texture_material_prototypes_per_class=8,
                    reference_texture_edge_prototypes_per_class=8,
                    reference_texture_minimum_samples_per_prototype=4,
                    reference_texture_fit_iterations=1,
                    reference_texture_working_maximum_dimension=256,
                    reference_texture_edge_working_maximum_dimension=512,
                    reference_edge_minimum_working_seed_diameter_px=28.0,
                ),
                seed_instance_annotations=labels,
                instance_masks_enabled=False,
                seed_edge_curves_enabled=False,
            )

        self.assertEqual(derive.call_count, 1)
        self.assertEqual(derive.call_args.args[0].shape, (112, 448))
        profile = layers.reference_texture_profile
        self.assertEqual(
            dict(profile.class_sample_counts)["foreground"],
            0,
        )
        self.assertEqual(
            profile.sample_count_unit_for("physical_edge"),
            "edge-working-resolution strip samples",
        )

    def test_profile_reports_the_effective_six_pixel_edge_diameter_floor(self) -> None:
        import numpy as np

        from seedvision.visualization import AnalysisLayerSettings, build_analysis_layers

        height, width = 64, 1024
        image = np.full((height, width, 3), (205, 215, 225), np.uint8)
        layers = build_analysis_layers(
            image,
            np.full((height, width), 255, np.uint8),
            np.empty((0, 2), np.float32),
            np.empty((0,), np.float32),
            2.0,
            offset_x=0,
            offset_y=0,
            settings=AnalysisLayerSettings(
                reference_texture_working_maximum_dimension=256,
                reference_texture_edge_working_maximum_dimension=512,
            ),
            instance_masks_enabled=False,
            seed_edge_curves_enabled=False,
        )

        profile = layers.reference_texture_profile
        self.assertAlmostEqual(profile.edge_working_scale, 0.5)
        self.assertAlmostEqual(profile.edge_working_seed_diameter_px, 6.0)


if __name__ == "__main__":
    unittest.main()
