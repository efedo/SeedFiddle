from __future__ import annotations

import unittest


class AnalysisLayerTests(unittest.TestCase):
    def test_contrastive_other_evidence_preserves_equal_positive_matches(self) -> None:
        import torch

        from seedvision.cuda import apply_contrastive_negative_evidence

        positive = torch.tensor((0.90, 0.20, 0.60), dtype=torch.float32)
        negative = torch.tensor((0.90, 0.90, 0.30), dtype=torch.float32)
        result = apply_contrastive_negative_evidence(
            positive, negative, strength=0.95
        )

        # A colour shared by a positive class and Other is ambiguous, not a
        # reason to erase otherwise strong foreground/background evidence.
        self.assertAlmostEqual(float(result[0]), 0.90, delta=1e-5)
        # Other still rejects colours where its learned fit is clearly better.
        self.assertLess(float(result[1]), 0.10)
        self.assertAlmostEqual(float(result[2]), 0.60, delta=1e-5)

    def test_adding_diverse_foreground_references_cannot_evict_existing_colour(self) -> None:
        import torch

        from seedvision.cuda import lab_colour_frequency_distribution

        height = width = 128
        lab = torch.full((height, width, 3), 128.0, dtype=torch.float32)
        lab[:, :, 0] = 220.0
        original_colour = torch.tensor((60.0, 155.0, 145.0))
        lab[0, 0] = original_colour
        original = torch.zeros((height, width), dtype=torch.bool)
        original[0, 0] = True
        expanded = original.clone()
        colour_index = 0
        for top in range(1, 121, 12):
            for left in range(1, 121, 12):
                if colour_index >= 80:
                    break
                colour = torch.tensor(
                    (
                        160.0 + (colour_index % 5) * 12.0,
                        70.0 + ((colour_index // 5) % 4) * 30.0,
                        70.0 + ((colour_index // 20) % 4) * 30.0,
                    )
                )
                lab[top : top + 8, left : left + 8] = colour
                expanded[top : top + 8, left : left + 8] = True
                colour_index += 1
            if colour_index >= 80:
                break

        eligible = torch.ones((height, width), dtype=torch.bool)
        initial, *_ = lab_colour_frequency_distribution(
            lab,
            lab[original],
            eligible,
            maximum_bins=64,
            refinement_iterations=0,
            frequency_weight_power=0.0,
        )
        enlarged, *_ = lab_colour_frequency_distribution(
            lab,
            lab[expanded],
            eligible,
            maximum_bins=64,
            refinement_iterations=0,
            frequency_weight_power=0.0,
        )

        self.assertGreater(float(initial[0, 0]), 0.85)
        self.assertGreater(float(enlarged[0, 0]), 0.85)
        self.assertGreaterEqual(float(enlarged[0, 0]), float(initial[0, 0]) - 0.05)

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

    def test_colour_distribution_evaluates_many_modes_in_bounded_chunks(self) -> None:
        from unittest.mock import patch

        import torch

        from seedvision.cuda import lab_colour_distribution

        mode_index = torch.arange(32, dtype=torch.float32)
        mode_colours = torch.stack(
            (
                20.0 + (mode_index % 8.0) * 25.0,
                30.0 + torch.floor(mode_index / 8.0) * 50.0,
                45.0 + ((mode_index * 3.0) % 8.0) * 22.0,
            ),
            dim=1,
        )
        samples = mode_colours.repeat_interleave(64, dim=0)
        lab = mode_colours.repeat_interleave(2, dim=0).reshape(8, 8, 3)
        eligible = torch.ones((8, 8), dtype=torch.bool)
        exp_shapes = []
        original_exp = torch.exp

        def recording_exp(value):
            exp_shapes.append(tuple(value.shape))
            return original_exp(value)

        with patch.object(torch, "exp", side_effect=recording_exp):
            probability, centres, scales, weights, *_ = lab_colour_distribution(
                lab,
                samples,
                eligible,
                maximum_components=32,
                fit_iterations=1,
                refinement_iterations=0,
                combine_modes="maximum",
            )

        self.assertEqual(tuple(probability.shape), (8, 8))
        self.assertEqual(int(centres.shape[0]), 32)
        self.assertEqual(len(exp_shapes), 8)
        self.assertTrue(all(shape[-1] <= 4 for shape in exp_shapes))
        adjusted = weights.clamp_min(1e-6).pow(0.35)
        adjusted /= adjusted.max().clamp_min(1e-6)
        distance = torch.sum(
            ((lab[:, :, None, :] - centres[None, None, :, :])
             / scales[None, None, :, :]).square()
            * torch.tensor((1.0, 1.25, 1.25))[None, None, None, :],
            dim=3,
        )
        expected = torch.max(
            torch.exp(-0.5 * distance) * adjusted[None, None, :],
            dim=2,
        ).values
        torch.testing.assert_close(probability, expected)

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

    def test_reference_edge_probabilities_use_sparse_classes_without_forcing(self) -> None:
        import cv2
        import numpy as np

        from seedvision.visualization import build_analysis_layers

        image = np.full((96, 96, 3), 220, dtype=np.uint8)
        cv2.circle(image, (48, 48), 25, (65, 105, 165), -1)
        cv2.line(image, (48, 28), (48, 68), (25, 55, 95), 3)
        valid = np.full((96, 96), 255, dtype=np.uint8)
        physical = np.zeros((96, 96), dtype=bool)
        physical[46:51, 22:28] = True
        non_edge = np.zeros((96, 96), dtype=bool)
        non_edge[43:54, 46:51] = True
        layers = build_analysis_layers(
            image,
            valid,
            np.empty((0, 2), np.float32),
            np.empty((0,), np.float32),
            50.0,
            offset_x=0,
            offset_y=0,
            physical_edge_reference_mask=physical,
            non_edge_reference_mask=non_edge,
            instance_masks_enabled=False,
            seed_edge_curves_enabled=False,
        )

        physical_probability = np.asarray(layers.physical_edge_probability)
        non_edge_probability = np.asarray(layers.non_edge_probability)
        self.assertFalse(layers.reference_edge_ridges.is_materialized)
        reference_ridges = np.asarray(layers.reference_edge_ridges)
        self.assertEqual(physical_probability.shape, image.shape[:2])
        self.assertEqual(non_edge_probability.shape, image.shape[:2])
        self.assertEqual(reference_ridges.shape, image.shape[:2])
        self.assertFalse(np.array_equal(physical_probability, non_edge_probability))
        self.assertGreater(int(physical_probability.max()), 0)
        self.assertGreater(int(non_edge_probability.max()), 0)
        self.assertGreater(int(reference_ridges.max()), 0)
        self.assertLess(
            int(np.count_nonzero(reference_ridges)),
            int(np.count_nonzero(physical_probability)),
        )
        # Painted coordinates are samples, not hard output assignments.
        self.assertLess(int(physical_probability[48, 24]), 255)
        self.assertLess(int(non_edge_probability[48, 48]), 255)
        self.assertEqual(
            layers.reference_edge_probability_rgba(True).shape,
            (96, 96, 4),
        )
        self.assertEqual(layers.reference_edge_ridges_rgba().shape, (96, 96, 4))

        disabled = build_analysis_layers(
            image,
            valid,
            np.empty((0, 2), np.float32),
            np.empty((0,), np.float32),
            50.0,
            offset_x=0,
            offset_y=0,
            physical_edge_reference_mask=physical,
            non_edge_reference_mask=non_edge,
            reference_edge_ridges_enabled=False,
            instance_masks_enabled=False,
            seed_edge_curves_enabled=False,
        )
        self.assertEqual(int(np.asarray(disabled.reference_edge_ridges).max()), 0)

    def test_probability_ridge_thinning_centres_and_hysteretically_connects(self) -> None:
        import torch

        from seedvision.cuda.layers import thin_probability_ridges

        height, width = 32, 41
        x = torch.arange(width, dtype=torch.float32)
        cross_section = torch.exp(-0.5 * ((x - 20.0) / 2.8).square())
        amplitude = torch.zeros(height, dtype=torch.float32)
        amplitude[3:9] = 0.80
        amplitude[9:20] = 0.18
        amplitude[24:29] = 0.18
        probability = amplitude[:, None] * cross_section[None, :]
        normal_x = torch.ones_like(probability)
        normal_y = torch.zeros_like(probability)
        valid = torch.ones_like(probability, dtype=torch.bool)
        valid[3, 20] = False

        ridge = thin_probability_ridges(
            probability,
            normal_x,
            normal_y,
            valid,
            normal_sampling_step_px=1.0,
            low_threshold=0.10,
            high_threshold=0.40,
            hysteresis_iterations=16,
        )[0, 0]

        self.assertEqual(float(ridge[3, 20]), 0.0)
        self.assertGreater(float(ridge[8, 20]), 0.7)
        self.assertGreater(float(ridge[19, 20]), 0.1)
        self.assertEqual(float(ridge[24:29].max()), 0.0)
        retained_x = torch.nonzero(ridge > 0, as_tuple=False)[:, 1]
        self.assertTrue(torch.all(retained_x == 20).item())
        self.assertLess(int(torch.count_nonzero(ridge)), int(torch.count_nonzero(probability)))

    def test_reference_texture_bank_preserves_many_material_and_edge_modes(self) -> None:
        import cv2
        import numpy as np

        from seedvision.visualization import AnalysisLayerSettings, build_analysis_layers

        height = width = 192
        yy, xx = np.indices((height, width))
        image = np.full((height, width, 3), (210, 220, 225), np.uint8)
        image[:, :, 0] = np.clip(
            image[:, :, 0].astype(np.int16)
            + (((xx // 7 + yy // 9) % 2) * 6),
            0,
            255,
        ).astype(np.uint8)
        seeds = (
            (55, 65, 32, (45, 80, 135)),
            (135, 62, 31, (85, 125, 175)),
            (95, 137, 34, (40, 62, 88)),
        )
        for centre_x, centre_y, radius, colour in seeds:
            cv2.circle(image, (centre_x, centre_y), radius, colour, -1)
            interior = (
                (xx - centre_x) ** 2 + (yy - centre_y) ** 2
            ) <= radius**2
            stripe = ((xx + 2 * yy) // 6) % 3 == 0
            image[interior & stripe] = np.clip(
                image[interior & stripe].astype(np.int16)
                + np.asarray((35, 45, 55), np.int16),
                0,
                255,
            ).astype(np.uint8)
            cv2.circle(image, (centre_x, centre_y), radius, (20, 30, 45), 2)
        image[8:50, 145:185] = (190, 80, 35)

        background = np.zeros((height, width), bool)
        background[4:22, 4:80] = True
        background[165:188, 110:188] = True
        foreground = np.zeros((height, width), bool)
        foreground[48:82, 40:70] = True
        foreground[48:78, 122:150] = True
        foreground[122:154, 80:110] = True
        other = np.zeros((height, width), bool)
        other[12:45, 150:180] = True
        physical = np.zeros((height, width), bool)
        physical[60:70, 20:30] = True
        physical[55:70, 163:172] = True
        non_edge = np.zeros((height, width), bool)
        non_edge[48:78, 54:60] = True

        settings = AnalysisLayerSettings(
            reference_texture_prototypes_per_class=16,
            reference_texture_minimum_samples_per_prototype=8,
            reference_texture_fit_iterations=3,
            reference_texture_working_maximum_dimension=512,
        )
        layers = build_analysis_layers(
            image,
            np.full((height, width), 255, np.uint8),
            np.empty((0, 2), np.float32),
            np.empty((0,), np.float32),
            60.0,
            offset_x=0,
            offset_y=0,
            settings=settings,
            background_reference_mask=background,
            foreground_reference_mask=foreground,
            background_exclusion_mask=other,
            foreground_exclusion_mask=other,
            physical_edge_reference_mask=physical,
            non_edge_reference_mask=non_edge,
            instance_masks_enabled=False,
            seed_edge_curves_enabled=False,
        )

        profile = layers.reference_texture_profile
        self.assertIsNotNone(profile)
        self.assertEqual(
            {name: profile.count_for(name) for name in (
                "background",
                "foreground",
                "other",
                "physical_edge",
                "non_edge",
            )},
            {
                "background": 16,
                "foreground": 16,
                "other": 16,
                "physical_edge": 16,
                "non_edge": 16,
            },
        )
        self.assertTrue(
            all(not prototype.patch_bgr.flags.writeable for prototype in profile.prototypes)
        )
        self.assertTrue(
            all(
                prototype.tangent_degrees is not None
                for prototype in profile.prototypes
                if prototype.class_name in {"physical_edge", "non_edge"}
            )
        )

        seed_surface = np.asarray(layers.reference_seed_surface_probability)
        background_probability = np.asarray(
            layers.reference_background_texture_probability
        )
        other_probability = np.asarray(layers.reference_other_texture_probability)
        self.assertGreater(float(seed_surface[foreground].mean()), 180.0)
        self.assertGreater(float(background_probability[background].mean()), 220.0)
        self.assertGreater(float(other_probability[other].mean()), 210.0)
        self.assertLess(float(seed_surface[background].mean()), 10.0)
        self.assertLess(float(seed_surface[other].mean()), 10.0)
        # Reference pixels are classifier examples, never hard-assigned output.
        self.assertLess(int(seed_surface[foreground].max()), 255)

    def test_automatic_background_range_comes_from_exterior_lab_samples(self) -> None:
        import cv2
        import numpy as np
        import torch

        from seedvision.cuda import CudaContext
        from seedvision.cuda.layers import background_colour_likelihood

        # The crop deliberately contains no exterior-table colour. The compact
        # prior samples represent the independently measured annulus.
        crop = np.full((64, 64, 3), (30, 60, 180), np.uint8)
        valid = np.full((64, 64), 255, np.uint8)
        exterior_bgr = np.tile(
            np.asarray((200, 214, 216), np.uint8), (128, 1)
        )
        exterior_lab = cv2.cvtColor(
            exterior_bgr[:, None, :], cv2.COLOR_BGR2LAB
        ).reshape(-1, 3).astype(np.float32)

        _likelihood, mode, _count, profile = background_colour_likelihood(
            crop,
            valid,
            background_prior_lab=tuple(np.median(exterior_lab, axis=0)),
            background_prior_samples_lab=torch.from_numpy(exterior_lab),
            cuda_context=CudaContext.resolve(requested="cpu"),
        )

        self.assertEqual(mode, "automatic")
        self.assertEqual(profile.bgr_low, (200, 214, 216))
        self.assertEqual(profile.bgr_high, (200, 214, 216))

    def test_surrounding_background_reuses_exact_fitted_colour_profile(self) -> None:
        import numpy as np

        from seedvision.cuda import CudaContext, GpuRaster
        from seedvision.cuda.layers import (
            background_colour_likelihood,
            background_colour_profile_likelihood,
        )
        from seedvision.cuda.ops import image_to_tensor
        from seedvision.visualization import AnalysisLayerSettings

        image = np.full((48, 48, 3), (210, 218, 222), np.uint8)
        image[:, 24:] = (75, 120, 175)
        valid = np.full((48, 48), 255, np.uint8)
        background = np.zeros((48, 48), np.uint8)
        background[4:20, 4:20] = 255
        background[4:10, 28:34] = 255
        excluded = np.zeros((48, 48), np.uint8)
        excluded[28:44, 28:44] = 255
        settings = AnalysisLayerSettings(background_refinement_iterations=0)
        context = CudaContext.resolve(requested="cpu")
        source = image_to_tensor(image, context)
        valid_tensor = image_to_tensor(valid, context) > 0

        canonical, _mode, _count, profile = background_colour_likelihood(
            image,
            valid,
            background_reference_mask=background,
            background_exclusion_mask=excluded,
            settings=settings,
            cuda_context=context,
            source_tensor=source,
            valid_tensor=valid_tensor,
        )
        surrounding, surrounding_valid = background_colour_profile_likelihood(
            source,
            valid_tensor,
            profile,
            settings,
            cuda_context=context,
        )

        self.assertIsInstance(surrounding, GpuRaster)
        self.assertIsInstance(surrounding_valid, GpuRaster)
        self.assertFalse(surrounding.is_materialized)
        self.assertFalse(surrounding_valid.is_materialized)
        np.testing.assert_allclose(
            np.asarray(surrounding).astype(np.int16),
            np.asarray(canonical).astype(np.int16),
            atol=1,
        )

    def test_surrounding_background_layer_is_lazy_and_cached(self) -> None:
        import numpy as np

        from seedvision.cuda import CudaContext, GpuRaster
        from seedvision.cuda.ops import image_to_tensor
        from seedvision.visualization import build_analysis_layers

        image = np.full((48, 48, 3), (210, 218, 222), np.uint8)
        valid = np.full((48, 48), 255, np.uint8)
        context = CudaContext.resolve(requested="cpu")
        surrounding_image = np.full((24, 24, 3), (210, 218, 222), np.uint8)
        surrounding_source = image_to_tensor(surrounding_image, context)
        surrounding_valid = image_to_tensor(
            np.full((24, 24), 255, np.uint8), context
        ) > 0
        cache: dict[str, object] = {}
        common = dict(
            offset_x=0,
            offset_y=0,
            surrounding_noise_source_tensor=surrounding_source,
            surrounding_noise_valid_tensor=surrounding_valid,
            instance_masks_enabled=False,
            seed_edge_curves_enabled=False,
            cuda_context=context,
            cache_values=cache,
        )

        first_layers = build_analysis_layers(
            image,
            valid,
            np.empty((0, 2), np.float32),
            np.empty((0,), np.float32),
            16.0,
            **common,
        )
        first = first_layers.surrounding_background_likelihood
        self.assertIsInstance(first, GpuRaster)
        self.assertFalse(first.is_materialized)

        second_layers = build_analysis_layers(
            image,
            valid,
            np.empty((0, 2), np.float32),
            np.empty((0,), np.float32),
            16.0,
            **common,
        )
        self.assertIs(second_layers.surrounding_background_likelihood, first)

        changed_layers = build_analysis_layers(
            image,
            valid,
            np.empty((0, 2), np.float32),
            np.empty((0,), np.float32),
            16.0,
            dirty_nodes={"background_likelihood"},
            **common,
        )
        self.assertIsNot(changed_layers.surrounding_background_likelihood, first)

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

    def test_foreground_noise_profile_classifies_foreground_texture(self) -> None:
        import numpy as np

        from seedvision.visualization import build_analysis_layers

        rng = np.random.default_rng(41)
        image = np.full((96, 96, 3), 218, dtype=np.uint8)
        foreground_noise = rng.integers(
            -22, 23, size=(44, 44, 1), dtype=np.int16
        )
        image[26:70, 26:70] = np.clip(
            105 + foreground_noise, 0, 255
        ).astype(np.uint8)
        foreground_probability = np.full((96, 96), 18, np.uint8)
        foreground_probability[26:70, 26:70] = 235
        valid = np.full((96, 96), 255, np.uint8)
        layers = build_analysis_layers(
            image,
            valid,
            np.empty((0, 2), np.float32),
            np.empty((0,), np.float32),
            32.0,
            offset_x=0,
            offset_y=0,
            foreground_probability=foreground_probability,
            instance_masks_enabled=False,
            seed_edge_curves_enabled=False,
        )

        likelihood = np.asarray(layers.foreground_noise_likelihood)
        self.assertGreater(
            float(np.mean(likelihood[34:62, 34:62])),
            float(np.mean(likelihood[4:20, 4:20])) + 80.0,
        )
        self.assertGreater(
            layers.foreground_noise_frequency_profile.separation, 0.5
        )
        self.assertEqual(layers.foreground_noise_rgba().shape, (96, 96, 4))

    def test_noise_profiles_train_directly_from_both_painted_classes(self) -> None:
        import numpy as np

        from seedvision.cuda import CudaContext
        from seedvision.cuda.layers import (
            noise_frequency_background_likelihood,
            noise_frequency_foreground_likelihood,
        )
        from seedvision.visualization import AnalysisLayerSettings

        rng = np.random.default_rng(7)
        image = np.full((96, 96, 3), 218, np.uint8)
        image[48:80, 48:80] = np.clip(
            120 + rng.integers(-35, 36, (32, 32, 1)), 0, 255
        ).astype(np.uint8)
        valid = np.full((96, 96), 255, np.uint8)
        # Deliberately uninformative colour evidence: only the painted masks can
        # identify which texture belongs to which class.
        colour = np.full((96, 96), 128, np.uint8)
        background = np.zeros((96, 96), bool)
        background[14:30, 14:30] = True
        foreground = np.zeros((96, 96), bool)
        foreground[56:72, 56:72] = True
        settings = AnalysisLayerSettings()
        context = CudaContext.resolve(requested="cpu")

        background_result = noise_frequency_background_likelihood(
            image,
            valid,
            colour,
            32.0,
            settings,
            background_reference_mask=background,
            foreground_reference_mask=foreground,
            cuda_context=context,
        )
        foreground_result = noise_frequency_foreground_likelihood(
            image,
            valid,
            colour,
            32.0,
            settings,
            background_reference_mask=background,
            foreground_reference_mask=foreground,
            cuda_context=context,
        )
        background_profile = background_result[1]
        foreground_profile = foreground_result[1]

        self.assertEqual(background_profile.background_sample_count, 256)
        self.assertEqual(background_profile.nonbackground_sample_count, 256)
        self.assertEqual(foreground_profile.background_sample_count, 256)
        self.assertEqual(foreground_profile.nonbackground_sample_count, 256)
        self.assertLess(
            sum(background_profile.background_log_rms),
            sum(foreground_profile.background_log_rms),
        )
        # References train the probability distribution; their coordinates are
        # not overwritten to artificial exact-zero or exact-one outputs.
        self.assertTrue(
            np.all(np.asarray(background_result[0])[background] < 255)
        )
        self.assertTrue(
            np.all(np.asarray(foreground_result[0])[foreground] < 255)
        )

    def test_refined_background_integrates_directions_without_ray_overlays(self) -> None:
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
        self.assertEqual(layers.directional_background_likelihoods, ())
        self.assertEqual(
            layers.directional_background_angles_degrees,
            tuple(float(angle) for angle in range(0, 360, 15)),
        )
        self.assertIsNotNone(layers.background_colour_profile)
        self.assertGreater(layers.background_colour_profile.sample_count, 31)
        self.assertEqual(layers.refined_background_rgba().shape, (96, 96, 4))

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

    def test_manual_regions_constrain_colour_and_train_noise_without_overrides(self) -> None:
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
        self.assertGreater(int(layers.background_likelihood[12, 12]), 220)
        self.assertGreater(int(layers.background_likelihood[14, 10]), 220)
        self.assertLess(int(layers.background_likelihood[48, 48]), 80)
        noise = np.asarray(layers.refined_background_likelihood)
        self.assertGreater(float(np.mean(noise[background_mask])), 0.0)
        self.assertLess(float(np.mean(noise[background_mask])), 255.0)
        self.assertGreater(
            float(np.mean(noise[background_mask])),
            float(np.mean(noise[foreground_mask])),
        )

        # A foreground mark on a colour identical to the background sample is
        # still evaluated by the fitted model; its coordinate is not forced to
        # zero merely because it was painted.
        uniform = np.full((48, 48, 3), 225, dtype=np.uint8)
        uniform_valid = np.full((48, 48), 255, dtype=np.uint8)
        positive = np.zeros((48, 48), dtype=bool)
        positive[5:10, 5:10] = True
        negative = np.zeros((48, 48), dtype=bool)
        negative[30:35, 30:35] = True
        uniform_layers = build_analysis_layers(
            uniform,
            uniform_valid,
            np.empty((0, 2), np.float32),
            np.empty((0,), np.float32),
            24.0,
            offset_x=0,
            offset_y=0,
            background_reference_mask=positive,
            foreground_reference_mask=negative,
            instance_masks_enabled=False,
            seed_edge_curves_enabled=False,
        )
        self.assertGreater(int(uniform_layers.background_likelihood[32, 32]), 220)

    def test_other_evidence_is_contrastive_without_pixel_overrides(self) -> None:
        import numpy as np

        from seedvision.cuda import CudaContext
        from seedvision.segmentation.baseline import BaselineSettings, _foreground_feature
        from seedvision.visualization import build_analysis_layers

        image = np.full((96, 96, 3), 220, np.uint8)
        image[24:72, 24:72] = (65, 85, 135)
        valid = np.full((96, 96), 255, np.uint8)
        foreground_probability = np.full((96, 96), 20, np.uint8)
        foreground_probability[24:72, 24:72] = 235
        background_exclusion = np.zeros((96, 96), dtype=bool)
        foreground_exclusion = np.zeros((96, 96), dtype=bool)
        background_exclusion[8:16, 8:16] = True
        foreground_exclusion[40:48, 40:48] = True
        baseline_layers = build_analysis_layers(
            image,
            valid,
            np.empty((0, 2), np.float32),
            np.empty((0,), np.float32),
            32.0,
            offset_x=0,
            offset_y=0,
            foreground_probability=foreground_probability,
            instance_masks_enabled=False,
            seed_edge_curves_enabled=False,
        )
        layers = build_analysis_layers(
            image,
            valid,
            np.empty((0, 2), np.float32),
            np.empty((0,), np.float32),
            32.0,
            offset_x=0,
            offset_y=0,
            foreground_probability=foreground_probability,
            background_exclusion_mask=background_exclusion,
            foreground_exclusion_mask=foreground_exclusion,
            instance_masks_enabled=False,
            seed_edge_curves_enabled=False,
        )
        background_colour = np.asarray(layers.background_likelihood)
        background_noise = np.asarray(layers.refined_background_likelihood)
        foreground_noise = np.asarray(layers.foreground_noise_likelihood)
        # The brush coordinates are not overwritten. A same-colour unpainted
        # patch receives the same response, and an Other sample whose colour is
        # also an excellent background match cannot veto the positive model.
        remote_background = np.s_[8:16, 70:78]
        self.assertGreater(float(np.mean(background_colour[background_exclusion])), 0.0)
        self.assertAlmostEqual(
            float(np.mean(background_colour[background_exclusion])),
            float(np.mean(background_colour[remote_background])),
            delta=2.0,
        )
        self.assertGreaterEqual(
            float(np.mean(background_colour[remote_background])),
            float(np.mean(np.asarray(baseline_layers.background_likelihood)[remote_background]))
            - 2.0,
        )
        self.assertGreater(float(np.mean(background_noise[background_exclusion])), 0.0)
        self.assertGreater(float(np.mean(foreground_noise[foreground_exclusion])), 0.0)
        self.assertTrue(
            layers.background_colour_profile.excluded_component_centres_lab
        )

        baseline_foreground = _foreground_feature(
            image,
            BaselineSettings(),
            CudaContext.resolve(requested="cpu"),
            seed_diameter=32.0,
        )
        foreground_result = _foreground_feature(
            image,
            BaselineSettings(),
            CudaContext.resolve(requested="cpu"),
            foreground_exclusion_mask=foreground_exclusion,
            seed_diameter=32.0,
        )
        foreground_colour = np.asarray(foreground_result[1])
        remote_foreground = np.s_[40:48, 56:64]
        self.assertGreater(float(np.mean(foreground_colour[foreground_exclusion])), 0.0)
        self.assertAlmostEqual(
            float(np.mean(foreground_colour[foreground_exclusion])),
            float(np.mean(foreground_colour[remote_foreground])),
            delta=2.0,
        )
        self.assertGreaterEqual(
            float(np.mean(foreground_colour[remote_foreground])),
            float(np.mean(np.asarray(baseline_foreground[1])[remote_foreground]))
            - 2.0,
        )

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

    def test_surface_gradients_separate_lightening_and_darkening_directions(self) -> None:
        import numpy as np

        from seedvision.cuda import CudaContext
        from seedvision.cuda.layers import surface_directional_darkness_gradients
        from seedvision.visualization import AnalysisLayerSettings

        height = width = 96
        ramp = np.tile(
            np.linspace(20, 235, width, dtype=np.uint8), (height, 1)
        )
        image = np.repeat(ramp[:, :, None], 3, axis=2)
        valid = np.full((height, width), 255, dtype=np.uint8)
        products = surface_directional_darkness_gradients(
            image,
            valid,
            30.0,
            AnalysisLayerSettings(
                surface_gradient_blur_sigma=0.5,
                surface_gradient_radius_fraction=0.30,
                surface_gradient_direction_step_degrees=15,
                surface_gradient_sample_count=8,
            ),
            cuda_context=CudaContext.resolve(requested="cpu"),
        )
        center = np.s_[20:76, 20:76]
        lightening_hue = np.asarray(products.lightening_hue_raster)[center]
        darkening_hue = np.asarray(products.darkening_hue_raster)[center]

        self.assertGreater(
            float(np.median(products.lightening_magnitude[center].cpu())), 0.5
        )
        self.assertGreater(
            float(np.median(products.darkening_magnitude[center].cpu())), 0.5
        )
        # Increasing L* toward image-right is 0 degrees (hue 0); decreasing L*
        # toward image-left is 180 degrees (OpenCV hue 90).
        self.assertLessEqual(float(np.median(lightening_hue)), 2.0)
        self.assertAlmostEqual(float(np.median(darkening_hue)), 90.0, delta=2.0)

    def test_surface_gradient_ceiling_zeros_only_slopes_above_the_limit(self) -> None:
        import numpy as np

        from seedvision.cuda import CudaContext
        from seedvision.cuda.layers import (
            surface_directional_darkness_gradients,
            upper_magnitude_surface_gradient,
        )
        from seedvision.visualization import AnalysisLayerSettings

        ramp = np.tile(np.linspace(20, 235, 96, dtype=np.uint8), (96, 1))
        image = np.repeat(ramp[:, :, None], 3, axis=2)
        valid = np.full((96, 96), 255, dtype=np.uint8)
        products = surface_directional_darkness_gradients(
            image,
            valid,
            30.0,
            AnalysisLayerSettings(
                surface_gradient_blur_sigma=0.5,
                surface_gradient_radius_fraction=0.30,
            ),
            cuda_context=CudaContext.resolve(requested="cpu"),
        )
        ceiling = 1.0
        filtered, _ = upper_magnitude_surface_gradient(
            products, ceiling, "lightening"
        )
        filtered_values = np.asarray(filtered)
        raw_values = products.lightening_magnitude.cpu().numpy()
        above = raw_values > ceiling
        retained = (raw_values > 0.0) & (raw_values <= ceiling)

        self.assertTrue(np.any(above))
        self.assertTrue(np.any(retained))
        self.assertFalse(np.any(filtered_values[above]))
        self.assertTrue(np.any(filtered_values[retained] > 0))

    def test_frequency_noise_masks_separate_darkness_and_colour_energy(self) -> None:
        import cv2
        import numpy as np

        from seedvision.cuda import CudaContext
        from seedvision.cuda.layers import multiscale_frequency_noise_masks
        from seedvision.visualization import AnalysisLayerSettings

        valid = np.full((128, 128), 255, dtype=np.uint8)
        checker = (np.indices((128, 64)).sum(axis=0) % 2) * 56 - 28
        darkness = np.full((128, 128), 140, dtype=np.int16)
        darkness[:, 64:] += checker
        darkness_image = np.repeat(
            np.clip(darkness, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2
        )
        settings = AnalysisLayerSettings(
            frequency_noise_fine_scale_fraction=0.008,
            frequency_noise_medium_scale_fraction=0.030,
            frequency_noise_coarse_scale_fraction=0.100,
            frequency_noise_context_fraction=0.025,
        )
        context = CudaContext.resolve(requested="cpu")
        darkness_products = multiscale_frequency_noise_masks(
            darkness_image,
            valid,
            40.0,
            settings,
            cuda_context=context,
        )
        darkness_fine = np.asarray(darkness_products.darkness_masks[0])
        self.assertGreater(
            float(np.mean(darkness_fine[:, 72:120])),
            float(np.mean(darkness_fine[:, 8:56])) + 40.0,
        )

        lab = np.empty((128, 128, 3), dtype=np.uint8)
        lab[:, :, 0] = 150
        lab[:, :, 1] = 128
        lab[:, :, 2] = 128
        lab[:, 64:, 1] = np.where(checker > 0, 170, 86).astype(np.uint8)
        colour_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        colour_products = multiscale_frequency_noise_masks(
            colour_image,
            valid,
            40.0,
            settings,
            cuda_context=context,
        )
        colour_fine = np.asarray(colour_products.colour_masks[0])
        self.assertGreater(
            float(np.mean(colour_fine[:, 72:120])),
            float(np.mean(colour_fine[:, 8:56])) + 40.0,
        )
        self.assertEqual(len(colour_products.darkness_masks), 3)
        self.assertEqual(len(colour_products.colour_masks), 3)
        self.assertTrue(
            all(np.asarray(mask).shape == (128, 128) for mask in (
                *colour_products.darkness_masks,
                *colour_products.colour_masks,
            ))
        )

    def test_surface_filter_change_reuses_gradient_and_frequency_node_caches(self) -> None:
        import numpy as np

        from seedvision.visualization import AnalysisLayerSettings, build_analysis_layers

        ramp = np.tile(np.linspace(25, 225, 80, dtype=np.uint8), (80, 1))
        image = np.repeat(ramp[:, :, None], 3, axis=2)
        valid = np.full((80, 80), 255, dtype=np.uint8)
        cache: dict[str, object] = {}
        build_analysis_layers(
            image,
            valid,
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            28.0,
            offset_x=0,
            offset_y=0,
            cache_values=cache,
        )
        first_gradients = cache["layer.surface_darkness_gradients"]
        first_filter = cache["layer.lightening_gradient_ceiling"]
        first_frequency = cache["layer.frequency_noise_masks"]

        build_analysis_layers(
            image,
            valid,
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            28.0,
            offset_x=0,
            offset_y=0,
            cache_values=cache,
            dirty_nodes={"lightening_gradient_ceiling"},
            settings=AnalysisLayerSettings(
                lightening_gradient_maximum_slope=0.75
            ),
        )
        self.assertIs(cache["layer.surface_darkness_gradients"], first_gradients)
        self.assertIs(cache["layer.frequency_noise_masks"], first_frequency)
        self.assertIsNot(cache["layer.lightening_gradient_ceiling"], first_filter)

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

    def test_annotated_seed_interiors_seed_distinct_instance_masks(self) -> None:
        import cv2
        import numpy as np

        from seedvision.visualization import build_analysis_layers

        image = np.full((96, 96, 3), 225, dtype=np.uint8)
        cv2.circle(image, (30, 30), 12, (30, 75, 175), -1)
        cv2.circle(image, (65, 65), 12, (45, 105, 155), -1)
        valid = np.full((96, 96), 255, dtype=np.uint8)
        annotations = np.zeros((96, 96), dtype=np.uint16)
        annotations[28:33, 28:33] = 17
        annotations[63:68, 63:68] = 42
        layers = build_analysis_layers(
            image,
            valid,
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            28.0,
            offset_x=0,
            offset_y=0,
            seed_instance_annotations=annotations,
        )

        labels = np.asarray(layers.instance_labels)
        self.assertEqual(int(labels[30, 30]), 1)
        self.assertEqual(int(labels[65, 65]), 2)
        self.assertEqual(set(np.unique(labels)), {0, 1, 2})
        self.assertGreater(np.count_nonzero(labels == 1), 25)
        self.assertGreater(np.count_nonzero(labels == 2), 25)

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
            layers.lightening_surface_gradient,
            layers.lightening_surface_direction,
            layers.darkening_surface_gradient,
            layers.darkening_surface_direction,
            layers.weak_lightening_surface_gradient,
            layers.weak_darkening_surface_gradient,
            *layers.darkness_frequency_noise_masks,
            *layers.colour_frequency_noise_masks,
            layers.edge_ridges,
            layers.reference_edge_ridges,
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
        first_reference_probability = cache["layer.reference_edge_probability"]
        first_reference_ridge = cache["layer.reference_edge_ridges"]

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
        self.assertIs(cache["layer.reference_edge_ridges"], first_reference_ridge)

        build_analysis_layers(
            image,
            valid,
            centers,
            radii,
            46.0,
            offset_x=0,
            offset_y=0,
            cache_values=cache,
            dirty_nodes={"reference_edge_ridges"},
            settings=AnalysisLayerSettings(reference_ridge_high_threshold=0.30),
        )
        self.assertIs(
            cache["layer.reference_edge_probability"],
            first_reference_probability,
        )
        self.assertIsNot(
            cache["layer.reference_edge_ridges"], first_reference_ridge
        )
        self.assertIs(cache["layer.seed_edge_curves"], final_only)

        build_analysis_layers(
            image,
            valid,
            centers,
            radii,
            52.0,
            offset_x=0,
            offset_y=0,
            cache_values=cache,
            dirty_nodes={"edge_traces", "seed_edge_curves"},
        )
        scale_changed = cache["layer.seed_edge_curves"]
        self.assertIs(scale_changed.ridge_state, final_only.ridge_state)
        self.assertIsNot(scale_changed.trace_state, final_only.trace_state)

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
        self.assertIs(trace_changed.ridge_state, scale_changed.ridge_state)
        self.assertIsNot(trace_changed.trace_state, scale_changed.trace_state)

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
            settings=AnalysisLayerSettings(
                trace_tangent_tolerance_degrees=30.0,
                trace_curvature_policy="require",
                trace_curvature_tolerance_degrees=4.0,
            ),
        )
        curvature_changed = cache["layer.seed_edge_curves"]
        self.assertIs(curvature_changed.ridge_state, trace_changed.ridge_state)
        self.assertIsNot(
            curvature_changed.trace_state, trace_changed.trace_state
        )

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
        self.assertIsNot(
            ridge_changed.ridge_state, curvature_changed.ridge_state
        )
        self.assertIsNot(
            ridge_changed.trace_state, curvature_changed.trace_state
        )


if __name__ == "__main__":
    unittest.main()
