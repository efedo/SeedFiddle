from __future__ import annotations

import unittest


class AnalysisLayerTests(unittest.TestCase):
    def test_reference_seed_traits_are_isolated_normalized_material_models(
        self,
    ) -> None:
        import cv2
        import numpy as np
        import torch

        from seedvision.cuda import CudaContext, GpuRaster
        from seedvision.cuda.layers import (
            directional_edges,
            multiscale_frequency_noise_masks,
            reference_seed_trait_probabilities,
        )
        from seedvision.persistence import SeedInstanceAnnotation
        from seedvision.visualization import AnalysisLayerSettings

        height = width = 96
        image = np.full((height, width, 3), (178, 194, 214), np.uint8)
        valid = np.full((height, width), 255, np.uint8)
        labels = np.zeros((height, width), np.uint16)
        for seed_id, centre in enumerate(
            ((25, 25), (71, 25), (25, 71), (71, 71)), start=1
        ):
            cv2.circle(labels, centre, 14, seed_id, -1)
        context = CudaContext.resolve(requested="cpu")
        settings = AnalysisLayerSettings(
            reference_seed_trait_prototypes_per_class=8,
            reference_seed_trait_minimum_samples_per_prototype=4,
            reference_seed_trait_fit_iterations=1,
            reference_seed_trait_context_fraction=0.02,
            reference_seed_trait_interior_buffer_fraction=0.05,
            reference_seed_trait_working_maximum_dimension=256,
        )
        gradients = directional_edges(
            image, valid, settings, cuda_context=context
        )
        frequency_noise = multiscale_frequency_noise_masks(
            image, valid, 28.0, settings, cuda_context=context
        )
        ridges = GpuRaster(
            torch.zeros((1, 1, height, width), dtype=torch.uint8),
            numpy_dtype=np.uint8,
            name="empty trait-test ridges",
        )
        seed_material = GpuRaster(
            torch.from_numpy(np.uint8(labels > 0))[None, None] * 255,
            numpy_dtype=np.uint8,
            name="trait-test seed material",
        )
        traits = (
            SeedInstanceAnnotation(
                1, "white", ("split",), conditions_reviewed=True
            ),
            SeedInstanceAnnotation(2, "white", (), conditions_reviewed=True),
            SeedInstanceAnnotation(
                3, "banded_dark", ("stained",), conditions_reviewed=True
            ),
            SeedInstanceAnnotation(
                4, "banded_dark", (), conditions_reviewed=False
            ),
        )

        products = reference_seed_trait_probabilities(
            gradients,
            ridges,
            frequency_noise,
            seed_material,
            28.0,
            settings,
            seed_instance_annotations=labels,
            seed_instance_traits=traits,
            annotation_species="lupinus_mutabilis",
            coat_patterns=("white", "banded_light", "banded_dark", "other"),
            conditions=("immature", "split", "wrinkled", "stained"),
            cuda_context=context,
        )

        coat = {
            name: np.asarray(raster)
            for name, raster in products.coat_probabilities
        }
        condition = {
            name: np.asarray(raster)
            for name, raster in products.condition_probabilities
        }
        support = labels > 0
        coat_sum = sum(values.astype(np.int16) for values in coat.values())
        self.assertTrue(products.profile.coat_model_available)
        self.assertGreaterEqual(int(coat_sum[support].min()), 254)
        self.assertLessEqual(int(coat_sum[support].max()), 256)
        self.assertFalse(np.any(coat_sum[~support]))
        self.assertFalse(np.any(coat["banded_light"]))
        self.assertFalse(np.any(coat["other"]))

        # Identical seed material in differently labelled seeds remains an
        # ambiguous model result. Authored pixels are not overwritten with
        # their labels, and the independent conditions can overlap.
        for values in (coat["white"], coat["banded_dark"]):
            self.assertGreater(int(values[25, 25]), 0)
            self.assertLess(int(values[25, 25]), 255)
        self.assertGreater(int(condition["split"][25, 25]), 0)
        self.assertGreater(int(condition["stained"][25, 25]), 0)
        self.assertFalse(np.any(condition["immature"]))
        self.assertFalse(np.any(condition["wrinkled"]))
        availability = dict(products.profile.condition_models_available)
        self.assertTrue(availability["split"])
        self.assertTrue(availability["stained"])
        self.assertFalse(availability["immature"])
        self.assertFalse(availability["wrinkled"])
        class_seed_counts = dict(products.profile.class_seed_counts)
        self.assertEqual(class_seed_counts["condition:split:absent"], 2)
        self.assertEqual(class_seed_counts["condition:stained:absent"], 2)

    def test_other_probability_overlays_are_direct_bright_and_lazy(self) -> None:
        import numpy as np
        import torch

        from seedvision.cuda import GpuRaster
        from seedvision.visualization import AnalysisLayers, NoiseFrequencyProfile

        colour_values = np.asarray(
            ((0, 64, 128), (192, 224, 255)), dtype=np.uint8
        )
        noise_values = np.asarray(
            ((255, 200, 150), (100, 50, 0)), dtype=np.uint8
        )
        valid_values = np.asarray(
            ((255, 255, 0), (255, 0, 255)), dtype=np.uint8
        )

        def lazy(values: np.ndarray, name: str) -> GpuRaster:
            return GpuRaster(
                torch.from_numpy(values)[None, None],
                numpy_dtype=np.uint8,
                name=name,
            )

        other_colour = lazy(colour_values, "Other colour probability")
        other_noise = lazy(noise_values, "Other noise probability")
        valid = lazy(valid_values, "valid mask")
        zeros = np.zeros_like(colour_values)
        layers = AnalysisLayers(
            offset_x=4,
            offset_y=7,
            instance_labels=zeros.astype(np.uint16),
            instance_colours=np.zeros((1, 3), dtype=np.uint8),
            background_likelihood=zeros,
            refined_background_likelihood=zeros,
            noise_frequency_profile=NoiseFrequencyProfile(
                band_scales_px=(1.0, 2.0, 4.0),
                descriptor_centre=(0.0, 0.0, 0.0),
                target_sample_count=0,
            ),
            edge_likelihood=zeros,
            directed_edge_hue=zeros,
            undirected_edge_hue=zeros,
            seed_edge_curve_likelihood=zeros,
            seed_edge_curve_radius_px=zeros.astype(np.float32),
            valid_mask=valid,
            other_colour_probability=other_colour,
            other_noise_probability=other_noise,
        )

        self.assertFalse(other_colour.is_materialized)
        self.assertFalse(other_noise.is_materialized)
        self.assertFalse(valid.is_materialized)

        colour_rgba = layers.other_colour_rgba()
        np.testing.assert_array_equal(colour_rgba[..., 0], colour_values)
        np.testing.assert_array_equal(colour_rgba[..., 1], colour_values)
        np.testing.assert_array_equal(colour_rgba[..., 2], colour_values)
        np.testing.assert_array_equal(
            colour_rgba[..., 3], np.uint8(valid_values > 0) * 255
        )
        self.assertEqual(other_colour.download_count, 1)
        self.assertEqual(valid.download_count, 1)
        self.assertFalse(other_noise.is_materialized)

        layers.other_colour_rgba()
        self.assertEqual(other_colour.download_count, 1)
        self.assertEqual(valid.download_count, 1)
        noise_rgba = layers.other_noise_rgba()
        np.testing.assert_array_equal(noise_rgba[..., 0], noise_values)
        np.testing.assert_array_equal(noise_rgba[..., 1], noise_values)
        np.testing.assert_array_equal(noise_rgba[..., 2], noise_values)
        np.testing.assert_array_equal(
            noise_rgba[..., 3], np.uint8(valid_values > 0) * 255
        )
        self.assertEqual(other_noise.download_count, 1)
        self.assertEqual(other_colour.download_count, 1)

    def test_missing_other_probabilities_render_black_not_an_unrelated_layer(self) -> None:
        import numpy as np

        from seedvision.visualization import AnalysisLayers, NoiseFrequencyProfile

        valid = np.asarray(((255, 0), (255, 255)), dtype=np.uint8)
        zeros = np.zeros((2, 2), dtype=np.uint8)
        layers = AnalysisLayers(
            offset_x=0,
            offset_y=0,
            instance_labels=zeros.astype(np.uint16),
            instance_colours=np.zeros((1, 3), dtype=np.uint8),
            background_likelihood=zeros,
            refined_background_likelihood=zeros,
            noise_frequency_profile=NoiseFrequencyProfile(
                band_scales_px=(1.0, 2.0, 4.0),
                descriptor_centre=(0.0, 0.0, 0.0),
                target_sample_count=0,
            ),
            edge_likelihood=np.full((2, 2), 255, dtype=np.uint8),
            directed_edge_hue=np.full((2, 2), 90, dtype=np.uint8),
            undirected_edge_hue=zeros,
            seed_edge_curve_likelihood=zeros,
            seed_edge_curve_radius_px=zeros.astype(np.float32),
            valid_mask=valid,
            other_colour_probability=None,
            other_noise_probability=None,
        )

        for rgba in (layers.other_colour_rgba(), layers.other_noise_rgba()):
            self.assertFalse(np.any(rgba[..., :3]))
            np.testing.assert_array_equal(
                rgba[..., 3], np.uint8(valid > 0) * 255
            )

    def test_signed_probability_excess_overlays_are_lazy_and_never_purple(
        self,
    ) -> None:
        import numpy as np
        import torch

        from seedvision.cuda import GpuRaster
        from seedvision.visualization import AnalysisLayers, NoiseFrequencyProfile

        valid_values = np.asarray(((255, 255), (0, 255)), dtype=np.uint8)
        foreground_colour_values = np.asarray(
            ((200, 50), (100, 70)), dtype=np.uint8
        )
        background_colour_values = np.asarray(
            ((100, 100), (60, 70)), dtype=np.uint8
        )
        other_colour_values = np.asarray(
            ((50, 80), (130, 70)), dtype=np.uint8
        )
        foreground_noise_values = np.asarray(
            ((20, 240), (90, 100)), dtype=np.uint8
        )
        background_noise_values = np.asarray(
            ((80, 100), (140, 100)), dtype=np.uint8
        )
        other_noise_values = np.asarray(
            ((60, 180), (120, 40)), dtype=np.uint8
        )
        physical_values = np.asarray(((190, 60), (90, 120)), dtype=np.uint8)
        nonphysical_values = np.asarray(
            ((100, 90), (120, 120)), dtype=np.uint8
        )

        def lazy(values: np.ndarray, name: str) -> GpuRaster:
            return GpuRaster(
                torch.from_numpy(values)[None, None],
                numpy_dtype=np.uint8,
                name=name,
            )

        foreground_colour = lazy(foreground_colour_values, "foreground colour")
        background_colour = lazy(background_colour_values, "background colour")
        other_colour = lazy(other_colour_values, "other colour")
        foreground_noise = lazy(foreground_noise_values, "foreground noise")
        background_noise = lazy(background_noise_values, "background noise")
        other_noise = lazy(other_noise_values, "other noise")
        physical = lazy(physical_values, "physical edge")
        nonphysical = lazy(nonphysical_values, "non-physical edge")
        valid = lazy(valid_values, "valid mask")
        zeros = np.zeros_like(valid_values)
        layers = AnalysisLayers(
            offset_x=0,
            offset_y=0,
            instance_labels=zeros.astype(np.uint16),
            instance_colours=np.zeros((1, 3), dtype=np.uint8),
            background_likelihood=background_colour,
            refined_background_likelihood=background_noise,
            noise_frequency_profile=NoiseFrequencyProfile(
                band_scales_px=(1.0, 2.0, 4.0),
                descriptor_centre=(0.0, 0.0, 0.0),
                target_sample_count=0,
            ),
            edge_likelihood=zeros,
            directed_edge_hue=zeros,
            undirected_edge_hue=zeros,
            seed_edge_curve_likelihood=zeros,
            seed_edge_curve_radius_px=zeros.astype(np.float32),
            valid_mask=valid,
            foreground_noise_likelihood=foreground_noise,
            other_colour_probability=other_colour,
            other_noise_probability=other_noise,
            physical_edge_probability=physical,
            non_edge_probability=nonphysical,
        )

        colour_rgba = layers.foreground_colour_excess_rgba(foreground_colour)
        np.testing.assert_array_equal(
            colour_rgba[..., 0], np.asarray(((0, 50), (30, 0)), np.uint8)
        )
        np.testing.assert_array_equal(
            colour_rgba[..., 2], np.asarray(((100, 0), (0, 0)), np.uint8)
        )
        self.assertFalse(foreground_noise.is_materialized)
        self.assertFalse(background_noise.is_materialized)
        self.assertFalse(other_noise.is_materialized)
        self.assertFalse(physical.is_materialized)
        self.assertFalse(nonphysical.is_materialized)

        noise_rgba = layers.foreground_noise_excess_rgba()
        np.testing.assert_array_equal(
            noise_rgba[..., 0], np.asarray(((60, 0), (50, 0)), np.uint8)
        )
        np.testing.assert_array_equal(
            noise_rgba[..., 2], np.asarray(((0, 60), (0, 0)), np.uint8)
        )

        # The original comparison still deliberately exposes overlapping class
        # support as magenta; the new direct margin cancels that overlap.
        comparison_rgba = layers.reference_edge_comparison_rgba()
        self.assertGreater(int(comparison_rgba[0, 0, 0]), 0)
        self.assertGreater(int(comparison_rgba[0, 0, 2]), 0)
        edge_rgba = layers.reference_edge_excess_rgba()
        np.testing.assert_array_equal(
            edge_rgba[..., 0], np.asarray(((0, 30), (30, 0)), np.uint8)
        )
        np.testing.assert_array_equal(
            edge_rgba[..., 2], np.asarray(((90, 0), (0, 0)), np.uint8)
        )

        for rgba in (colour_rgba, noise_rgba, edge_rgba):
            self.assertFalse(np.any(rgba[..., 1]))
            self.assertFalse(np.any((rgba[..., 0] > 0) & (rgba[..., 2] > 0)))
            np.testing.assert_array_equal(
                rgba[..., 3], np.uint8(valid_values > 0) * 255
            )

    def test_other_colour_probability_is_the_raw_painted_colour_model(self) -> None:
        import numpy as np

        from seedvision.cuda import CudaContext
        from seedvision.cuda.layers import background_colour_likelihood
        from seedvision.visualization import AnalysisLayerSettings

        image = np.full((64, 64, 3), (218, 222, 226), np.uint8)
        other_bgr = (35, 80, 180)
        image[8:24, 8:24] = other_bgr
        image[8:24, 40:56] = other_bgr
        valid = np.full((64, 64), 255, np.uint8)
        background = np.zeros((64, 64), bool)
        background[40:56, 8:24] = True
        other = np.zeros((64, 64), bool)
        other[8:24, 8:24] = True

        (
            _background_probability,
            _mode,
            _reference_count,
            _profile,
            other_probability,
        ) = background_colour_likelihood(
            image,
            valid,
            background_reference_mask=background,
            background_exclusion_mask=other,
            settings=AnalysisLayerSettings(
                background_refinement_iterations=0
            ),
            cuda_context=CudaContext.resolve(requested="cpu"),
            include_other_probability=True,
        )

        self.assertIsNotNone(other_probability)
        assert other_probability is not None
        self.assertFalse(other_probability.is_materialized)
        probability = np.asarray(other_probability)
        # The painted patch and an unpainted identical patch receive the same
        # learned colour membership. The output is not the painted mask and is
        # not an inverse of the Background probability.
        np.testing.assert_array_equal(
            probability[8:24, 8:24], probability[8:24, 40:56]
        )
        self.assertGreater(float(probability[8:24, 8:24].mean()), 200.0)
        self.assertLess(float(probability[40:56, 8:24].mean()), 20.0)

        without_other = background_colour_likelihood(
            image,
            valid,
            background_reference_mask=background,
            settings=AnalysisLayerSettings(
                background_refinement_iterations=0
            ),
            cuda_context=CudaContext.resolve(requested="cpu"),
            include_other_probability=True,
        )
        self.assertIsNone(without_other[4])

    def test_other_noise_probability_uses_other_as_positive_texture_class(self) -> None:
        import numpy as np

        from seedvision.cuda import CudaContext
        from seedvision.cuda.layers import noise_frequency_background_likelihood
        from seedvision.visualization import AnalysisLayerSettings

        image = np.full((64, 64, 3), 210, np.uint8)
        checker = (
            (np.indices((16, 16)).sum(axis=0) % 2) * 80 + 60
        ).astype(np.uint8)
        for left in (8, 40):
            image[8:24, left : left + 16] = checker[..., None]
        valid = np.full((64, 64), 255, np.uint8)
        background = np.zeros((64, 64), bool)
        background[40:48, 8:16] = True
        foreground = np.zeros((64, 64), bool)
        foreground[40:48, 40:48] = True
        other = np.zeros((64, 64), bool)
        other[8:24, 8:24] = True
        background_colour = np.full((64, 64), 220, np.uint8)
        other_colour = np.zeros((64, 64), np.uint8)
        other_colour[8:24, 8:24] = 250
        other_colour[8:24, 40:56] = 250
        settings = AnalysisLayerSettings(
            noise_vector_length_fraction=0.20,
            noise_vector_sample_count=5,
        )

        result = noise_frequency_background_likelihood(
            image,
            valid,
            background_colour,
            24.0,
            settings,
            background_reference_mask=background,
            foreground_reference_mask=foreground,
            target_exclusion_mask=other,
            other_colour_likelihood=other_colour,
            cuda_context=CudaContext.resolve(requested="cpu"),
            include_other_probability=True,
        )
        other_noise = result[4]
        other_profile = result[5]
        self.assertIsNotNone(other_noise)
        self.assertIsNotNone(other_profile)
        assert other_noise is not None and other_profile is not None
        self.assertFalse(other_noise.is_materialized)
        self.assertEqual(other_profile.target_sample_count, 16 * 16)
        self.assertGreaterEqual(other_profile.compatibility_half_distance, 1.0)
        self.assertEqual(len(other_profile.descriptor_names), 9)
        probability = np.asarray(other_noise)
        painted_mean = float(probability[11:21, 11:21].mean())
        matching_mean = float(probability[11:21, 43:53].mean())
        nonother_mean = float(
            np.concatenate(
                (probability[40:48, 8:16], probability[40:48, 40:48])
            ).mean()
        )
        self.assertGreater(painted_mean, nonother_mean + 60.0)
        self.assertGreater(matching_mean, nonother_mean + 60.0)
        self.assertAlmostEqual(painted_mean, matching_mean, delta=12.0)

        without_other = noise_frequency_background_likelihood(
            image,
            valid,
            background_colour,
            24.0,
            settings,
            background_reference_mask=background,
            foreground_reference_mask=foreground,
            cuda_context=CudaContext.resolve(requested="cpu"),
            include_other_probability=True,
        )
        self.assertIsNone(without_other[4])
        self.assertIsNone(without_other[5])

    def test_noise_classifiers_are_target_only_and_invariant_to_sibling_content(self) -> None:
        import numpy as np

        from seedvision.cuda import CudaContext
        from seedvision.cuda.layers import (
            noise_frequency_background_likelihood,
            noise_frequency_foreground_likelihood,
        )
        from seedvision.visualization import AnalysisLayerSettings

        image = np.full((128, 128, 3), 220, np.uint8)
        checker = (
            (np.indices((64, 64)).sum(axis=0) % 2) * 85 + 65
        ).astype(np.uint8)
        stripes = np.broadcast_to(
            ((np.arange(64)[:, None] // 2) % 2) * 70 + 90,
            (64, 64),
        ).astype(np.uint8)
        image[:64, 64:] = checker[..., None]
        image[64:, :64] = stripes[..., None]
        valid = np.full((128, 128), 255, np.uint8)

        foreground = np.zeros((128, 128), bool)
        foreground[16:48, 80:112] = True
        other_once = np.zeros((128, 128), bool)
        other_once[80:88, 16:24] = True
        other_alternative = np.zeros((128, 128), bool)
        other_alternative[80:112, 80:112] = True
        background_once = np.zeros((128, 128), bool)
        background_once[16:24, 16:24] = True
        background_alternative = np.zeros((128, 128), bool)
        background_alternative[0:32, 64:80] = True
        foreground_alternative = np.zeros((128, 128), bool)
        foreground_alternative[80:112, 80:112] = True

        settings = AnalysisLayerSettings(
            noise_direction_step_degrees=90,
            noise_vector_length_fraction=0.10,
            noise_vector_sample_count=2,
            foreground_noise_direction_step_degrees=90,
            foreground_noise_vector_length_fraction=0.10,
            foreground_noise_vector_sample_count=2,
        )
        context = CudaContext.resolve(requested="cpu")

        def background_result(background, other):
            return noise_frequency_background_likelihood(
                image,
                valid,
                None,
                24.0,
                settings,
                background_reference_mask=background,
                foreground_reference_mask=foreground,
                other_reference_mask=other,
                include_other_probability=True,
                cuda_context=context,
            )

        other_first = background_result(background_once, other_once)
        other_changed_background = background_result(
            background_alternative, other_once
        )
        np.testing.assert_array_equal(
            np.asarray(other_first[4]),
            np.asarray(other_changed_background[4]),
        )

        # Neither different Other content nor different Foreground content can
        # alter a fixed Background target model.
        background_first = background_result(background_once, other_once)
        background_changed_other = background_result(
            background_once, other_alternative
        )
        np.testing.assert_array_equal(
            np.asarray(background_first[0]),
            np.asarray(background_changed_other[0]),
        )
        background_changed_foreground = noise_frequency_background_likelihood(
            image,
            valid,
            None,
            24.0,
            settings,
            background_reference_mask=background_once,
            foreground_reference_mask=foreground_alternative,
            other_reference_mask=other_once,
            include_other_probability=True,
            cuda_context=context,
        )
        np.testing.assert_array_equal(
            np.asarray(background_first[0]),
            np.asarray(background_changed_foreground[0]),
        )
        np.testing.assert_array_equal(
            np.asarray(other_first[4]),
            np.asarray(background_changed_foreground[4]),
        )

        def foreground_result(background, other):
            return noise_frequency_foreground_likelihood(
                image,
                valid,
                None,
                24.0,
                settings,
                background_reference_mask=background,
                foreground_reference_mask=foreground,
                foreground_exclusion_mask=other,
                cuda_context=context,
            )

        foreground_first = foreground_result(background_once, other_once)
        foreground_changed_siblings = foreground_result(
            background_alternative, other_alternative
        )
        np.testing.assert_array_equal(
            np.asarray(foreground_first[0]),
            np.asarray(foreground_changed_siblings[0]),
        )

        # Changing the target itself must still change the corresponding map.
        other_changed_target = background_result(
            background_once, other_alternative
        )
        self.assertFalse(
            np.array_equal(
                np.asarray(other_first[4]),
                np.asarray(other_changed_target[4]),
            )
        )

    def test_noise_descriptor_retains_multiscale_orientation_structure(self) -> None:
        import torch

        from seedvision.cuda.layers import _noise_texture_features

        columns = torch.arange(96, dtype=torch.float32)
        stripes = ((columns // 2).remainder(2.0))[None, None, None, :]
        lab = stripes.expand(1, 3, 96, 96).contiguous()
        features = _noise_texture_features(lab, (0.65, 2.0, 6.0))

        self.assertEqual(tuple(features.shape), (1, 9, 96, 96))
        for offset in (0, 3, 6):
            principal = float(features[0, offset + 1, 16:-16, 16:-16].mean())
            cross_axis = float(features[0, offset + 2, 16:-16, 16:-16].mean())
            self.assertGreater(principal, cross_axis + 0.25)

    def test_background_noise_fits_directly_from_external_perimeter_texture(self) -> None:
        import numpy as np

        from seedvision.cuda import CudaContext, image_to_tensor
        from seedvision.cuda.layers import noise_frequency_background_likelihood
        from seedvision.visualization import AnalysisLayerSettings

        context = CudaContext.resolve(requested="cpu")
        image = np.full((96, 96, 3), 220, np.uint8)
        checker = ((np.indices((96, 48)).sum(axis=0) % 2) * 120 + 55).astype(
            np.uint8
        )
        image[:, 48:] = checker[..., None]
        valid = np.full((96, 96), 255, np.uint8)
        nonbackground = np.zeros((96, 96), bool)
        nonbackground[12:84, 54:90] = True
        colour = np.full((96, 96), 128, np.uint8)

        exterior = np.full((80, 120, 3), 222, np.uint8)
        exterior_valid = np.zeros((80, 120), np.uint8)
        exterior_valid[24:56, 40:72] = 255
        result = noise_frequency_background_likelihood(
            image,
            valid,
            colour,
            32.0,
            AnalysisLayerSettings(
                noise_direction_integration="mean",
                noise_vector_length_fraction=0.20,
            ),
            foreground_reference_mask=nonbackground,
            external_target_source_tensor=image_to_tensor(exterior, context),
            external_target_valid_tensor=image_to_tensor(
                exterior_valid, context
            )
            > 0,
            cuda_context=context,
        )

        probability = np.asarray(result[0])
        profile = result[1]
        self.assertEqual(profile.target_sample_count, 32 * 32)
        self.assertGreaterEqual(profile.compatibility_half_distance, 1.0)
        self.assertGreater(
            float(probability[18:78, 8:40].mean()),
            float(probability[18:78, 56:88].mean()) + 35.0,
        )

    def test_other_probability_rasters_follow_existing_node_cache_boundaries(self) -> None:
        import numpy as np

        from seedvision.visualization import build_analysis_layers

        image = np.full((48, 48, 3), (215, 220, 225), np.uint8)
        image[8:20, 8:20] = (35, 80, 180)
        valid = np.full((48, 48), 255, np.uint8)
        background = np.zeros((48, 48), bool)
        background[28:40, 4:16] = True
        foreground = np.zeros((48, 48), bool)
        foreground[28:40, 32:44] = True
        other = np.zeros((48, 48), bool)
        other[8:20, 8:20] = True
        common = dict(
            background_reference_mask=background,
            foreground_reference_mask=foreground,
            background_exclusion_mask=other,
            foreground_exclusion_mask=other,
            instance_masks_enabled=False,
            seed_edge_curves_enabled=False,
            reference_texture_prototypes_enabled=False,
            reference_edge_probability_enabled=False,
            reference_edge_ridges_enabled=False,
        )
        cache: dict[str, object] = {}

        def calculate(dirty_nodes=frozenset()):
            return build_analysis_layers(
                image,
                valid,
                np.empty((0, 2), np.float32),
                np.empty((0,), np.float32),
                20.0,
                offset_x=0,
                offset_y=0,
                cache_values=cache,
                dirty_nodes=dirty_nodes,
                **common,
            )

        first = calculate()
        self.assertIsNotNone(first.other_colour_probability)
        self.assertIsNotNone(first.other_noise_probability)
        second = calculate()
        self.assertIs(second.other_colour_probability, first.other_colour_probability)
        self.assertIs(second.other_noise_probability, first.other_noise_probability)
        unrelated = calculate({"directed_edges"})
        self.assertIs(
            unrelated.other_colour_probability, first.other_colour_probability
        )
        self.assertIs(
            unrelated.other_noise_probability, first.other_noise_probability
        )
        repainted = calculate({"project"})
        self.assertIsNot(
            repainted.other_colour_probability, first.other_colour_probability
        )
        self.assertIsNot(
            repainted.other_noise_probability, first.other_noise_probability
        )

    def test_kept_perimeter_background_source_is_ring_only_and_additive_to_paint(self) -> None:
        import cv2
        import numpy as np
        import torch

        from seedvision.cuda import CudaContext
        from seedvision.cuda.layers import background_colour_likelihood
        from seedvision.visualization import AnalysisLayerSettings

        height, width = 48, 64
        image = np.full((height, width, 3), (25, 75, 175), np.uint8)
        perimeter_bgr = np.asarray((222, 224, 226), np.uint8)
        painted_bgr = np.asarray((60, 165, 95), np.uint8)
        image[4:12, 4:12] = perimeter_bgr
        image[4:12, 16:24] = perimeter_bgr
        image[4:12, 28:36] = perimeter_bgr
        image[4:12, 40:48] = perimeter_bgr
        image[4:12, 52:60] = perimeter_bgr
        image[28:36, 4:12] = painted_bgr
        valid = np.full((height, width), 255, np.uint8)
        painted = np.zeros((height, width), bool)
        painted[28:36, 4:12] = True
        perimeter_source = np.zeros((height, width), bool)
        perimeter_source[4:12, 4:12] = True
        foreground = np.zeros((height, width), bool)
        foreground[4:12, 28:36] = True
        other = np.zeros((height, width), bool)
        other[4:12, 40:48] = True
        annotations = np.zeros((height, width), np.uint16)
        annotations[4:12, 52:60] = 7
        perimeter_lab = cv2.cvtColor(
            perimeter_bgr.reshape(1, 1, 3), cv2.COLOR_BGR2LAB
        )[0, 0].astype(np.float32)
        ring_samples = torch.as_tensor(
            np.repeat(perimeter_lab[None, :], 16, axis=0)
        )
        settings = AnalysisLayerSettings(
            background_keep_perimeter_reference=True,
            background_prior_tolerance=8.0,
            background_colour_components=8,
            background_refinement_iterations=0,
        )

        (
            kept_probability,
            _mode,
            painted_count,
            kept_profile,
            _other_probability,
            source_mask,
        ) = background_colour_likelihood(
            image,
            valid,
            background_reference_mask=painted,
            foreground_reference_mask=foreground,
            background_exclusion_mask=other,
            seed_instance_annotations=annotations,
            background_prior_lab=tuple(float(value) for value in perimeter_lab),
            background_prior_samples_lab=ring_samples,
            background_prior_source_mask=perimeter_source,
            settings=settings,
            cuda_context=CudaContext.resolve(requested="cpu"),
            include_other_probability=True,
            include_reference_source_mask=True,
        )

        self.assertIsNotNone(source_mask)
        assert source_mask is not None
        self.assertFalse(source_mask.is_materialized)
        expected_source = perimeter_source.copy()
        expected_source &= ~painted
        expected_source &= ~foreground
        expected_source &= ~other
        expected_source &= annotations == 0
        np.testing.assert_array_equal(
            np.asarray(source_mask) > 0, expected_source
        )
        self.assertEqual(painted_count, int(painted.sum()))
        self.assertEqual(
            kept_profile.sample_count,
            int(painted.sum()) + len(ring_samples),
        )
        kept_values = np.asarray(kept_probability)
        self.assertGreater(float(kept_values[4:12, 4:12].mean()), 220.0)
        # Identical in-dish colour remains a likely model output, but is not a
        # retained source region merely because it resembles the ring.
        self.assertGreater(float(kept_values[4:12, 16:24].mean()), 220.0)
        self.assertFalse(bool(np.asarray(source_mask)[4:12, 16:24].any()))
        self.assertGreater(float(kept_values[28:36, 4:12].mean()), 150.0)

        (
            manual_probability,
            _manual_mode,
            manual_count,
            manual_profile,
            _manual_other,
            omitted_source,
        ) = background_colour_likelihood(
            image,
            valid,
            background_reference_mask=painted,
            foreground_reference_mask=foreground,
            background_exclusion_mask=other,
            seed_instance_annotations=annotations,
            background_prior_lab=tuple(float(value) for value in perimeter_lab),
            background_prior_samples_lab=ring_samples,
            settings=AnalysisLayerSettings(
                background_keep_perimeter_reference=False,
                background_prior_tolerance=8.0,
                background_colour_components=8,
                background_refinement_iterations=0,
            ),
            cuda_context=CudaContext.resolve(requested="cpu"),
            include_other_probability=True,
            include_reference_source_mask=True,
        )
        self.assertIsNone(omitted_source)
        self.assertEqual(manual_count, int(painted.sum()))
        self.assertEqual(manual_profile.sample_count, int(painted.sum()))
        self.assertGreater(
            float(np.asarray(manual_probability)[28:36, 4:12].mean()),
            220.0,
        )

    def test_tiny_painted_background_anchor_is_balanced_with_ring_samples_only(self) -> None:
        import cv2
        import numpy as np
        import torch

        from seedvision.cuda import CudaContext
        from seedvision.cuda.layers import background_colour_likelihood
        from seedvision.visualization import AnalysisLayerSettings

        perimeter_bgr = np.asarray((222, 225, 228), np.uint8)
        painted_bgr = np.asarray((45, 165, 85), np.uint8)
        image = np.empty((96, 96, 3), np.uint8)
        image[:] = perimeter_bgr
        painted = np.zeros(image.shape[:2], bool)
        painted[9:12, 13:16] = True
        image[painted] = painted_bgr
        valid = np.full(image.shape[:2], 255, np.uint8)
        perimeter_source = np.zeros(image.shape[:2], bool)
        perimeter_source[0, :64] = True
        perimeter_lab = cv2.cvtColor(
            perimeter_bgr.reshape(1, 1, 3), cv2.COLOR_BGR2LAB
        )[0, 0].astype(np.float32)

        probability, _mode, count, profile, source_mask = (
            background_colour_likelihood(
                image,
                valid,
                background_reference_mask=painted,
                background_prior_lab=tuple(
                    float(value) for value in perimeter_lab
                ),
                background_prior_samples_lab=torch.as_tensor(
                    np.repeat(perimeter_lab[None, :], 64, axis=0)
                ),
                background_prior_source_mask=perimeter_source,
                settings=AnalysisLayerSettings(
                    background_keep_perimeter_reference=True,
                    background_prior_tolerance=8.0,
                    background_colour_components=8,
                    background_refinement_iterations=0,
                ),
                cuda_context=CudaContext.resolve(requested="cpu"),
                include_reference_source_mask=True,
            )
        )
        self.assertEqual(count, int(painted.sum()))
        self.assertIsNotNone(source_mask)
        assert source_mask is not None
        np.testing.assert_array_equal(
            np.asarray(source_mask) > 0, perimeter_source
        )
        self.assertEqual(profile.sample_count, int(painted.sum()) + 64)
        values = np.asarray(probability)
        self.assertGreater(float(values[painted].mean()), 220.0)
        self.assertGreater(float(values[~painted].mean()), 220.0)

    def test_perimeter_opt_out_without_paint_uses_nonperimeter_fallback(self) -> None:
        import cv2
        import numpy as np
        import torch

        from seedvision.cuda import CudaContext
        from seedvision.cuda.layers import background_colour_likelihood
        from seedvision.visualization import AnalysisLayerSettings

        image = np.full((32, 40, 3), (205, 212, 220), np.uint8)
        image[:, :20] = (75, 95, 155)
        valid = np.full(image.shape[:2], 255, np.uint8)
        annotations = np.zeros(image.shape[:2], np.uint16)
        annotations[:, 20:] = 3
        perimeter_bgr = np.asarray((235, 238, 240), np.uint8)
        perimeter_lab = cv2.cvtColor(
            perimeter_bgr.reshape(1, 1, 3), cv2.COLOR_BGR2LAB
        )[0, 0].astype(np.float32)
        ring_samples = torch.as_tensor(
            np.repeat(perimeter_lab[None, :], 64, axis=0)
        )

        result = background_colour_likelihood(
            image,
            valid,
            background_prior_lab=tuple(float(value) for value in perimeter_lab),
            background_prior_samples_lab=ring_samples,
            seed_instance_annotations=annotations,
            settings=AnalysisLayerSettings(
                background_keep_perimeter_reference=False,
                background_refinement_iterations=0,
            ),
            cuda_context=CudaContext.resolve(requested="cpu"),
            include_other_probability=True,
            include_reference_source_mask=True,
        )
        probability, mode, count, profile, _other, source_mask = result
        self.assertIsNone(source_mask)
        self.assertEqual(mode, "automatic")
        self.assertEqual(count, 0)
        self.assertGreater(profile.sample_count, 0)
        self.assertLessEqual(
            profile.sample_count, int(np.count_nonzero(annotations == 0))
        )
        self.assertEqual(np.asarray(probability).shape, valid.shape)
        probability_values = np.asarray(probability)
        self.assertGreater(
            float(probability_values[:, :20].mean()),
            float(probability_values[:, 20:].mean()) + 50.0,
        )

    def test_automatic_background_source_trains_noise_and_prototypes_gpu_safely(self) -> None:
        import numpy as np
        import torch

        from seedvision.cuda import CudaContext, GpuRaster
        from seedvision.cuda.layers import (
            directional_edges,
            multiscale_frequency_noise_masks,
            noise_frequency_background_likelihood,
            reference_texture_probabilities,
        )
        from seedvision.visualization import AnalysisLayerSettings

        height = width = 48
        yy, xx = np.indices((height, width))
        texture = ((xx * 11 + yy * 7) % 43).astype(np.uint8)
        image = np.empty((height, width, 3), np.uint8)
        image[:, :, 0] = 170 + texture // 4
        image[:, :, 1] = 185 + texture // 5
        image[:, :, 2] = 200 + texture // 6
        valid = np.full((height, width), 255, np.uint8)
        painted = np.zeros((height, width), bool)
        painted[2:10, 2:10] = True
        automatic = np.zeros((height, width), bool)
        automatic[8:34, 8:34] = True
        foreground = np.zeros((height, width), bool)
        foreground[16:25, 16:25] = True
        other = np.zeros((height, width), bool)
        other[25:32, 25:32] = True
        annotations = np.zeros((height, width), np.uint16)
        annotations[10:15, 25:30] = 11
        expected_noise_automatic = (
            automatic & ~painted & ~foreground & ~other
        )
        expected_automatic = expected_noise_automatic & (annotations == 0)
        expected_noise_background = painted | expected_noise_automatic
        expected_background = painted | expected_automatic
        # Instance IDs alone remain edge supervision. They become material
        # Foreground only when the opt-in derived source is supplied explicitly.
        expected_foreground = foreground
        context = CudaContext.resolve(requested="cpu")
        source = GpuRaster(
            torch.from_numpy(np.uint8(automatic)[None, None] * 255),
            numpy_dtype=np.uint8,
            name="retained automatic Background source",
        )
        self.assertFalse(source.is_materialized)
        settings = AnalysisLayerSettings(
            noise_vector_length_fraction=0.20,
            noise_vector_sample_count=5,
            reference_texture_minimum_samples_per_prototype=16,
            reference_texture_fit_iterations=1,
        )

        noise_result = noise_frequency_background_likelihood(
            image,
            valid,
            np.full((height, width), 128, np.uint8),
            20.0,
            settings,
            background_reference_mask=painted,
            automatic_target_reference_mask=source,
            foreground_reference_mask=foreground,
            target_exclusion_mask=other,
            cuda_context=context,
        )
        self.assertEqual(
            noise_result[1].target_sample_count,
            int(expected_noise_background.sum()),
        )
        self.assertFalse(source.is_materialized)

        gradients = directional_edges(
            image, valid, settings, cuda_context=context
        )
        frequency_noise = multiscale_frequency_noise_masks(
            image, valid, 20.0, settings, cuda_context=context
        )
        ridges = GpuRaster(
            torch.zeros((1, 1, height, width), dtype=torch.uint8),
            numpy_dtype=np.uint8,
            name="empty test ridges",
        )
        products = reference_texture_probabilities(
            image,
            gradients,
            ridges,
            frequency_noise,
            20.0,
            settings,
            background_reference_mask=painted,
            background_reference_source_mask=source,
            foreground_reference_mask=foreground,
            other_reference_mask=other,
            seed_instance_annotations=annotations,
            cuda_context=context,
        )
        source_counts = dict(products.profile.class_sample_counts)
        self.assertEqual(
            products.background_sample_count, int(expected_background.sum())
        )
        self.assertEqual(
            source_counts["background"], int(expected_background.sum())
        )
        # Automatic Background is subordinate: overlapping semantic paint
        # remains available to its authored class instead of overlap cleanup
        # deleting both labels.
        self.assertEqual(
            products.foreground_sample_count, int(expected_foreground.sum())
        )
        self.assertEqual(products.other_sample_count, int(other.sum()))
        self.assertFalse(source.is_materialized)

    def test_annotated_foreground_source_trains_noise_and_material_prototypes_only_when_supplied(self) -> None:
        import cv2
        import numpy as np
        import torch

        from seedvision.cuda import CudaContext, GpuRaster
        from seedvision.cuda.layers import (
            directional_edges,
            multiscale_frequency_noise_masks,
            noise_frequency_foreground_likelihood,
            reference_texture_probabilities,
        )
        from seedvision.visualization import AnalysisLayerSettings

        height = width = 64
        rng = np.random.default_rng(4)
        image = np.full((height, width, 3), 218, np.uint8)
        texture = rng.integers(
            -25, 26, (height, width, 1), dtype=np.int16
        )
        image = np.clip(
            image.astype(np.int16) + texture, 0, 255
        ).astype(np.uint8)
        valid = np.full((height, width), 255, np.uint8)
        painted_foreground = np.zeros((height, width), bool)
        painted_foreground[4:12, 4:12] = True
        painted_background = np.zeros((height, width), bool)
        painted_background[50:58, 4:12] = True
        other = np.zeros((height, width), bool)
        other[4:12, 50:58] = True
        automatic = np.zeros((height, width), bool)
        automatic[16:48, 16:48] = True
        # Direct consumers must repeat semantic precedence defensively rather
        # than trusting that every caller supplied a pre-cleaned source.
        automatic[50:55, 4:9] = True
        automatic[5:10, 52:57] = True
        expected_automatic = (
            automatic
            & ~painted_foreground
            & ~painted_background
            & ~other
        )
        expected_foreground = painted_foreground | expected_automatic
        annotations = np.zeros((height, width), np.uint16)
        cv2.circle(annotations, (32, 32), 18, 1, -1)
        context = CudaContext.resolve(requested="cpu")
        source = GpuRaster(
            torch.from_numpy(automatic.astype(np.uint8))[None, None] * 255,
            numpy_dtype=np.uint8,
            name="automatic annotated Foreground source",
        )
        settings = AnalysisLayerSettings(
            noise_vector_length_fraction=0.20,
            noise_vector_sample_count=5,
            reference_texture_minimum_samples_per_prototype=8,
            reference_texture_fit_iterations=1,
        )

        noise_result = noise_frequency_foreground_likelihood(
            image,
            valid,
            np.full((height, width), 128, np.uint8),
            36.0,
            settings,
            background_reference_mask=painted_background,
            foreground_reference_mask=painted_foreground,
            automatic_foreground_reference_mask=source,
            foreground_exclusion_mask=other,
            cuda_context=context,
        )
        self.assertEqual(
            noise_result[1].target_sample_count,
            int(expected_foreground.sum()),
        )
        self.assertEqual(
            len(noise_result[1].descriptor_centre),
            9,
        )
        self.assertFalse(source.is_materialized)

        gradients = directional_edges(
            image, valid, settings, cuda_context=context
        )
        frequency_noise = multiscale_frequency_noise_masks(
            image, valid, 36.0, settings, cuda_context=context
        )
        ridges = GpuRaster(
            torch.zeros((1, 1, height, width), dtype=torch.uint8),
            numpy_dtype=np.uint8,
            name="empty test ridges",
        )
        enabled = reference_texture_probabilities(
            image,
            gradients,
            ridges,
            frequency_noise,
            36.0,
            settings,
            background_reference_mask=painted_background,
            foreground_reference_mask=painted_foreground,
            foreground_reference_source_mask=source,
            other_reference_mask=other,
            seed_instance_annotations=annotations,
            cuda_context=context,
        )
        enabled_counts = dict(enabled.profile.class_sample_counts)
        self.assertEqual(
            enabled_counts["foreground"], int(expected_foreground.sum())
        )
        self.assertEqual(
            enabled.foreground_sample_count, int(expected_foreground.sum())
        )
        self.assertGreater(enabled.physical_sample_count, 0)
        self.assertFalse(source.is_materialized)

        opted_out = reference_texture_probabilities(
            image,
            gradients,
            ridges,
            frequency_noise,
            36.0,
            settings,
            background_reference_mask=painted_background,
            foreground_reference_mask=painted_foreground,
            other_reference_mask=other,
            # Annotations remain edge supervision, but are not silently reused
            # as Foreground material when the derived source is absent.
            seed_instance_annotations=annotations,
            cuda_context=context,
        )
        opted_out_counts = dict(opted_out.profile.class_sample_counts)
        self.assertEqual(
            opted_out_counts["foreground"], int(painted_foreground.sum())
        )
        self.assertEqual(
            opted_out.foreground_sample_count, int(painted_foreground.sum())
        )
        self.assertEqual(
            opted_out.physical_sample_count, enabled.physical_sample_count
        )

    def test_background_reference_source_obeys_background_cache_boundaries(self) -> None:
        import cv2
        import numpy as np
        import torch

        from seedvision.visualization import AnalysisLayerSettings, build_analysis_layers

        image = np.full((40, 48, 3), (45, 90, 175), np.uint8)
        perimeter_bgr = np.asarray((220, 225, 230), np.uint8)
        image[5:19, 5:19] = perimeter_bgr
        valid = np.full(image.shape[:2], 255, np.uint8)
        painted = np.zeros(image.shape[:2], bool)
        painted[25:33, 6:14] = True
        perimeter_source = np.zeros(image.shape[:2], bool)
        perimeter_source[5:19, 5:19] = True
        perimeter_lab = cv2.cvtColor(
            perimeter_bgr.reshape(1, 1, 3), cv2.COLOR_BGR2LAB
        )[0, 0].astype(np.float32)
        common = dict(
            background_reference_mask=painted,
            background_prior_lab=tuple(float(value) for value in perimeter_lab),
            background_prior_samples_lab=torch.as_tensor(
                np.repeat(perimeter_lab[None, :], 32, axis=0)
            ),
            background_prior_source_mask=perimeter_source,
            foreground_noise_enabled=False,
            reference_edge_probability_enabled=False,
            reference_edge_ridges_enabled=False,
            reference_texture_prototypes_enabled=False,
            surface_darkness_gradients_enabled=False,
            lightening_gradient_ceiling_enabled=False,
            darkening_gradient_ceiling_enabled=False,
            instance_masks_enabled=False,
            seed_edge_curves_enabled=False,
        )
        cache: dict[str, object] = {}

        def calculate(settings, dirty_nodes=frozenset()):
            return build_analysis_layers(
                image,
                valid,
                np.empty((0, 2), np.float32),
                np.empty((0,), np.float32),
                20.0,
                offset_x=0,
                offset_y=0,
                settings=settings,
                cache_values=cache,
                dirty_nodes=dirty_nodes,
                **common,
            )

        kept_settings = AnalysisLayerSettings(
            background_keep_perimeter_reference=True,
            background_prior_tolerance=8.0,
            background_refinement_iterations=0,
        )
        first = calculate(kept_settings)
        self.assertIsNotNone(first.background_reference_source_mask)
        source = first.background_reference_source_mask
        noise_profile = first.noise_frequency_profile
        reused = calculate(kept_settings)
        self.assertIs(reused.background_reference_source_mask, source)
        self.assertIs(reused.noise_frequency_profile, noise_profile)
        unrelated = calculate(kept_settings, {"directed_edges"})
        self.assertIs(unrelated.background_reference_source_mask, source)
        self.assertIs(unrelated.noise_frequency_profile, noise_profile)

        opted_out = calculate(
            AnalysisLayerSettings(
                background_keep_perimeter_reference=False,
                background_prior_tolerance=8.0,
                background_refinement_iterations=0,
            ),
            {"background_likelihood"},
        )
        self.assertIsNone(opted_out.background_reference_source_mask)
        self.assertIsNot(opted_out.noise_frequency_profile, noise_profile)

        restored = calculate(
            kept_settings, {"background_likelihood"}
        )
        self.assertIsNotNone(restored.background_reference_source_mask)
        self.assertIsNot(restored.background_reference_source_mask, source)

    def test_annotated_foreground_source_obeys_foreground_cache_boundaries(self) -> None:
        import numpy as np
        import torch

        from seedvision.cuda import GpuRaster
        from seedvision.visualization import AnalysisLayerSettings, build_analysis_layers

        height = width = 48
        image = np.full((height, width, 3), (105, 145, 185), np.uint8)
        valid = np.full((height, width), 255, np.uint8)
        foreground_probability = np.full((height, width), 180, np.uint8)
        painted_background = np.zeros((height, width), bool)
        painted_background[2:10, 2:10] = True
        painted_foreground = np.zeros((height, width), bool)
        painted_foreground[36:44, 36:44] = True
        source_values_1 = np.zeros((height, width), np.uint8)
        source_values_1[12:22, 12:22] = 255
        source_values_2 = np.zeros((height, width), np.uint8)
        source_values_2[12:32, 12:32] = 255

        def source(values, name):
            return GpuRaster(
                torch.from_numpy(values)[None, None],
                numpy_dtype=np.uint8,
                name=name,
            )

        first_source = source(source_values_1, "first annotated FG source")
        second_source = source(source_values_2, "second annotated FG source")
        settings = AnalysisLayerSettings(
            reference_texture_material_prototypes_per_class=8,
            reference_texture_minimum_samples_per_prototype=4,
            reference_texture_fit_iterations=1,
        )
        cache: dict[str, object] = {}
        common = dict(
            foreground_probability=foreground_probability,
            background_reference_mask=painted_background,
            foreground_reference_mask=painted_foreground,
            settings=settings,
            cache_values=cache,
            reference_edge_probability_enabled=False,
            reference_edge_ridges_enabled=False,
            surface_darkness_gradients_enabled=False,
            lightening_gradient_ceiling_enabled=False,
            darkening_gradient_ceiling_enabled=False,
            instance_masks_enabled=False,
            seed_edge_curves_enabled=False,
        )

        def calculate(reference_source, dirty_nodes=frozenset()):
            return build_analysis_layers(
                image,
                valid,
                np.empty((0, 2), np.float32),
                np.empty((0,), np.float32),
                20.0,
                offset_x=0,
                offset_y=0,
                foreground_reference_source_mask=reference_source,
                dirty_nodes=dirty_nodes,
                **common,
            )

        first = calculate(first_source)
        reused = calculate(first_source)
        self.assertIs(
            reused.foreground_noise_likelihood,
            first.foreground_noise_likelihood,
        )
        self.assertIs(
            reused.reference_seed_surface_probability,
            first.reference_seed_surface_probability,
        )
        self.assertIs(reused.reference_texture_profile, first.reference_texture_profile)

        unrelated = calculate(first_source, {"directed_edges"})
        self.assertIs(
            unrelated.foreground_noise_likelihood,
            first.foreground_noise_likelihood,
        )
        self.assertIs(
            unrelated.reference_seed_surface_probability,
            first.reference_seed_surface_probability,
        )

        changed = calculate(second_source, {"foreground_segmentation"})
        self.assertIsNot(
            changed.foreground_noise_likelihood,
            first.foreground_noise_likelihood,
        )
        self.assertIsNot(
            changed.reference_seed_surface_probability,
            first.reference_seed_surface_probability,
        )
        self.assertIsNot(
            changed.reference_texture_profile,
            first.reference_texture_profile,
        )

        combined_noise_changed = calculate(
            second_source, {"refined_background_likelihood"}
        )
        self.assertIsNot(
            combined_noise_changed.foreground_noise_likelihood,
            changed.foreground_noise_likelihood,
        )
        self.assertIsNot(
            combined_noise_changed.refined_background_likelihood,
            changed.refined_background_likelihood,
        )
        self.assertIs(
            changed.foreground_reference_source_mask,
            second_source,
        )
        self.assertEqual(
            dict(changed.reference_texture_profile.class_sample_counts)[
                "foreground"
            ],
            int(painted_foreground.sum())
            + int(np.count_nonzero(source_values_2)),
        )

    def test_edge_prototype_bank_honours_capacity_above_256(self) -> None:
        import torch

        from seedvision.cuda.layers import _fit_feature_prototype_bank

        prototype_count = 300
        values = torch.arange(
            prototype_count, dtype=torch.float32
        ).repeat_interleave(4)
        features = torch.stack(
            (
                values / float(prototype_count),
                torch.sin(values),
                torch.cos(values),
            ),
            dim=0,
        ).reshape(1, 3, 30, 40)
        mask = torch.ones((1, 1, 30, 40), dtype=torch.bool)

        bank = _fit_feature_prototype_bank(
            features,
            mask,
            class_name="physical_edge",
            maximum_prototypes=prototype_count,
            minimum_support=4,
            iterations=1,
            scale_floor=0.01,
        )

        self.assertIsNotNone(bank)
        self.assertEqual(int(bank.centres.shape[0]), prototype_count)

    def test_prototype_capacities_and_class_calibrations_participate_in_cache_identity(self) -> None:
        from unittest.mock import patch

        import numpy as np

        from seedvision.cuda.layers import reference_texture_probabilities
        from seedvision.visualization import (
            AnalysisLayerSettings,
            build_analysis_layers,
        )

        image = np.full((48, 48, 3), (100, 135, 170), np.uint8)
        valid = np.full((48, 48), 255, np.uint8)
        cache: dict[str, object] = {}

        def calculate(settings):
            return build_analysis_layers(
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

        with patch(
            "seedvision.cuda.layers.reference_texture_probabilities",
            wraps=reference_texture_probabilities,
        ) as fit:
            calculate(AnalysisLayerSettings())
            self.assertEqual(fit.call_count, 1)
            calculate(AnalysisLayerSettings())
            self.assertEqual(fit.call_count, 1)
            calculate(
                AnalysisLayerSettings(
                    reference_texture_material_prototypes_per_class=72
                )
            )
            self.assertEqual(fit.call_count, 2)
            calculate(
                AnalysisLayerSettings(
                    reference_texture_material_prototypes_per_class=72,
                    reference_texture_edge_prototypes_per_class=320,
                )
            )
            self.assertEqual(fit.call_count, 3)
            calculate(
                AnalysisLayerSettings(
                    reference_texture_material_prototypes_per_class=72,
                    reference_texture_edge_prototypes_per_class=320,
                    reference_texture_class_contrast=5.0,
                )
            )
            self.assertEqual(fit.call_count, 4)
            calculate(
                AnalysisLayerSettings(
                    reference_texture_material_prototypes_per_class=72,
                    reference_texture_edge_prototypes_per_class=320,
                    reference_texture_class_contrast=5.0,
                    reference_edge_class_contrast=5.0,
                )
            )
            self.assertEqual(fit.call_count, 5)

    def test_seed_trait_changes_reuse_existing_material_and_edge_models(self) -> None:
        from unittest.mock import patch

        import cv2
        import numpy as np

        from seedvision.cuda.layers import (
            reference_seed_trait_probabilities,
            reference_texture_probabilities,
        )
        from seedvision.persistence import SeedInstanceAnnotation
        from seedvision.visualization import (
            AnalysisLayerSettings,
            build_analysis_layers,
        )

        image = np.full((48, 48, 3), (145, 175, 205), np.uint8)
        valid = np.full((48, 48), 255, np.uint8)
        labels = np.zeros((48, 48), np.uint16)
        cv2.circle(labels, (15, 24), 10, 1, -1)
        cv2.circle(labels, (33, 24), 10, 2, -1)
        foreground = np.uint8(labels > 0) * 230
        settings = AnalysisLayerSettings(
            reference_texture_material_prototypes_per_class=8,
            reference_texture_minimum_samples_per_prototype=4,
            reference_texture_fit_iterations=1,
            reference_seed_trait_prototypes_per_class=8,
            reference_seed_trait_minimum_samples_per_prototype=4,
            reference_seed_trait_fit_iterations=1,
            reference_seed_trait_working_maximum_dimension=256,
        )
        cache: dict[str, object] = {}
        common = dict(
            offset_x=0,
            offset_y=0,
            seed_instance_annotations=labels,
            seed_trait_species="lupinus_mutabilis",
            seed_trait_coat_patterns=("white", "banded_dark"),
            seed_trait_conditions=("split",),
            foreground_probability=foreground,
            settings=settings,
            cache_values=cache,
            background_noise_enabled=False,
            foreground_noise_enabled=False,
            reference_edge_probability_enabled=False,
            reference_edge_ridges_enabled=False,
            surface_darkness_gradients_enabled=False,
            lightening_gradient_ceiling_enabled=False,
            darkening_gradient_ceiling_enabled=False,
            instance_masks_enabled=False,
            seed_edge_curves_enabled=False,
        )

        def calculate(traits, dirty=frozenset()):
            return build_analysis_layers(
                image,
                valid,
                np.empty((0, 2), np.float32),
                np.empty((0,), np.float32),
                20.0,
                seed_instance_traits=traits,
                dirty_nodes=dirty,
                **common,
            )

        first_traits = (
            SeedInstanceAnnotation(1, "white"),
            SeedInstanceAnnotation(2, "banded_dark"),
        )
        revised_traits = (
            SeedInstanceAnnotation(1, "banded_dark"),
            SeedInstanceAnnotation(2, "white"),
        )
        with (
            patch(
                "seedvision.cuda.layers.reference_texture_probabilities",
                wraps=reference_texture_probabilities,
            ) as texture_fit,
            patch(
                "seedvision.cuda.layers.reference_seed_trait_probabilities",
                wraps=reference_seed_trait_probabilities,
            ) as trait_fit,
        ):
            first = calculate(first_traits)
            self.assertEqual(texture_fit.call_count, 1)
            self.assertEqual(trait_fit.call_count, 1)
            reused = calculate(first_traits)
            self.assertEqual(texture_fit.call_count, 1)
            self.assertEqual(trait_fit.call_count, 1)
            revised = calculate(revised_traits, {"reference_seed_traits"})
            self.assertEqual(texture_fit.call_count, 1)
            self.assertEqual(trait_fit.call_count, 2)

        self.assertIs(
            reused.reference_texture_profile, first.reference_texture_profile
        )
        self.assertIs(
            revised.reference_texture_profile, first.reference_texture_profile
        )
        self.assertIsNot(
            revised.reference_seed_trait_profile,
            first.reference_seed_trait_profile,
        )

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

    def test_instance_edge_probabilities_use_sparse_classes_without_forcing(self) -> None:
        import cv2
        import numpy as np

        from seedvision.visualization import build_analysis_layers

        image = np.full((96, 96, 3), 220, dtype=np.uint8)
        cv2.circle(image, (48, 48), 25, (65, 105, 165), -1)
        cv2.line(image, (48, 28), (48, 68), (25, 55, 95), 3)
        valid = np.full((96, 96), 255, dtype=np.uint8)
        instances = np.zeros((96, 96), dtype=np.uint16)
        cv2.circle(instances, (48, 48), 25, 1, -1)
        layers = build_analysis_layers(
            image,
            valid,
            np.empty((0, 2), np.float32),
            np.empty((0,), np.float32),
            50.0,
            offset_x=0,
            offset_y=0,
            seed_instance_annotations=instances,
            instance_masks_enabled=False,
            seed_edge_curves_enabled=False,
        )

        physical_probability = np.asarray(layers.physical_edge_probability)
        non_edge_probability = np.asarray(layers.non_edge_probability)
        net_prototype_compatibility = np.asarray(
            layers.net_physical_edge_probability
        )
        reference_edge_probability = np.asarray(
            layers.reference_edge_probability
        )
        conservative_evidence = np.asarray(
            layers.conservative_net_physical_edge_evidence
        )
        true_edge_support = np.asarray(layers.edge_likelihood)
        generic_ridges = np.asarray(layers.edge_ridges)
        # Reference classification must retain real gradients discarded by
        # generic NMS; reference-specific thinning happens only afterwards.
        self.assertTrue(np.any(
            (generic_ridges == 0) & (reference_edge_probability > 1)
        ))
        self.assertFalse(
            layers.physical_edge_interior_direction_x.is_materialized
        )
        self.assertFalse(
            layers.physical_edge_interior_direction_y.is_materialized
        )
        self.assertFalse(
            layers.physical_edge_interior_direction_confidence.is_materialized
        )
        interior_direction_x = np.asarray(
            layers.physical_edge_interior_direction_x
        )
        interior_direction_y = np.asarray(
            layers.physical_edge_interior_direction_y
        )
        interior_direction_confidence = np.asarray(
            layers.physical_edge_interior_direction_confidence
        )
        self.assertFalse(layers.reference_edge_ridges.is_materialized)
        self.assertFalse(layers.net_reference_edge_ridges.is_materialized)
        self.assertFalse(
            layers.locally_normalized_net_physical_edge.is_materialized
        )
        self.assertFalse(
            layers.normalized_net_reference_edge_ridges.is_materialized
        )
        reference_ridges = np.asarray(layers.reference_edge_ridges)
        net_reference_ridges = np.asarray(layers.net_reference_edge_ridges)
        normalized_net = np.asarray(
            layers.locally_normalized_net_physical_edge
        )
        normalized_net_ridges = np.asarray(
            layers.normalized_net_reference_edge_ridges
        )
        self.assertEqual(physical_probability.shape, image.shape[:2])
        self.assertEqual(non_edge_probability.shape, image.shape[:2])
        self.assertEqual(reference_edge_probability.shape, image.shape[:2])
        expected_supported = np.rint(
            physical_probability.astype(np.float32)
            * true_edge_support.astype(np.float32)
            / 255.0
        ).astype(np.uint8)
        np.testing.assert_allclose(
            reference_edge_probability,
            expected_supported,
            atol=1,
        )
        expected_conservative = np.rint(
            net_prototype_compatibility.astype(np.float32)
            * true_edge_support.astype(np.float32) / 255.0
        )
        np.testing.assert_allclose(conservative_evidence, expected_conservative, atol=1)
        self.assertTrue(np.all(conservative_evidence <= reference_edge_probability))
        for supported in (
            reference_edge_probability, conservative_evidence, reference_ridges,
            net_reference_ridges, normalized_net, normalized_net_ridges,
        ):
            # The displayed gradient is quantized; zero there can still mean
            # a tiny positive float gradient (and normalization can boost it).
            self.assertTrue(np.all(supported[0:8, 0:8] == 0))
        self.assertEqual(interior_direction_x.shape, image.shape[:2])
        self.assertEqual(interior_direction_y.shape, image.shape[:2])
        self.assertEqual(
            interior_direction_confidence.shape, image.shape[:2]
        )
        self.assertGreater(float(interior_direction_confidence.max()), 0.0)
        confident = interior_direction_confidence > 1e-4
        self.assertTrue(np.any(confident))
        np.testing.assert_allclose(
            np.hypot(
                interior_direction_x[confident],
                interior_direction_y[confident],
            ),
            1.0,
            atol=1e-4,
        )
        self.assertEqual(reference_ridges.shape, image.shape[:2])
        self.assertEqual(normalized_net.shape, image.shape[:2])
        self.assertEqual(normalized_net_ridges.shape, image.shape[:2])
        self.assertFalse(np.array_equal(physical_probability, non_edge_probability))
        self.assertGreater(int(physical_probability.max()), 0)
        self.assertGreater(int(non_edge_probability.max()), 0)
        self.assertGreater(int(reference_ridges.max()), 0)
        self.assertGreater(int(normalized_net.max()), 1)
        self.assertGreaterEqual(int(net_reference_ridges.max()), 0)
        self.assertLess(
            int(np.count_nonzero(reference_ridges)),
            int(np.count_nonzero(physical_probability)),
        )
        self.assertLessEqual(
            int(np.count_nonzero(net_reference_ridges)),
            int(np.count_nonzero(reference_ridges)),
        )
        # Annotation coordinates are samples, not hard output assignments.
        self.assertLess(int(physical_probability[48, 24]), 255)
        self.assertLess(int(non_edge_probability[48, 48]), 255)
        self.assertEqual(
            layers.reference_edge_prototype_compatibility_rgba(True).shape,
            (96, 96, 4),
        )
        self.assertEqual(
            layers.reference_edge_supported_probability_rgba().shape,
            (96, 96, 4),
        )
        self.assertEqual(
            layers.physical_edge_interior_direction_rgba().shape,
            (96, 96, 4),
        )
        self.assertEqual(layers.reference_edge_ridges_rgba().shape, (96, 96, 4))
        self.assertEqual(
            layers.net_reference_edge_ridges_rgba().shape, (96, 96, 4)
        )
        self.assertEqual(
            layers.locally_normalized_net_physical_edge_rgba().shape,
            (96, 96, 4),
        )
        self.assertEqual(
            layers.normalized_net_reference_edge_ridges_rgba().shape,
            (96, 96, 4),
        )

        disabled = build_analysis_layers(
            image,
            valid,
            np.empty((0, 2), np.float32),
            np.empty((0,), np.float32),
            50.0,
            offset_x=0,
            offset_y=0,
            seed_instance_annotations=instances,
            reference_edge_ridges_enabled=False,
            instance_masks_enabled=False,
            seed_edge_curves_enabled=False,
        )
        self.assertEqual(int(np.asarray(disabled.reference_edge_ridges).max()), 0)

    def test_reference_probability_and_normalization_ignore_conservative_weight(self) -> None:
        import cv2
        import numpy as np

        from seedvision.visualization import AnalysisLayerSettings, build_analysis_layers

        image = np.full((96, 96, 3), 220, dtype=np.uint8)
        cv2.circle(image, (48, 48), 25, (65, 105, 165), -1)
        cv2.line(image, (48, 28), (48, 68), (25, 55, 95), 3)
        valid = np.full((96, 96), 255, dtype=np.uint8)
        instances = np.zeros((96, 96), dtype=np.uint16)
        cv2.circle(instances, (48, 48), 25, 1, -1)
        cache = {}

        def calculate(weight, dirty=()):
            # Keep genuine class overlap so subtraction cannot trivially be a
            # no-op on an already perfectly exclusive synthetic classifier.
            return build_analysis_layers(
                image, valid, np.empty((0, 2), np.float32),
                np.empty((0,), np.float32), 50.0,
                offset_x=0, offset_y=0, seed_instance_annotations=instances,
                instance_masks_enabled=False, seed_edge_curves_enabled=False,
                cache_values=cache, dirty_nodes=dirty,
                settings=AnalysisLayerSettings(
                    net_physical_edge_internal_scale=weight,
                    reference_edge_class_contrast=1.0,
                    reference_edge_similarity_scale=4.0,
                ),
            )

        initial = calculate(0.0)
        self.assertIs(
            initial.reference_edge_probability,
            initial.edge_supported_physical_compatibility,
        )
        np.testing.assert_array_equal(
            initial.reference_edge_probability,
            initial.conservative_net_physical_edge_evidence,
        )
        np.testing.assert_array_equal(
            initial.reference_edge_ridges, initial.net_reference_edge_ridges
        )
        gradients = cache["layer.edge_gradients"]
        support = gradients.strength.detach().cpu().numpy()[0, 0]
        physical = np.asarray(initial.physical_edge_probability).astype(np.float32)
        nonphysical = np.asarray(initial.non_edge_probability).astype(np.float32)
        self.assertGreater(float(np.sum(support * nonphysical)), 0.0)
        for weight in (0.5, 1.25, 2.0):
            changed = calculate(weight, {"reference_edge_probability"})
            self.assertIs(cache["layer.edge_gradients"], gradients)
            self.assertIs(changed.physical_edge_probability, initial.physical_edge_probability)
            self.assertIs(changed.non_edge_probability, initial.non_edge_probability)
            for attribute in (
                "reference_edge_probability", "reference_edge_ridges",
                "locally_normalized_net_physical_edge",
                "normalized_net_reference_edge_ridges",
            ):
                np.testing.assert_array_equal(
                    getattr(changed, attribute), getattr(initial, attribute),
                    err_msg=f"{attribute} changed with conservative weight={weight}",
                )
            conservative = np.asarray(changed.conservative_net_physical_edge_evidence)
            expected = support * np.maximum(physical - weight * nonphysical, 0.0)
            np.testing.assert_allclose(conservative, np.rint(expected), atol=1)
            self.assertTrue(np.all(conservative[support == 0] == 0))
            self.assertTrue(
                np.any(conservative < np.asarray(initial.reference_edge_probability)),
                f"Conservative weight {weight} should affect this overlapping-class fixture",
            )

    def test_local_physical_edge_normalization_equalizes_support_without_promoting_noise(self) -> None:
        from types import SimpleNamespace

        import numpy as np
        import torch

        from seedvision.cuda import CudaContext, GpuRaster
        from seedvision.cuda.layers import (
            locally_normalized_reference_edge_probability,
        )
        from seedvision.visualization import AnalysisLayerSettings

        height, width = 48, 128
        physical = np.zeros((height, width), np.float32)
        physical[:, :56] = 0.90
        physical[:, 72:] = 0.90
        physical[16:32, 58:70] = 0.20
        physical[0:8, 58:70] = 0.90
        edge_strength = np.zeros_like(physical)
        edge_strength[:, :56] = 0.10
        edge_strength[:, 72:] = 0.80
        edge_strength[16:32, 58:70] = 0.40
        edge_strength[0:8, 58:70] = 0.01
        context = CudaContext.resolve(requested="cpu")

        def raster(values, name):
            return GpuRaster(
                torch.from_numpy(values)[None, None],
                numpy_dtype=np.float32,
                name=name,
            )

        gradients = SimpleNamespace(
            strength=torch.from_numpy(edge_strength)[None, None],
            valid=torch.ones((1, 1, height, width), dtype=torch.bool),
        )
        support_u8 = np.rint(edge_strength * 255.0).astype(np.uint8)
        normalized = locally_normalized_reference_edge_probability(
            raster(physical, "test physical prototype probability"),
            GpuRaster(
                torch.from_numpy(support_u8)[None, None],
                numpy_dtype=np.uint8,
                name="test thinned true-edge support",
            ),
            gradients,
            20.0,
            AnalysisLayerSettings(
                reference_ridge_working_maximum_dimension=256,
                reference_edge_normalization_radius_fraction=0.30,
                reference_edge_normalization_target_support=0.35,
                reference_edge_normalization_maximum_gain=2.50,
                reference_edge_normalization_absolute_floor=0.04,
            ),
            cuda_context=context,
        )
        values = np.asarray(normalized)
        support = support_u8.astype(np.float32) / 255.0
        raw_weak = float(support[24, 24] * physical[24, 24])
        raw_strong = float(support[24, 104] * physical[24, 104])
        normalized_weak = float(values[24, 24])
        normalized_strong = float(values[24, 104])
        self.assertGreater(normalized_weak, raw_weak)
        self.assertGreaterEqual(normalized_strong, raw_strong - 1e-6)
        self.assertLess(
            normalized_strong / normalized_weak,
            raw_strong / raw_weak,
        )
        # The 2.5x control bounds enhancement, while supported strong edges are
        # never attenuated below the Physical probability × support product.
        self.assertLessEqual(
            normalized_weak,
            support[24, 24] * 2.50 * physical[24, 24] + 1e-6,
        )
        self.assertGreaterEqual(normalized_strong, raw_strong - 1e-6)
        self.assertGreater(float(values[20, 64]), 0.0)
        self.assertLessEqual(float(values[20, 64]), float(physical[20, 64]))
        self.assertLess(float(values[20, 64]), normalized_weak)
        self.assertLess(float(values[3, 64]), 0.01 * 0.90)

    def test_reference_edge_comparison_and_net_physical_overlays(self) -> None:
        from dataclasses import replace

        import numpy as np

        from seedvision.visualization import AnalysisLayers, NoiseFrequencyProfile

        physical = np.asarray(
            ((0, 64, 200), (255, 10, 99)), dtype=np.uint8
        )
        non_edge = np.asarray(
            ((0, 128, 80), (255, 30, 0)), dtype=np.uint8
        )
        valid = np.asarray(((255, 255, 255), (0, 255, 255)), dtype=np.uint8)
        zeros = np.zeros_like(valid)
        layers = AnalysisLayers(
            offset_x=0,
            offset_y=0,
            instance_labels=zeros,
            instance_colours=np.zeros((1, 3), dtype=np.uint8),
            background_likelihood=zeros,
            refined_background_likelihood=zeros,
            noise_frequency_profile=NoiseFrequencyProfile(
                band_scales_px=(1.0, 2.0, 3.0),
                descriptor_centre=(0.0, 0.0, 0.0),
                target_sample_count=0,
            ),
            edge_likelihood=zeros,
            directed_edge_hue=zeros,
            undirected_edge_hue=zeros,
            seed_edge_curve_likelihood=zeros,
            seed_edge_curve_radius_px=zeros,
            valid_mask=valid,
            physical_edge_probability=physical,
            non_edge_probability=non_edge,
        )

        comparison = layers.reference_edge_comparison_rgba()
        np.testing.assert_array_equal(comparison[:, :, 0], non_edge)
        np.testing.assert_array_equal(comparison[:, :, 1], zeros)
        np.testing.assert_array_equal(comparison[:, :, 2], physical)
        np.testing.assert_array_equal(comparison[:, :, 3], valid)

        direction_layers = replace(
            layers,
            physical_edge_interior_direction_x=np.asarray(
                ((1.0, 0.0, -1.0), (0.0, 0.0, 1.0)), np.float32
            ),
            physical_edge_interior_direction_y=np.asarray(
                ((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0)), np.float32
            ),
            physical_edge_interior_direction_confidence=np.asarray(
                ((1.0, 1.0, 1.0), (1.0, 0.0, 0.5)), np.float32
            ),
        )
        direction_rgba = direction_layers.physical_edge_interior_direction_rgba()
        np.testing.assert_array_equal(direction_rgba[0, 0, :3], (255, 0, 0))
        np.testing.assert_array_equal(direction_rgba[0, 2, :3], (0, 255, 255))
        np.testing.assert_array_equal(direction_rgba[1, 1, :3], (0, 0, 0))
        np.testing.assert_array_equal(direction_rgba[:, :, 3], valid)

        net = layers.net_physical_edge_probability_rgba()
        np.testing.assert_array_equal(net[:, :, 0], zeros)
        np.testing.assert_array_equal(net[:, :, 1], zeros)
        np.testing.assert_array_equal(
            net[:, :, 2],
            np.asarray(((0, 0, 160), (128, 0, 99)), dtype=np.uint8),
        )
        np.testing.assert_array_equal(net[:, :, 3], valid)

        authoritative_net = np.asarray(
            ((7, 8, 9), (10, 11, 12)), dtype=np.uint8
        )
        cached_layers = replace(
            layers,
            net_physical_edge_probability=authoritative_net,
            reference_edge_probability=np.asarray(
                ((1, 2, 3), (4, 5, 6)), dtype=np.uint8
            ),
            conservative_net_physical_edge_evidence=np.asarray(
                ((0, 1, 2), (0, 0, 3)), dtype=np.uint8
            ),
            locally_normalized_net_physical_edge=authoritative_net,
            net_physical_edge_internal_scale=1.75,
        )
        raw_net_rgba = cached_layers.net_physical_edge_probability_rgba()
        normalized_net_rgba = (
            cached_layers.locally_normalized_net_physical_edge_rgba()
        )
        np.testing.assert_array_equal(raw_net_rgba[:, :, 2], authoritative_net)
        np.testing.assert_array_equal(normalized_net_rgba, raw_net_rgba)
        np.testing.assert_array_equal(
            cached_layers.reference_edge_supported_probability_rgba()[:, :, 2],
            cached_layers.reference_edge_probability,
        )
        conservative_rgba = cached_layers.conservative_net_physical_edge_evidence_rgba()
        np.testing.assert_array_equal(conservative_rgba[:, :, 0], zeros)
        np.testing.assert_array_equal(conservative_rgba[:, :, 1], zeros)
        np.testing.assert_array_equal(
            conservative_rgba[:, :, 2], cached_layers.conservative_net_physical_edge_evidence
        )
        np.testing.assert_array_equal(conservative_rgba[:, :, 3], valid)

        stronger_subtraction = replace(
            layers, net_physical_edge_internal_scale=1.25
        )
        np.testing.assert_array_equal(
            stronger_subtraction.net_physical_edge_probability_rgba()[:, :, 2],
            np.asarray(((0, 0, 100), (0, 0, 99)), dtype=np.uint8),
        )
        np.testing.assert_array_equal(
            np.asarray(stronger_subtraction.physical_edge_probability), physical
        )
        np.testing.assert_array_equal(
            np.asarray(stronger_subtraction.non_edge_probability), non_edge
        )

        without_non_edge = replace(layers, non_edge_probability=None)
        np.testing.assert_array_equal(
            without_non_edge.reference_edge_comparison_rgba()[:, :, 0],
            zeros,
        )
        np.testing.assert_array_equal(
            without_non_edge.net_physical_edge_probability_rgba()[:, :, 2],
            physical,
        )
        without_physical = replace(layers, physical_edge_probability=None)
        np.testing.assert_array_equal(
            without_physical.reference_edge_comparison_rgba()[:, :, 2],
            zeros,
        )
        np.testing.assert_array_equal(
            without_physical.net_physical_edge_probability_rgba()[:, :, 2],
            zeros,
        )

    def test_normalized_ridge_extends_connected_weak_edges_without_weakening_seeds(self) -> None:
        from types import SimpleNamespace

        import numpy as np
        import torch

        from seedvision.cuda import CudaContext, GpuRaster
        from seedvision.cuda.layers import (
            locally_normalized_reference_edge_probability,
            reference_probability_ridges,
        )
        from seedvision.visualization import AnalysisLayerSettings

        height, width = 48, 65
        x = torch.arange(width, dtype=torch.float32)
        cross_section = torch.exp(-0.5 * ((x - 32.0) / 2.0).square())
        amplitude = torch.zeros(height, dtype=torch.float32)
        amplitude[4:12] = 0.80
        amplitude[12:36] = 0.08
        amplitude[40:44] = 0.08
        strength = amplitude[:, None] * cross_section[None, :]
        physical = torch.full((1, 1, height, width), 0.90)
        internal = torch.zeros_like(physical)
        gradients = SimpleNamespace(
            strength=strength[None, None],
            normal_x=torch.ones((1, 1, height, width)),
            normal_y=torch.zeros((1, 1, height, width)),
            valid=torch.ones((1, 1, height, width), dtype=torch.bool),
        )
        settings = AnalysisLayerSettings(
            reference_ridge_working_maximum_dimension=256,
            reference_ridge_low_threshold=0.10,
            reference_ridge_high_threshold=0.40,
            reference_ridge_hysteresis_iterations=32,
            reference_edge_normalization_radius_fraction=0.30,
            reference_edge_normalization_target_support=0.35,
            reference_edge_normalization_maximum_gain=2.50,
            reference_edge_normalization_absolute_floor=0.04,
        )
        context = CudaContext.resolve(requested="cpu")
        margin_raster = GpuRaster(
            physical,
            numpy_dtype=np.float32,
            name="net prototype compatibility",
        )
        thinned_support = torch.zeros_like(strength)
        thinned_support[:, 32] = amplitude
        support_raster = GpuRaster(
            torch.round(thinned_support * 255.0).to(torch.uint8)[None, None],
            numpy_dtype=np.uint8,
            name="thinned true-edge support",
        )
        normalized = locally_normalized_reference_edge_probability(
            margin_raster,
            support_raster,
            gradients,
            20.0,
            settings,
            cuda_context=context,
        )
        normalized_values = np.asarray(normalized)
        raw_ridge = np.asarray(
            reference_probability_ridges(
                GpuRaster(
                    physical * thinned_support,
                    numpy_dtype=np.float32,
                    name="edge-supported reference probability",
                ),
                gradients,
                settings,
                include_gradient_strength=False,
                restrict_to_source_support=True,
                cuda_context=context,
            )
        )
        normalized_ridge = np.asarray(
            reference_probability_ridges(
                normalized,
                gradients,
                settings,
                include_gradient_strength=False,
                restrict_to_source_support=True,
                cuda_context=context,
            )
        )

        self.assertGreater(int(raw_ridge[8, 32]), 0)
        self.assertGreaterEqual(
            int(normalized_ridge[8, 32]), int(raw_ridge[8, 32])
        )
        self.assertEqual(int(raw_ridge[30, 32]), 0)
        self.assertGreater(
            int(normalized_ridge[30, 32]),
            0,
            msg=(
                f"normalized weak support={normalized_values[30, 32]:.4f}; "
                f"centreline={normalized_ridge[:, 32].tolist()}"
            ),
        )
        self.assertEqual(int(normalized_ridge[42, 32]), 0)
        self.assertGreater(
            int(np.count_nonzero(normalized_ridge)),
            int(np.count_nonzero(raw_ridge)),
        )

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
        instances = np.zeros((height, width), np.uint16)
        for instance_id, (centre_x, centre_y, radius, _colour) in enumerate(
            seeds, start=1
        ):
            cv2.circle(
                instances,
                (centre_x, centre_y),
                radius,
                instance_id,
                -1,
            )

        settings = AnalysisLayerSettings(
            reference_texture_material_prototypes_per_class=16,
            reference_texture_edge_prototypes_per_class=8,
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
            seed_instance_annotations=instances,
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
                "physical_edge": 8,
                "non_edge": 8,
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
        material_sum = (
            seed_surface.astype(np.uint16)
            + background_probability.astype(np.uint16)
            + other_probability.astype(np.uint16)
        )
        self.assertLessEqual(int(material_sum.max()), 255)
        self.assertLess(float(material_sum[90:110, 5:25].mean()), 255.0)
        self.assertGreater(float(seed_surface[foreground].mean()), 180.0)
        self.assertGreater(float(background_probability[background].mean()), 220.0)
        self.assertGreater(float(other_probability[other].mean()), 210.0)
        self.assertLess(float(seed_surface[background].mean()), 10.0)
        self.assertLess(float(seed_surface[other].mean()), 10.0)
        # Reference pixels are classifier examples, never hard-assigned output.
        self.assertLess(int(seed_surface[foreground].max()), 255)

    def test_one_class_material_prototypes_publish_no_probability(self) -> None:
        import numpy as np

        from seedvision.visualization import AnalysisLayerSettings, build_analysis_layers

        image = np.full((96, 96, 3), (90, 130, 190), np.uint8)
        foreground = np.zeros((96, 96), dtype=bool)
        foreground[24:72, 24:72] = True
        settings = AnalysisLayerSettings(
            background_keep_perimeter_reference=False,
            reference_texture_material_prototypes_per_class=16,
            reference_texture_minimum_samples_per_prototype=8,
            reference_texture_fit_iterations=2,
            reference_texture_working_maximum_dimension=256,
        )
        layers = build_analysis_layers(
            image,
            np.full((96, 96), 255, np.uint8),
            np.empty((0, 2), np.float32),
            np.empty((0,), np.float32),
            32.0,
            offset_x=0,
            offset_y=0,
            settings=settings,
            foreground_reference_mask=foreground,
            instance_masks_enabled=False,
            seed_edge_curves_enabled=False,
        )
        self.assertGreater(
            layers.reference_texture_profile.count_for("foreground"), 0
        )
        self.assertEqual(
            layers.reference_texture_profile.count_for("background"), 0
        )
        self.assertIsNone(layers.reference_seed_surface_probability)
        self.assertIsNone(layers.reference_background_texture_probability)
        self.assertIsNone(layers.reference_other_texture_probability)

    def test_material_prototype_calibration_separates_class_contrast_from_unknown(self) -> None:
        import torch

        from seedvision.cuda.layers import _prototype_class_probabilities

        # These are kernel similarities, not probabilities. The first two
        # pixels are intentionally identical to demonstrate that the transform
        # has no access to painted masks or coordinates. The third is a genuine
        # Background/Other tie, and the fourth is uniformly unsupported.
        scores = {
            "background": torch.tensor([[[[0.18, 0.18, 0.65, 0.02]]]]),
            "foreground": torch.tensor([[[[0.32, 0.32, 0.10, 0.02]]]]),
            "other": torch.tensor([[[[0.65, 0.65, 0.65, 0.02]]]]),
        }
        probabilities = _prototype_class_probabilities(
            scores,
            torch.ones((1, 1, 1, 4)),
            class_contrast=4.0,
        )

        self.assertGreater(float(probabilities["other"][0, 0, 0, 0]), 0.80)
        for probability in probabilities.values():
            self.assertAlmostEqual(
                float(probability[0, 0, 0, 0]),
                float(probability[0, 0, 0, 1]),
                places=7,
            )
        self.assertAlmostEqual(
            float(probabilities["background"][0, 0, 0, 2]),
            float(probabilities["other"][0, 0, 0, 2]),
            places=7,
        )
        known_mass = sum(probability for probability in probabilities.values())
        self.assertLess(float(known_mass[0, 0, 0, 3]), 0.25)

        unsharpened = _prototype_class_probabilities(
            scores,
            torch.ones((1, 1, 1, 4)),
            class_contrast=1.0,
        )
        self.assertLess(
            float(unsharpened["other"][0, 0, 0, 0]),
            float(probabilities["other"][0, 0, 0, 0]),
        )

    def test_edge_prototype_calibration_separates_class_contrast_from_unknown(self) -> None:
        import torch

        from seedvision.cuda.layers import _prototype_class_probabilities

        # The first two coordinates deliberately have identical score vectors:
        # the transform cannot see whether either was an annotation coordinate.
        scores = {
            "physical_edge": torch.tensor([[[[0.65, 0.65, 0.65, 0.02]]]]),
            "non_edge": torch.tensor([[[[0.20, 0.20, 0.65, 0.02]]]]),
        }
        probabilities = _prototype_class_probabilities(
            scores,
            torch.ones((1, 1, 1, 4)),
            class_contrast=4.0,
            unknown_mass=0.10,
        )

        old_physical = 0.65 / (0.65 + 0.20 + 0.10)
        self.assertGreater(
            float(probabilities["physical_edge"][0, 0, 0, 0]),
            old_physical + 0.15,
        )
        for probability in probabilities.values():
            self.assertAlmostEqual(
                float(probability[0, 0, 0, 0]),
                float(probability[0, 0, 0, 1]),
                places=7,
            )
        self.assertAlmostEqual(
            float(probabilities["physical_edge"][0, 0, 0, 2]),
            float(probabilities["non_edge"][0, 0, 0, 2]),
            places=7,
        )
        known_mass = sum(probabilities.values())
        self.assertLess(float(known_mass[0, 0, 0, 3]), 0.20)

        unsharpened = _prototype_class_probabilities(
            scores,
            torch.ones((1, 1, 1, 4)),
            class_contrast=1.0,
            unknown_mass=0.10,
        )
        self.assertLess(
            float(unsharpened["physical_edge"][0, 0, 0, 0]),
            float(probabilities["physical_edge"][0, 0, 0, 0]),
        )

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
        self.assertGreater(profile.target_sample_count, 128)
        self.assertGreaterEqual(profile.compatibility_half_distance, 1.0)
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
        foreground_reference = np.zeros((96, 96), dtype=bool)
        foreground_reference[34:46, 34:46] = True
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
            foreground_reference_mask=foreground_reference,
            instance_masks_enabled=False,
            seed_edge_curves_enabled=False,
        )

        likelihood = np.asarray(layers.foreground_noise_likelihood)
        self.assertGreater(
            float(np.mean(likelihood[34:62, 34:62])),
            float(np.mean(likelihood[4:20, 4:20])) + 80.0,
        )
        self.assertGreaterEqual(
            layers.foreground_noise_frequency_profile.compatibility_half_distance,
            1.0,
        )
        self.assertEqual(layers.foreground_noise_rgba().shape, (96, 96, 4))

    def test_first_tertile_is_the_exact_directional_one_third_quantile(self) -> None:
        import torch

        from seedvision.cuda.layers import _integrate_directional_noise

        stack = torch.tensor(
            (0.0, 10.0, 20.0, 30.0, 40.0), dtype=torch.float32
        )[:, None, None]
        first_tertile = _integrate_directional_noise(stack, "1st tertile")

        self.assertAlmostEqual(float(first_tertile.item()), 40.0 / 3.0, places=5)
        self.assertLess(
            float(_integrate_directional_noise(stack, "minimum").item()),
            float(first_tertile.item()),
        )
        self.assertLess(
            float(first_tertile.item()),
            float(_integrate_directional_noise(stack, "median").item()),
        )
        self.assertLess(
            float(_integrate_directional_noise(stack, "median").item()),
            float(_integrate_directional_noise(stack, "maximum").item()),
        )

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

        self.assertEqual(background_profile.target_sample_count, 256)
        self.assertEqual(foreground_profile.target_sample_count, 256)
        self.assertLess(
            sum(background_profile.descriptor_centre),
            sum(foreground_profile.descriptor_centre),
        )
        # References train the probability distribution; their coordinates are
        # not overwritten to artificial exact-zero or exact-one outputs.
        self.assertTrue(
            np.all(np.asarray(background_result[0])[background] < 255)
        )
        self.assertTrue(
            np.all(np.asarray(foreground_result[0])[foreground] < 255)
        )

    def test_directly_supervised_noise_probability_is_independent_of_colour_map(self) -> None:
        import numpy as np

        from seedvision.cuda import CudaContext
        from seedvision.cuda.layers import noise_frequency_background_likelihood
        from seedvision.visualization import AnalysisLayerSettings

        rng = np.random.default_rng(29)
        image = np.full((72, 72, 3), 214, np.uint8)
        image[36:, 36:] = np.clip(
            125 + rng.integers(-32, 33, (36, 36, 1)), 0, 255
        ).astype(np.uint8)
        valid = np.full((72, 72), 255, np.uint8)
        target = np.zeros((72, 72), bool)
        target[8:28, 8:28] = True
        nontarget = np.zeros((72, 72), bool)
        nontarget[42:62, 42:62] = True
        settings = AnalysisLayerSettings(
            noise_vector_length_fraction=0.20,
            noise_vector_sample_count=5,
        )
        context = CudaContext.resolve(requested="cpu")

        def calculate(colour):
            return noise_frequency_background_likelihood(
                image,
                valid,
                colour,
                28.0,
                settings,
                background_reference_mask=target,
                foreground_reference_mask=nontarget,
                cuda_context=context,
            )[0]

        low = calculate(np.zeros((72, 72), np.uint8))
        high = calculate(np.full((72, 72), 255, np.uint8))
        np.testing.assert_array_equal(np.asarray(low), np.asarray(high))

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

    def test_painted_background_fits_multimodal_distribution_without_refinement(self) -> None:
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
        # The retired self-refinement controls remain accepted only for legacy
        # constructor compatibility; they must not alter the fitted evidence.
        self.assertEqual(profile.refinement_iterations, 0)
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
        self.assertGreater(int(layers.refined_background_likelihood.max()), 0)
        self.assertGreater(layers.noise_frequency_profile.target_sample_count, 0)
        self.assertGreater(np.count_nonzero(layers.instance_labels == 1), 0)
        self.assertGreater(int(layers.edge_likelihood.max()), 0)

        noise_disabled = build_analysis_layers(
            image,
            valid,
            np.asarray(((48, 48),), dtype=np.float32),
            np.asarray((16,), dtype=np.float32),
            32.0,
            offset_x=0,
            offset_y=0,
            background_noise_enabled=False,
        )
        self.assertNotEqual(noise_disabled.background_mode, "disabled")
        self.assertGreater(int(noise_disabled.background_likelihood.max()), 0)
        self.assertFalse(np.any(noise_disabled.refined_background_likelihood))

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

    def test_other_evidence_is_independent_without_pixel_overrides(self) -> None:
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
        foreground_reference = np.zeros((96, 96), dtype=bool)
        foreground_reference[24:32, 24:32] = True
        baseline_layers = build_analysis_layers(
            image,
            valid,
            np.empty((0, 2), np.float32),
            np.empty((0,), np.float32),
            32.0,
            offset_x=0,
            offset_y=0,
            foreground_probability=foreground_probability,
            foreground_reference_mask=foreground_reference,
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
            foreground_reference_mask=foreground_reference,
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
        self.assertAlmostEqual(
            float(np.mean(foreground_noise[foreground_exclusion])),
            float(np.mean(foreground_noise[56:64, 56:64])),
            delta=2.0,
        )
        self.assertTrue(
            layers.background_colour_profile.excluded_component_centres_lab
        )

        baseline_foreground = _foreground_feature(
            image,
            BaselineSettings(),
            CudaContext.resolve(requested="cpu"),
            foreground_reference_mask=foreground_reference,
            seed_diameter=32.0,
        )
        foreground_result = _foreground_feature(
            image,
            BaselineSettings(),
            CudaContext.resolve(requested="cpu"),
            foreground_reference_mask=foreground_reference,
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
            layers.net_physical_edge_probability,
            layers.reference_edge_probability,
            layers.conservative_net_physical_edge_evidence,
            layers.edge_supported_physical_compatibility,
            layers.edge_supported_nonphysical_compatibility,
            layers.reference_edge_ridges,
            layers.net_reference_edge_ridges,
            layers.locally_normalized_net_physical_edge,
            layers.normalized_net_reference_edge_ridges,
            layers.edge_trace_labels,
            layers.edge_trace_continuity,
            layers.edge_trace_gap_confidence,
            layers.edge_radius_confidence,
            layers.edge_circle_confidence,
            layers.edge_ellipse_confidence,
            layers.edge_fit_residual,
            layers.edge_centre_votes,
            layers.oval_centre_probability,
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
        self.assertFalse(layers.oval_centre_probability.is_materialized)

    def test_curved_edges_produce_proposal_independent_oval_centres(self) -> None:
        import cv2
        import numpy as np

        from seedvision.visualization import build_analysis_layers

        image = np.full((128, 128, 3), 228, dtype=np.uint8)
        cv2.ellipse(
            image,
            (64, 64),
            (27, 20),
            18,
            0,
            360,
            (35, 78, 165),
            -1,
        )
        valid = np.full((128, 128), 255, dtype=np.uint8)
        layers = build_analysis_layers(
            image,
            valid,
            np.empty((0, 2), np.float32),
            np.empty((0,), np.float32),
            48.0,
            offset_x=0,
            offset_y=0,
        )

        self.assertFalse(layers.oval_centre_probability.is_materialized)
        geometry = layers.edge_fit_geometry.materialize()
        retained = geometry["oval_confidence"] > 0.0
        self.assertEqual(int(retained.sum()), 1)
        centre = geometry["oval_centres_xy"][retained][0]
        axes = geometry["oval_axes_xy"][retained][0]
        angle_degrees = float(
            np.rad2deg(geometry["oval_angle_radians"][retained][0])
        )
        axial_error = abs(((angle_degrees - 18.0 + 90.0) % 180.0) - 90.0)
        self.assertLess(float(np.linalg.norm(centre - (64.0, 64.0))), 3.0)
        self.assertTrue(24.0 <= float(axes[0]) <= 32.0)
        self.assertTrue(17.0 <= float(axes[1]) <= 24.0)
        self.assertLess(axial_error, 10.0)

        centre_probability = np.asarray(layers.oval_centre_probability)
        peak_y, peak_x = np.unravel_index(
            int(np.argmax(centre_probability)), centre_probability.shape
        )
        self.assertLess(np.hypot(peak_x - 64.0, peak_y - 64.0), 3.0)
        self.assertGreater(int(centre_probability[peak_y, peak_x]), 180)

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
        first_normalized_net = cache[
            "layer.locally_normalized_net_physical_edge"
        ]

        build_analysis_layers(
            image,
            valid,
            np.asarray(((24, 31), (88, 79)), np.float32),
            np.asarray((12, 18), np.float32),
            46.0,
            offset_x=0,
            offset_y=0,
            cache_values=cache,
            dirty_nodes={"instance_masks"},
        )
        self.assertIs(cache["layer.seed_edge_curves"], first)

        scaled_overlay = build_analysis_layers(
            image,
            valid,
            centers,
            radii,
            46.0,
            offset_x=0,
            offset_y=0,
            cache_values=cache,
            settings=AnalysisLayerSettings(
                net_physical_edge_internal_scale=1.25
            ),
        )
        self.assertEqual(scaled_overlay.net_physical_edge_internal_scale, 0.5)
        self.assertIs(
            cache["layer.reference_edge_probability"],
            first_reference_probability,
        )
        self.assertIs(cache["layer.reference_edge_ridges"], first_reference_ridge)
        self.assertIs(cache["layer.seed_edge_curves"], first)

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
            settings=AnalysisLayerSettings(
                reference_ridge_high_threshold=0.30,
                reference_edge_normalization_maximum_gain=3.0,
            ),
        )
        self.assertIs(
            cache["layer.reference_edge_probability"],
            first_reference_probability,
        )
        self.assertIsNot(
            cache["layer.reference_edge_ridges"], first_reference_ridge
        )
        self.assertIsNot(
            cache["layer.locally_normalized_net_physical_edge"],
            first_normalized_net,
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
                trace_edge_source="normalized_net_reference_ridges",
            ),
        )
        source_changed = cache["layer.seed_edge_curves"]
        self.assertIs(source_changed.ridge_state, trace_changed.ridge_state)
        self.assertIsNot(source_changed.trace_state, trace_changed.trace_state)
        self.assertEqual(
            source_changed.trace_state.signature[1],
            "normalized_net_reference_ridges",
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
            dirty_nodes={"edge_traces", "seed_edge_curves"},
            settings=AnalysisLayerSettings(
                trace_tangent_tolerance_degrees=30.0,
                trace_curvature_policy="require",
                trace_curvature_tolerance_degrees=4.0,
            ),
        )
        curvature_changed = cache["layer.seed_edge_curves"]
        self.assertIs(curvature_changed.ridge_state, source_changed.ridge_state)
        self.assertIsNot(
            curvature_changed.trace_state, source_changed.trace_state
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

    def test_net_subtraction_recomputes_only_reference_edges_and_dependents(self) -> None:
        import numpy as np

        from seedvision.visualization import AnalysisLayerSettings, build_analysis_layers

        image = np.full((48, 56, 3), 210, dtype=np.uint8)
        image[:, 28:] = (45, 70, 175)
        valid = np.full(image.shape[:2], 255, dtype=np.uint8)
        cache: dict[str, object] = {}
        first = build_analysis_layers(
            image,
            valid,
            np.empty((0, 2), np.float32),
            np.empty((0,), np.float32),
            24.0,
            offset_x=0,
            offset_y=0,
            cache_values=cache,
        )
        first_prototypes = cache["layer.reference_texture_prototypes"]
        first_edges = cache["layer.reference_edge_probability"]
        first_net = first.net_physical_edge_probability
        first_supported = first.reference_edge_probability

        changed = build_analysis_layers(
            image,
            valid,
            np.empty((0, 2), np.float32),
            np.empty((0,), np.float32),
            24.0,
            offset_x=0,
            offset_y=0,
            cache_values=cache,
            dirty_nodes={"reference_edge_probability"},
            settings=AnalysisLayerSettings(
                net_physical_edge_internal_scale=1.25
            ),
        )
        self.assertIs(
            cache["layer.reference_texture_prototypes"], first_prototypes
        )
        self.assertIsNot(cache["layer.reference_edge_probability"], first_edges)
        self.assertIsNot(changed.net_physical_edge_probability, first_net)
        self.assertIsNot(changed.reference_edge_probability, first_supported)
        self.assertEqual(changed.net_physical_edge_internal_scale, 1.25)

    def test_reference_prototype_footprints_show_exact_descriptor_geometry(self) -> None:
        import numpy as np

        from seedvision.visualization import (
            AnalysisLayers,
            NoiseFrequencyProfile,
            ReferenceTextureProfile,
            ReferenceTexturePrototype,
        )

        size = 72
        zeros = np.zeros((size, size), dtype=np.uint8)
        valid = np.full((size, size), 255, dtype=np.uint8)
        patch = np.zeros((20, 20, 3), dtype=np.uint8)
        profile = ReferenceTextureProfile(
            prototypes=(
                ReferenceTexturePrototype(
                    "foreground", patch, 1.0, 20, (18.0, 18.0)
                ),
                ReferenceTexturePrototype(
                    "physical_edge", patch, 1.0, 20, (48.0, 42.0), 0.0
                ),
            ),
            material_context_radius_px=6.0,
            edge_strip_normal_offset_px=5.0,
            edge_strip_tangent_half_length_px=9.0,
        )
        layers = AnalysisLayers(
            offset_x=0,
            offset_y=0,
            instance_labels=zeros,
            instance_colours=np.zeros((1, 3), dtype=np.uint8),
            background_likelihood=zeros,
            refined_background_likelihood=zeros,
            noise_frequency_profile=NoiseFrequencyProfile(
                band_scales_px=(1.0, 2.0, 3.0),
                descriptor_centre=(0.0, 0.0, 0.0),
                target_sample_count=0,
            ),
            edge_likelihood=zeros,
            directed_edge_hue=zeros,
            undirected_edge_hue=zeros,
            seed_edge_curve_likelihood=zeros,
            seed_edge_curve_radius_px=zeros,
            valid_mask=valid,
            reference_texture_profile=profile,
        )
        footprints = layers.reference_prototype_footprints_rgba()
        self.assertEqual(footprints.shape, (size, size, 4))
        self.assertGreater(int(footprints[18, 18, 3]), 0)
        self.assertGreater(int(footprints[42, 48, 3]), 0)
        self.assertGreater(int(footprints[37, 48, 3]), 0)
        self.assertGreater(int(footprints[47, 48, 3]), 0)
        self.assertEqual(int(footprints[30, 48, 3]), 0)

    def test_hue_only_display_uses_fixed_value_and_neutral_gray(self) -> None:
        import numpy as np

        from seedvision.cuda import CudaContext
        from seedvision.cuda.layers import hue_only_rgb
        from seedvision.cuda.ops import image_to_tensor

        image = np.asarray(
            (((0, 0, 255), (0, 255, 0), (96, 96, 96)),), dtype=np.uint8
        )
        context = CudaContext.resolve(requested="cpu")
        displayed = np.asarray(
            hue_only_rgb(image_to_tensor(image, context)), dtype=np.uint8
        )
        self.assertEqual(tuple(displayed[0, 0]), (158, 0, 0))
        self.assertEqual(tuple(displayed[0, 1]), (0, 158, 0))
        self.assertEqual(tuple(displayed[0, 2]), (158, 158, 158))

    def test_stationary_wavelet_layers_reconstruct_source_exactly(self) -> None:
        import numpy as np
        import torch

        from seedvision.cuda import CudaContext
        from seedvision.cuda.layers import stationary_wavelet_decomposition
        from seedvision.cuda.ops import image_to_tensor

        rng = np.random.default_rng(17)
        image = rng.integers(0, 256, (37, 43, 3), dtype=np.uint8)
        context = CudaContext.resolve(requested="cpu")
        source = image_to_tensor(image, context).float()
        details, residual = stationary_wavelet_decomposition(source, 2)
        reconstructed = residual.gpu_tensor(dtype=torch.float32).clone()
        for detail in details:
            reconstructed += detail.gpu_tensor(dtype=torch.float32)
        self.assertLess(float(torch.max(torch.abs(reconstructed - source))), 1e-4)
        self.assertEqual(
            float(details[2].gpu_tensor(dtype=torch.float32).abs().max()), 0.0
        )
        self.assertEqual(
            float(details[3].gpu_tensor(dtype=torch.float32).abs().max()), 0.0
        )

    def test_edge_gradients_consume_selected_wavelet_layers_and_method(self) -> None:
        import numpy as np

        from seedvision.cuda import CudaContext
        from seedvision.cuda.layers import (
            directional_edges,
            stationary_wavelet_decomposition,
        )
        from seedvision.cuda.ops import image_to_tensor
        from seedvision.visualization import AnalysisLayerSettings

        rng = np.random.default_rng(91)
        image = rng.integers(0, 256, (61, 67, 3), dtype=np.uint8)
        valid = np.full(image.shape[:2], 255, np.uint8)
        context = CudaContext.resolve(requested="cpu")
        details, residual = stationary_wavelet_decomposition(
            image_to_tensor(image, context).float(), 4
        )
        original = directional_edges(
            image,
            valid,
            AnalysisLayerSettings(edge_gradient_method="scharr"),
            cuda_context=context,
            wavelet_details=details,
            wavelet_residual=residual,
        )
        method_only = directional_edges(
            image,
            valid,
            AnalysisLayerSettings(edge_gradient_method="prewitt"),
            cuda_context=context,
            wavelet_details=details,
            wavelet_residual=residual,
        )
        wavelet = directional_edges(
            image,
            valid,
            AnalysisLayerSettings(
                edge_gradient_method="prewitt",
                edge_gradient_include_original=False,
                edge_gradient_include_wavelet_detail_1=True,
            ),
            cuda_context=context,
            wavelet_details=details,
            wavelet_residual=residual,
        )
        self.assertFalse(
            np.array_equal(
                np.asarray(original.strength_raster),
                np.asarray(wavelet.strength_raster),
            )
        )
        self.assertFalse(
            np.array_equal(
                np.asarray(original.strength_raster),
                np.asarray(method_only.strength_raster),
            )
        )
        self.assertFalse(
            np.array_equal(
                np.asarray(original.directed_hue_raster),
                np.asarray(wavelet.directed_hue_raster),
            )
        )

    def test_edge_gradients_can_replace_colour_primary_with_despeckled_grayscale(self) -> None:
        import numpy as np
        import torch

        from seedvision.cuda import CudaContext
        from seedvision.cuda.layers import directional_edges
        from seedvision.visualization import AnalysisLayerSettings

        image = np.full((64, 64, 3), 180, np.uint8)
        image[:, 31:, 2] = 245
        valid = np.full((64, 64), 255, np.uint8)
        context = CudaContext.resolve(requested="cpu")
        neutral = torch.full((1, 1, 64, 64), 0.5, dtype=torch.float32)
        original = directional_edges(
            image,
            valid,
            AnalysisLayerSettings(),
            cuda_context=context,
        )
        replaced = directional_edges(
            image,
            valid,
            AnalysisLayerSettings(
                edge_gradient_use_despeckled_flattened=True
            ),
            cuda_context=context,
            despeckled_flattened_tensor=neutral,
        )
        self.assertGreater(
            float(np.asarray(original.strength_raster)[:, 31:33].mean()),
            float(np.asarray(replaced.strength_raster)[:, 31:33].mean()) + 20.0,
        )
        with self.assertRaisesRegex(ValueError, "did not supply"):
            directional_edges(
                image,
                valid,
                AnalysisLayerSettings(
                    edge_gradient_use_despeckled_flattened=True
                ),
                cuda_context=context,
            )

    def test_oriented_traces_retain_non_generic_semantic_ridge_sources(self) -> None:
        import cv2
        import numpy as np

        from seedvision.cuda import CudaContext
        from seedvision.cuda.layers import directional_edges, seed_boundary_tracing
        from seedvision.visualization import AnalysisLayerSettings

        image = np.full((96, 96, 3), 225, np.uint8)
        cv2.circle(image, (48, 48), 25, (35, 55, 95), -1)
        valid = np.full((96, 96), 255, np.uint8)
        semantic_ridge = np.zeros((96, 96), np.uint8)
        cv2.circle(semantic_ridge, (48, 48), 25, 255, 1)
        settings = AnalysisLayerSettings(
            trace_maximum_gap_px=2,
            trace_minimum_length_fraction=0.20,
        )
        context = CudaContext.resolve(requested="cpu")
        gradients = directional_edges(
            image, valid, settings, cuda_context=context
        )
        for source_name in (
            "reference_ridges",
            "net_reference_ridges",
            "normalized_net_reference_ridges",
        ):
            traced = seed_boundary_tracing(
                gradients,
                50.0,
                settings,
                trace_ridge_override=semantic_ridge,
                trace_source_name=source_name,
                compute_final=False,
                cuda_context=context,
            )
            self.assertGreater(
                int(np.max(np.asarray(traced.trace_labels))),
                0,
                source_name,
            )

    def test_semantic_trace_source_is_rethinned_after_interpolation_halo(self) -> None:
        import numpy as np

        from seedvision.cuda import CudaContext
        from seedvision.cuda.layers import directional_edges, seed_boundary_tracing
        from seedvision.visualization import AnalysisLayerSettings

        image = np.full((80, 80, 3), 225, np.uint8)
        image[:, :40] = 35
        valid = np.full((80, 80), 255, np.uint8)
        # Model the three-pixel halo produced when a one-pixel ridge is restored
        # bilinearly from its bounded working resolution.
        semantic_halo = np.zeros((80, 80), np.uint8)
        semantic_halo[6:74, 39:42] = 255
        settings = AnalysisLayerSettings(
            trace_minimum_length_fraction=0.10,
            trace_maximum_gap_px=2,
        )
        context = CudaContext.resolve(requested="cpu")
        gradients = directional_edges(
            image, valid, settings, cuda_context=context
        )
        traced = seed_boundary_tracing(
            gradients,
            40.0,
            settings,
            trace_ridge_override=semantic_halo,
            trace_source_name="normalized_net_reference_ridges",
            compute_final=False,
            cuda_context=context,
        )

        locations = np.argwhere(np.asarray(traced.trace_labels) > 0)
        self.assertGreater(len(locations), 20)
        self.assertLessEqual(len(np.unique(locations[:, 1])), 1)


if __name__ == "__main__":
    unittest.main()
