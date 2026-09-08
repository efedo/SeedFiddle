from __future__ import annotations

import unittest

import cv2
import numpy as np

from seedvision.visualization import (
    ADVANCED_NODE_MODES,
    ADVANCED_OVERLAY_LABELS,
    AdvancedAnalysisSettings,
    build_advanced_analysis_layers,
)


def _synthetic_inputs():
    height = width = 96
    yy, xx = np.ogrid[:height, :width]
    first = (xx - 38) ** 2 + (yy - 48) ** 2 < 20**2
    second = (xx - 61) ** 2 + (yy - 48) ** 2 < 18**2
    foreground = first | second
    crop = np.full((height, width, 3), 224, np.uint8)
    crop[foreground] = (58, 104, 154)
    crop[first & ((xx + yy) % 11 < 3)] = (38, 72, 118)
    valid = np.full((height, width), 255, np.uint8)
    foreground_mask = np.uint8(foreground) * 255
    feature = np.float32(foreground) * 54.0
    distance = cv2.distanceTransform(foreground_mask, cv2.DIST_L2, 5)
    labels = np.zeros((height, width), np.int32)
    labels[first] = 1
    labels[second] = 2
    centers = np.asarray(((38, 48), (61, 48)), np.float32)
    radii = np.asarray((20, 18), np.float32)
    geometry = np.column_stack((centers, radii))
    return (
        crop,
        valid,
        feature,
        20.0,
        foreground_mask,
        distance,
        labels,
        centers,
        radii,
        geometry,
        geometry.copy(),
        38.0,
    )


class AdvancedAnalysisTests(unittest.TestCase):
    def _build(self, device: str):
        return build_advanced_analysis_layers(
            *_synthetic_inputs(),
            offset_x=10,
            offset_y=20,
            full_image_shape=(140, 160),
            calibration_anchors=((20.0, 20.0, 0.10), (130.0, 110.0, 0.20)),
            settings=AdvancedAnalysisSettings(
                compute_device=device,
                allow_cpu_fallback=False,
                maximum_dimension=256,
            ),
        )

    def test_all_fourteen_products_and_intermediates_are_renderable(self) -> None:
        layers = self._build("cpu")
        self.assertEqual(layers.backend.used, "cpu")
        self.assertEqual(len(ADVANCED_NODE_MODES), 14)
        for mode in ADVANCED_NODE_MODES.values():
            rgba = layers.rgba(mode)
            self.assertEqual(rgba.shape, (96, 96, 4), mode)
            self.assertEqual(rgba.dtype, np.uint8)
        for _, mode in ADVANCED_OVERLAY_LABELS:
            self.assertEqual(layers.rgba(mode).shape, (96, 96, 4), mode)
        self.assertEqual(len(layers.colour_probabilities), 6)
        self.assertEqual(len(layers.pattern_probabilities), 6)
        self.assertTrue(layers.contact_pairs)
        self.assertEqual(len(layers.instance_summaries), 2)
        first = layers.instance_summaries[0]
        self.assertGreater(first.pixel_count, 0)
        self.assertAlmostEqual(sum(first.colour_proportions), 1.0, places=4)
        self.assertAlmostEqual(sum(first.pattern_proportions), 1.0, places=4)
        self.assertIn(first.dominant_colour, layers.colour_class_names)

    def test_cuda_is_the_active_backend_when_available(self) -> None:
        try:
            import torch
        except ImportError as error:
            self.skipTest(str(error))
        if not torch.cuda.is_available():
            self.skipTest("PyTorch CUDA is unavailable")
        layers = self._build("cuda")
        self.assertEqual(layers.backend.used, "cuda")
        self.assertIn("NVIDIA", layers.backend.device_name.upper())
        from seedvision.cuda import GpuRaster

        raster_values = (
            layers.valid_mask,
            *layers.rasters.values(),
            *layers.colour_probabilities,
            *layers.pattern_probabilities,
            *(value for pair in layers.hue_rasters.values() for value in pair),
        )
        self.assertTrue(
            all(isinstance(value, GpuRaster) for value in raster_values)
        )
        self.assertTrue(all(value.gpu_resident for value in raster_values))
        self.assertTrue(all(not value.is_materialized for value in raster_values))

    def test_trait_only_change_reuses_independent_cached_products(self) -> None:
        first = self._build("cpu")
        second = build_advanced_analysis_layers(
            *_synthetic_inputs(),
            offset_x=10,
            offset_y=20,
            full_image_shape=(140, 160),
            calibration_anchors=((20.0, 20.0, 0.10),),
            settings=AdvancedAnalysisSettings(
                compute_device="cpu",
                maximum_dimension=256,
                wrinkle_scale_fraction=0.08,
            ),
            previous=first,
            dirty_nodes={"wrinkling"},
        )
        self.assertIs(
            second.rasters["illumination_field"],
            first.rasters["illumination_field"],
        )
        self.assertIs(
            second.rasters["calibration_residual_risk"],
            first.rasters["calibration_residual_risk"],
        )
        self.assertIs(second.colour_probabilities, first.colour_probabilities)
        self.assertIsNot(
            second.rasters["wrinkling_likelihood"],
            first.rasters["wrinkling_likelihood"],
        )

    def test_seed_interior_does_not_recombine_raw_foreground_noise(self) -> None:
        inputs = _synthetic_inputs()
        shape = inputs[0].shape[:2]
        settings = AdvancedAnalysisSettings(
            compute_device="cpu",
            maximum_dimension=256,
            interior_smoothing_fraction=0.005,
            interior_background_weight=0.0,
            interior_foreground_noise_weight=1.0,
        )
        quiet = build_advanced_analysis_layers(
            *inputs,
            offset_x=10,
            offset_y=20,
            full_image_shape=(140, 160),
            foreground_probability=np.full(shape, 255, np.uint8),
            foreground_noise_probability=np.zeros(shape, np.uint8),
            settings=settings,
        )
        noisy = build_advanced_analysis_layers(
            *inputs,
            offset_x=10,
            offset_y=20,
            full_image_shape=(140, 160),
            foreground_probability=np.full(shape, 255, np.uint8),
            foreground_noise_probability=np.full(shape, 255, np.uint8),
            settings=settings,
            previous=quiet,
            dirty_nodes={"seed_interior"},
        )

        quiet_interior = np.asarray(quiet.rasters["seed_interior_probability"])
        noisy_interior = np.asarray(noisy.rasters["seed_interior_probability"])
        self.assertGreater(float(np.mean(quiet_interior[16:-16, 16:-16])), 245.0)
        np.testing.assert_array_equal(noisy_interior, quiet_interior)
        self.assertIs(
            noisy.rasters["boundary_magnitude"],
            quiet.rasters["boundary_magnitude"],
        )

    def test_precomputed_sensor_noise_matches_image_quality_output(self) -> None:
        import torch

        from seedvision.visualization.advanced import (
            sensor_noise_likelihood_tensor,
        )

        inputs = _synthetic_inputs()
        crop, valid = inputs[:2]
        source_tensor = torch.from_numpy(crop.copy()).permute(2, 0, 1)[None].float()
        valid_tensor = torch.from_numpy(valid > 0)[None, None]
        settings = AdvancedAnalysisSettings(
            compute_device="cpu",
            allow_cpu_fallback=False,
            maximum_dimension=256,
        )
        sensor_noise = sensor_noise_likelihood_tensor(
            source_tensor,
            valid_tensor,
            inputs[11],
            settings,
        )
        direct = build_advanced_analysis_layers(
            *inputs,
            offset_x=10,
            offset_y=20,
            full_image_shape=(140, 160),
            settings=settings,
            source_tensor=source_tensor,
            valid_tensor=valid_tensor,
        )
        reused = build_advanced_analysis_layers(
            *inputs,
            offset_x=10,
            offset_y=20,
            full_image_shape=(140, 160),
            settings=settings,
            source_tensor=source_tensor,
            valid_tensor=valid_tensor,
            sensor_noise_tensor=sensor_noise,
        )
        difference = np.abs(
            np.asarray(direct.rasters["sensor_noise"], dtype=np.int16)
            - np.asarray(reused.rasters["sensor_noise"], dtype=np.int16)
        )
        self.assertLessEqual(int(difference.max()), 1)

    def test_local_lighting_flattens_gradient_and_classifies_extremes(self) -> None:
        import torch

        from seedvision.visualization.advanced import (
            local_lighting_evidence_tensors,
        )

        height = width = 128
        horizontal = np.linspace(0.35, 0.85, width, dtype=np.float32)
        luminance = np.tile(horizontal, (height, 1))
        cv2.circle(luminance, (40, 64), 12, 0.15, -1)
        cv2.circle(luminance, (90, 64), 12, 1.00, -1)
        image = np.uint8(np.clip(luminance * 255.0, 0.0, 255.0))
        image = np.repeat(image[:, :, None], 3, axis=2)
        source = torch.from_numpy(image).permute(2, 0, 1)[None].float()
        valid = torch.ones((1, 1, height, width), dtype=torch.bool)
        settings = AdvancedAnalysisSettings(
            compute_device="cpu",
            allow_cpu_fallback=False,
            maximum_dimension=256,
        )
        products = local_lighting_evidence_tensors(
            source, valid, 32.0, settings
        )
        _, flattened, shadow, highlight, _ = products

        raw_difference = abs(float(luminance[20, 20] - luminance[20, 108]))
        flat_difference = abs(
            float(flattened[0, 0, 20, 20] - flattened[0, 0, 20, 108])
        )
        self.assertLess(flat_difference, raw_difference * 0.35)
        self.assertGreater(float(shadow[0, 0, 64, 40]), 0.60)
        self.assertLess(float(highlight[0, 0, 64, 40]), 0.10)
        self.assertGreater(float(highlight[0, 0, 64, 90]), 0.60)
        self.assertLess(float(shadow[0, 0, 64, 90]), 0.10)

        inputs = _synthetic_inputs()
        crop, input_valid = inputs[:2]
        input_source = (
            torch.from_numpy(crop.copy()).permute(2, 0, 1)[None].float()
        )
        input_valid_tensor = torch.from_numpy(input_valid > 0)[None, None]
        precomputed = local_lighting_evidence_tensors(
            input_source, input_valid_tensor, inputs[11], settings
        )
        direct = build_advanced_analysis_layers(
            *inputs,
            offset_x=10,
            offset_y=20,
            full_image_shape=(140, 160),
            settings=settings,
            source_tensor=input_source,
            valid_tensor=input_valid_tensor,
        )
        reused = build_advanced_analysis_layers(
            *inputs,
            offset_x=10,
            offset_y=20,
            full_image_shape=(140, 160),
            settings=settings,
            source_tensor=input_source,
            valid_tensor=input_valid_tensor,
            local_lighting_tensors=precomputed,
        )
        for name in (
            "illumination_field",
            "flattened_grayscale",
            "shadow_likelihood",
            "highlight_likelihood",
            "reflectance_image",
        ):
            difference = np.abs(
                np.asarray(direct.rasters[name], dtype=np.int16)
                - np.asarray(reused.rasters[name], dtype=np.int16)
            )
            self.assertLessEqual(int(difference.max()), 1, name)

    def test_despeckling_removes_only_compact_surrounded_dark_residuals(self) -> None:
        import torch

        from seedvision.visualization.advanced import (
            local_lighting_evidence_tensors,
        )

        height = width = 128
        image = np.full((height, width, 3), 205, np.uint8)
        image[61:66, 61:66] = 45
        # A long dark trace is thin but not enclosed at the configured ring;
        # it must survive the same maximum-diameter setting.
        image[20:108, 30:32] = 45
        source = torch.from_numpy(image).permute(2, 0, 1)[None].float()
        valid = torch.ones((1, 1, height, width), dtype=torch.bool)
        products = local_lighting_evidence_tensors(
            source,
            valid,
            40.0,
            AdvancedAnalysisSettings(
                compute_device="cpu",
                allow_cpu_fallback=False,
                maximum_dimension=256,
                illumination_scale_fraction=1.2,
                despeckle_maximum_diameter_fraction=0.20,
                despeckle_minimum_darkness_levels=12.0,
                despeckle_periphery_width_fraction=0.05,
                despeckle_minimum_lighter_surround_fraction=0.90,
            ),
            include_despeckle=True,
        )
        _, flattened, _, _, _, despeckled, removed = products
        self.assertGreater(float(removed[0, 0, 63, 63]), 0.99)
        self.assertGreater(
            float(despeckled[0, 0, 63, 63]),
            float(flattened[0, 0, 63, 63]) + 0.20,
        )
        self.assertEqual(float(removed[0, 0, 64, 30]), 0.0)
        self.assertAlmostEqual(
            float(despeckled[0, 0, 64, 30]),
            float(flattened[0, 0, 64, 30]),
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
