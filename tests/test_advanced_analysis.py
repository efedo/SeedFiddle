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

    def test_all_fifteen_products_and_intermediates_are_renderable(self) -> None:
        layers = self._build("cpu")
        self.assertEqual(layers.backend.used, "cpu")
        self.assertEqual(len(ADVANCED_NODE_MODES), 15)
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


if __name__ == "__main__":
    unittest.main()
