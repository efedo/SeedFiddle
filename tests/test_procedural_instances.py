from __future__ import annotations

import unittest

import cv2
import numpy as np

from seedvision.cuda import GpuRaster
from seedvision.segmentation.procedural import (
    ProceduralInstanceSettings,
    procedural_seed_instances,
)


class ProceduralInstanceTests(unittest.TestCase):
    @staticmethod
    def _touching_seed_scene():
        shape = (240, 240)
        valid = np.zeros(shape, np.uint8)
        cv2.circle(valid, (120, 120), 112, 255, -1)
        identities = np.zeros(shape, np.int32)
        for identifier, centre, angle in (
            (1, (78, 120), -18),
            (2, (120, 120), 12),
            (3, (162, 120), -8),
        ):
            cv2.ellipse(identities, centre, (29, 22), angle, 0, 360, identifier, -1)
        material = np.uint8(identities > 0) * 235
        material = cv2.GaussianBlur(material, (0, 0), 2.0)
        boundary = np.zeros(shape, np.uint8)
        for identifier in (1, 2, 3):
            region = np.uint8(identities == identifier) * 255
            boundary = np.maximum(
                boundary,
                cv2.morphologyEx(region, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)),
            )
        background = np.uint8(255 - material)
        return valid, identities, material, background, boundary

    def test_separates_touching_seed_shapes_and_exposes_diagnostics(self) -> None:
        valid, _identities, material, background, boundary = self._touching_seed_scene()
        result = procedural_seed_instances(
            valid,
            54.0,
            foreground_probability=material,
            foreground_noise_probability=material,
            background_probability=background,
            refined_background_probability=background,
            edge_magnitude=boundary,
            edge_ridges=boundary,
            sensor_noise=boundary,
            shadow_likelihood=boundary,
            settings=ProceduralInstanceSettings(
                centre_minimum_separation_fraction=0.58,
                minimum_marker_score=0.10,
            ),
        )
        self.assertEqual(result.labels.shape, valid.shape)
        self.assertGreaterEqual(result.count, 3)
        self.assertEqual(int(result.labels.max()), result.count)
        self.assertEqual(result.instance_rgba().shape, (*valid.shape, 4))
        self.assertEqual(result.confidence_raster().shape, valid.shape)
        self.assertGreater(np.count_nonzero(result.boundary_cost), 0)
        self.assertGreater(np.count_nonzero(result.centre_likelihood), 0)

    def test_distinct_painted_instances_become_authoritative_markers(self) -> None:
        valid, identities, material, background, boundary = self._touching_seed_scene()
        annotations = np.zeros_like(identities, dtype=np.uint16)
        for identifier, centre in ((1, (78, 120)), (2, (120, 120)), (3, (162, 120))):
            cv2.circle(annotations, centre, 4, identifier, -1)
        valid = cv2.resize(valid, (480, 480), interpolation=cv2.INTER_NEAREST)
        material = cv2.resize(material, (480, 480), interpolation=cv2.INTER_LINEAR)
        background = cv2.resize(background, (480, 480), interpolation=cv2.INTER_LINEAR)
        boundary = cv2.resize(boundary, (480, 480), interpolation=cv2.INTER_LINEAR)
        annotations = cv2.resize(annotations, (480, 480), interpolation=cv2.INTER_NEAREST)
        result = procedural_seed_instances(
            valid,
            108.0,
            foreground_probability=material,
            foreground_noise_probability=material,
            background_probability=background,
            refined_background_probability=background,
            edge_magnitude=boundary,
            edge_ridges=boundary,
            sensor_noise=boundary,
            shadow_likelihood=boundary,
            seed_instance_annotations=annotations,
            settings=ProceduralInstanceSettings(
                minimum_marker_score=0.95,
                working_maximum_dimension=256,
            ),
        )
        self.assertGreaterEqual(result.count, 3)
        for expected in ((156, 240), (240, 240), (324, 240)):
            distances = np.hypot(
                result.centres_xy[:, 0] - expected[0],
                result.centres_xy[:, 1] - expected[1],
            )
            self.assertLess(float(distances.min()), 30.0)

    def test_gpu_rasters_resize_without_materializing_full_host_caches(self) -> None:
        import torch

        valid, _identities, material, background, boundary = self._touching_seed_scene()
        valid = cv2.resize(valid, (480, 480), interpolation=cv2.INTER_NEAREST)
        material = cv2.resize(material, (480, 480), interpolation=cv2.INTER_LINEAR)
        background = cv2.resize(background, (480, 480), interpolation=cv2.INTER_LINEAR)
        boundary = cv2.resize(boundary, (480, 480), interpolation=cv2.INTER_LINEAR)

        def raster(values: np.ndarray, name: str) -> GpuRaster:
            return GpuRaster(
                torch.from_numpy(values)[None, None],
                numpy_dtype=np.uint8,
                name=name,
            )

        inputs = [
            raster(valid, "valid"),
            raster(material, "material"),
            raster(material, "material noise"),
            raster(background, "background"),
            raster(background, "background noise"),
            raster(boundary, "edge"),
            raster(boundary, "ridges"),
            raster(boundary, "sensor"),
            raster(boundary, "shadow"),
        ]
        result = procedural_seed_instances(
            inputs[0],
            108.0,
            foreground_probability=inputs[1],
            foreground_noise_probability=inputs[2],
            background_probability=inputs[3],
            refined_background_probability=inputs[4],
            edge_magnitude=inputs[5],
            edge_ridges=inputs[6],
            sensor_noise=inputs[7],
            shadow_likelihood=inputs[8],
            settings=ProceduralInstanceSettings(working_maximum_dimension=256),
        )
        self.assertGreater(result.count, 0)
        self.assertEqual(result.source_shape, (480, 480))
        self.assertLessEqual(max(result.labels.shape), 256)
        self.assertEqual(result.occupancy_likelihood.shape, result.labels.shape)
        self.assertEqual(result.boundary_cost.shape, result.labels.shape)
        self.assertTrue(all(not item.is_materialized for item in inputs))


if __name__ == "__main__":
    unittest.main()
