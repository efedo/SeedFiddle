"""Photographs without a vessel are analysed around their confident seeds."""

from tests.functional_units import functional_units
from dataclasses import replace
from pathlib import Path
import unittest

import cv2
import numpy as np
import torch

from seedvision.cuda import CudaContext
from seedvision.cuda.ops import distance_transform, quantile
from seedvision.segmentation.automatic_references import ConfidentSeed
from seedvision.segmentation.baseline import (
    BaselineSettings,
    PipelineAnalysisCache,
    _calibration_object_polygons,
    _foreground_feature,
    _scene_analysis_crop,
    _scene_analysis_region,
    analyze_path,
)

ROOT = Path(__file__).resolve().parents[1]
CHART = np.array(((700.0, 40.0), (980.0, 40.0), (980.0, 300.0), (700.0, 300.0)))


def seed_at(x, y, radius=12):
    """A round confident seed of the given radius centred on (x, y)."""

    size = 2 * radius + 1
    mask = np.zeros((size, size), bool)
    cv2.circle(mask.view(np.uint8), (radius, radius), radius, 1, -1)
    return ConfidentSeed(left=x - radius, top=y - radius, mask=mask, diameter_px=2.0 * radius)


@functional_units
class SceneAnalysisRegionTests(unittest.TestCase):
    def check_test_region_spans_the_confident_seeds_and_leaves_out_the_chart(self):
        device = CudaContext.resolve().device
        seeds = (seed_at(200, 200), seed_at(500, 220), seed_at(350, 480))
        region = _scene_analysis_region((700, 1000), seeds, 24.0, (CHART,), device).cpu().numpy()
        # Inside the hull and within the margin beyond it (a quarter of the
        # spread's 300-odd px extent beats two 24 px seed widths).
        for x, y in ((350, 300), (200, 200), (140, 200), (350, 550)):
            self.assertTrue(region[y, x], (x, y))
        for x, y in ((40, 650), (950, 650), (60, 60)):
            self.assertFalse(region[y, x], (x, y))
        # The chart and its clearance stay out even where the margin reaches.
        self.assertFalse(region[200, 705])
        self.assertFalse(region[60:290, 700:980].any())

    def check_test_too_few_confident_seeds_analyse_the_whole_image_but_the_chart(self):
        device = CudaContext.resolve().device
        seeds = (seed_at(200, 200), seed_at(500, 220))
        region = _scene_analysis_region((700, 1000), seeds, 24.0, (CHART,), device).cpu().numpy()
        self.assertTrue(region[650, 40])
        self.assertTrue(region[650, 950])
        self.assertFalse(region[150, 850])

    def check_test_scene_crop_bounds_the_region_and_its_circle_reaches_every_corner(self):
        device = CudaContext.resolve().device
        image = np.zeros((700, 1000, 3), np.uint8)
        seeds = (seed_at(200, 200), seed_at(500, 220), seed_at(350, 480))
        region = _scene_analysis_region((700, 1000), seeds, 24.0, (CHART,), device).cpu().numpy()
        crop, x0, y0, circle, excluded = _scene_analysis_crop(
            image, seeds, 24.0, (CHART,), 0.90, device
        )
        rows, columns = np.nonzero(region)
        self.assertEqual((x0, y0), (columns.min(), rows.min()))
        self.assertEqual(crop.shape[:2], (rows.max() + 1 - y0, columns.max() + 1 - x0))
        np.testing.assert_array_equal(
            excluded.cpu().numpy(), ~region[y0 : y0 + crop.shape[0], x0 : x0 + crop.shape[1]]
        )
        for corner_x, corner_y in ((0, 0), (crop.shape[1], 0), (0, crop.shape[0]), crop.shape[1::-1]):
            distance = np.hypot(x0 + corner_x - circle.center_x, y0 + corner_y - circle.center_y)
            self.assertLessEqual(distance, circle.outer_radius * 0.90)

    def check_test_excluded_pixels_leave_both_evidence_regions(self):
        context = CudaContext.resolve()
        crop = np.full((120, 160, 3), 200, np.uint8)
        excluded = torch.zeros((120, 160), dtype=torch.bool, device=context.device)
        excluded[:, :40] = True
        _feature, _probability, valid, analysis_valid, *_ = _foreground_feature(
            crop, BaselineSettings(), context, valid_radius=400.0, analysis_radius=400.0,
            excluded_region=excluded,
        )
        valid = np.asarray(valid) > 0
        analysis_valid = np.asarray(analysis_valid) > 0
        self.assertFalse(valid[:, :40].any())
        self.assertFalse(analysis_valid[:, :40].any())
        self.assertTrue(valid[:, 40:].all())
        self.assertTrue(analysis_valid[:, 40:].all())

    def check_test_distance_transform_handles_elongated_frames(self):
        from scipy import ndimage

        device = CudaContext.resolve().device
        for height, width in ((1100, 500), (300, 2000)):
            mask = np.zeros((height, width), bool)
            mask[40 : height - 40, 30 : width - 30] = True
            mask[height // 2, width // 3] = False
            distance = distance_transform(torch.as_tensor(mask, device=device)[None, None])
            self.assertLess(
                float(np.abs(distance[0, 0].cpu().numpy() - ndimage.distance_transform_edt(mask)).max()),
                0.5,
            )

    def check_test_quantile_matches_torch_beyond_its_input_limit(self):
        device = CudaContext.resolve().device
        count = (1 << 24) + 1001
        values = torch.randperm(count, device=device).float()
        for q in (0.0, 0.25, 0.995, 1.0):
            self.assertAlmostEqual(float(quantile(values, q)), q * (count - 1), delta=0.5)
        small = torch.rand(5000, device=device)
        self.assertEqual(float(quantile(small, 0.37)), float(torch.quantile(small, 0.37)))

    def check_test_dishless_fixture_is_analysed_around_its_seeds(self):
        """pi_457995 has no vessel: its 25 seeds lie beside a chart and above a ruler."""

        path = ROOT / "images" / "pi_457995_s.jpg"
        if not path.exists():
            self.skipTest("GRIN photographs are not present")
        cache = PipelineAnalysisCache()
        result = analyze_path(path, node_cache=cache)
        crop, offset_x, offset_y = cache.values["crop"]
        self.assertIsNotNone(cache.values["crop.scene"])
        self.assertFalse(result.vessel_present)
        self.assertLess(result.vessel_rim_support, BaselineSettings().vessel_minimum_rim_support)
        self.assertGreater(len(result.analysis_region_outlines), 0)
        analysed = np.zeros(crop.shape[:2], bool) if cache.values["crop.excluded"] is None else (
            ~cache.values["crop.excluded"].cpu().numpy()
        )
        # Seeds at the four corners of the seed area, in corrected coordinates
        # read off the fixture (each centre within a few pixels).
        for x, y in ((57, 66), (51, 351), (414, 54), (369, 409)):
            self.assertTrue(analysed[y - offset_y, x - offset_x], (x, y))
            # The outline shown in place of the vessel encloses them too.
            self.assertTrue(any(
                cv2.pointPolygonTest(outline.reshape(-1, 1, 2), (float(x), float(y)), False) > 0
                for outline in result.analysis_region_outlines
            ), (x, y))
        height, width = result.calibration.gpu_corrected_bgr.shape[-2:]
        for polygon in _calibration_object_polygons(result.calibration):
            outline = np.zeros((height, width), np.uint8)
            cv2.fillConvexPoly(outline, np.round(polygon).astype(np.int32), 1)
            inside = outline[offset_y : offset_y + crop.shape[0], offset_x : offset_x + crop.shape[1]] > 0
            self.assertFalse((analysed & inside).any())
            for proposal in result.proposals:
                self.assertFalse(outline[round(proposal.center_y), round(proposal.center_x)])

        again = analyze_path(path, node_cache=cache)
        self.assertIn("foreground_segmentation", cache.last_reused_nodes)
        self.assertEqual(len(again.proposals), len(result.proposals))

        # Counting the fitted circle as a vessel brings back the dish crop.
        vessel = analyze_path(
            path, replace(BaselineSettings(), vessel_minimum_rim_support=0.0),
            node_cache=cache, dirty_nodes={"seed_scale_estimation"},
        )
        self.assertIsNone(cache.values["crop.scene"])
        self.assertTrue(vessel.vessel_present)
        self.assertEqual(vessel.analysis_region_outlines, ())
        self.assertLess(vessel.analysis_region_pixel_count, result.analysis_region_pixel_count)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for the analysis region")
    def test_region(self):
        """4 related SceneAnalysisRegionTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_region_spans_the_confident_seeds_and_leaves_out_the_chart)
        self.scenario(self.check_test_too_few_confident_seeds_analyse_the_whole_image_but_the_chart)
        self.scenario(self.check_test_scene_crop_bounds_the_region_and_its_circle_reaches_every_corner)
        self.scenario(self.check_test_excluded_pixels_leave_both_evidence_regions)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for the tensor operations")
    def test_large_frames(self):
        """2 related SceneAnalysisRegionTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_distance_transform_handles_elongated_frames)
        self.scenario(self.check_test_quantile_matches_torch_beyond_its_input_limit)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for the analysis")
    def test_dishless_fixture(self):
        self.scenario(self.check_test_dishless_fixture_is_analysed_around_its_seeds, labelled=False)


if __name__ == "__main__":
    unittest.main()
