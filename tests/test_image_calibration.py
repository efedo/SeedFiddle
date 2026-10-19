from __future__ import annotations

from tests.functional_units import functional_units

import unittest


@functional_units
class ImageCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import cv2  # noqa: F401
            import numpy  # noqa: F401
        except ImportError as error:
            raise unittest.SkipTest(f"Calibration dependencies unavailable: {error}")

    @staticmethod
    def _synthetic_calibration_image(
        rotation_degrees: float = 0.0,
        *,
        swatch_offset: tuple[int, int] = (0, 0),
        swatch_half_size: tuple[int, int] = (38, 43),
    ):
        import cv2
        import numpy as np

        image = np.full((800, 1200, 3), 238, dtype=np.uint8)
        cv2.rectangle(image, (45, 35), (465, 755), (25, 25, 25), -1)
        colours = (
            (235, 235, 235), (180, 110, 40), (35, 150, 225), (165, 205, 90),
            (205, 205, 205), (180, 45, 155), (210, 70, 60), (180, 145, 215),
            (165, 165, 165), (30, 220, 225), (80, 80, 210), (60, 125, 65),
            (125, 125, 125), (45, 45, 210), (155, 70, 90), (210, 155, 105),
            (85, 85, 85), (70, 180, 65), (65, 195, 160), (155, 120, 75),
            (45, 45, 45), (210, 65, 40), (40, 175, 220), (95, 65, 135),
        )
        for row in range(6):
            for column in range(4):
                center_x = 105 + column * 100
                center_y = 100 + row * 118
                center_x += swatch_offset[0]
                center_y += swatch_offset[1]
                colour = colours[row * 4 + column]
                cv2.rectangle(
                    image,
                    (
                        center_x - swatch_half_size[0],
                        center_y - swatch_half_size[1],
                    ),
                    (
                        center_x + swatch_half_size[0],
                        center_y + swatch_half_size[1],
                    ),
                    colour,
                    -1,
                )

        cv2.rectangle(image, (510, 635), (1135, 760), (224, 224, 224), -1)
        cv2.rectangle(image, (510, 635), (1135, 760), (55, 55, 55), 2)
        for tick in range(151):
            x = 525 + tick * 4
            tick_height = 48 if tick % 10 == 0 else 32 if tick % 5 == 0 else 20
            cv2.line(image, (x, 645), (x, 645 + tick_height), (25, 25, 25), 1)

        cast = image.astype(np.float32)
        cast[:, :, 0] *= 1.12
        cast[:, :, 2] *= 0.88
        image = np.uint8(np.clip(cast, 0, 255))
        if rotation_degrees:
            matrix = cv2.getRotationMatrix2D((600, 400), rotation_degrees, 1.0)
            image = cv2.warpAffine(
                image,
                matrix,
                (1200, 800),
                borderMode=cv2.BORDER_REPLICATE,
            )
        return image






    def check_test_scale_is_rejected_when_longest_ticks_are_not_major_increments(self) -> None:
        import cv2

        from seedvision.calibration import detect_ruler

        image = self._synthetic_calibration_image()
        # Shorten every designed centimetre mark so the longest surviving
        # family occurs at 5 mm. Periodic positions alone must not be allowed to
        # call this a valid 0--15 cm alignment.
        for tick in range(0, 151, 10):
            x = 525 + tick * 4
            cv2.line(image, (x, 666), (x, 696), (224, 224, 224), 3)

        ruler = detect_ruler(image)

        self.assertEqual(len(ruler.metric_tick_segments), 151)
        self.assertFalse(ruler.metric_hierarchy_reliable)
        self.assertFalse(ruler.scale_reliable)
        self.assertNotEqual(ruler.metric_unit_divider_indices[0], 0)


    def check_test_swatch_grid_snaps_to_shared_printed_edges(self) -> None:
        import numpy as np

        from seedvision.calibration import detect_colour_card

        image = self._synthetic_calibration_image(
            swatch_offset=(5, 4), swatch_half_size=(34, 39)
        )
        card = detect_colour_card(image)
        for swatch in card.swatches:
            center_x = 105 + swatch.column * 100 + 5
            center_y = 100 + swatch.row * 118 + 4
            expected = np.asarray(
                (
                    (center_x - 34, center_y - 39),
                    (center_x + 34, center_y - 39),
                    (center_x + 34, center_y + 39),
                    (center_x - 34, center_y + 39),
                ),
                dtype=np.float64,
            )
            detected = np.asarray(swatch.corners, dtype=np.float64)
            self.assertLess(float(np.max(np.abs(detected - expected))), 2.5)



    def check_test_missing_references_produce_a_safe_uncalibrated_result(self) -> None:
        import numpy as np

        from seedvision.calibration import calibrate_image

        image = np.full((300, 400, 3), 180, dtype=np.uint8)
        result = calibrate_image(image)
        self.assertIsNone(result.colour_card)
        self.assertIsNone(result.ruler)
        self.assertIsNone(result.pixels_per_mm)
        self.assertEqual(result.deskew_degrees, 0.0)
        self.assertTrue(result.warnings)




    def check_test_mild_elliptical_rim_is_retained_with_conservative_bounds(self) -> None:
        import cv2
        import numpy as np
        import torch

        from seedvision.calibration.geometry import (
            DishCircle,
            _fit_mild_dish_ellipse,
        )

        edge = np.zeros((256, 272), np.float32)
        cv2.ellipse(edge, (136, 126), (92, 84), 13, 0, 360, 1.0, 2)
        fitted = _fit_mild_dish_ellipse(
            torch.from_numpy(edge)[None, None], 136.0, 126.0, 88.0
        )
        self.assertIsNotNone(fitted)
        assert fitted is not None
        centre_x, centre_y, axis_x, axis_y, angle, confidence = fitted
        self.assertGreater(max(axis_x, axis_y) / min(axis_x, axis_y), 1.05)
        self.assertGreater(confidence, 0.5)
        dish = DishCircle(
            round(centre_x),
            round(centre_y),
            int(np.ceil(max(axis_x, axis_y))),
            confidence,
            outer_radius_x=axis_x,
            outer_radius_y=axis_y,
            ellipse_angle_degrees=angle,
        )
        left, top, right, bottom = dish.bounds
        self.assertLess(left, centre_x - min(axis_x, axis_y))
        self.assertGreater(right, centre_x + min(axis_x, axis_y))
        self.assertLessEqual(dish.outer_radius_x, dish.radius)
        self.assertLessEqual(dish.outer_radius_y, dish.radius)
        self.assertLess(top, bottom)

    def check_test_pilot_batch_rims_match_the_calibrated_96_mm_outer_edges(self) -> None:
        import cv2
        from pathlib import Path

        from seedvision.calibration.geometry import detect_dish
        from seedvision.calibration.image import calibrate_image
        from seedvision.cuda import CudaContext

        root = Path(__file__).resolve().parents[1]
        paths = sorted((root / "images").glob("*.JPG"))
        if not all(path.exists() for path in paths):
            self.skipTest("Petri-rim regression fixtures are not present")
        if len(paths) != 11:
            self.skipTest("The complete eleven-image pilot batch is not present")
        context = CudaContext.resolve()
        for path in paths:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            calibration = calibrate_image(image, cuda_context=context)
            dish = detect_dish(
                calibration.corrected_bgr,
                cuda_context=context,
                image_tensor=calibration.gpu_corrected_bgr,
                pixels_per_mm=calibration.pixels_per_mm,
            )
            with self.subTest(path=path.name):
                self.assertIsNotNone(calibration.pixels_per_mm)
                inner_radius_mm = dish.inner_radius / calibration.pixels_per_mm
                outer_radius_mm = dish.outer_radius / calibration.pixels_per_mm
                self.assertGreaterEqual(inner_radius_mm, 43.5)
                self.assertLessEqual(inner_radius_mm, 47.0)
                self.assertGreaterEqual(outer_radius_mm, 47.0)
                self.assertLessEqual(outer_radius_mm, 49.2)
                self.assertTrue(dish.rim_pair_detected)

    def test_pilot_batch_rims_match_the_calibrated_96_mm_outer_edges(self):
        self.scenario(self.check_test_pilot_batch_rims_match_the_calibrated_96_mm_outer_edges, labelled=False)

    def test_validation_and_recovery(self):
        """2 related ImageCalibrationTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_scale_is_rejected_when_longest_ticks_are_not_major_increments)
        self.scenario(self.check_test_missing_references_produce_a_safe_uncalibrated_result)

    def test_workflow_lifecycle(self):
        """2 related ImageCalibrationTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_swatch_grid_snaps_to_shared_printed_edges)
        self.scenario(self.check_test_mild_elliptical_rim_is_retained_with_conservative_bounds)


if __name__ == "__main__":
    unittest.main()
