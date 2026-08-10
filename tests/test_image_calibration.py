from __future__ import annotations

import unittest


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

    def test_detects_swatch_grid_and_ruler_independently(self) -> None:
        from seedvision.calibration import detect_colour_card, detect_ruler

        image = self._synthetic_calibration_image(rotation_degrees=2.0)
        card = detect_colour_card(image)
        ruler = detect_ruler(image)
        self.assertGreaterEqual(card.detected_swatch_count, 18)
        self.assertEqual(len(card.swatches), 24)
        self.assertGreater(card.confidence, 0.45)
        self.assertGreater(ruler.length_px, 500)
        self.assertGreater(ruler.confidence, 0.6)
        self.assertAlmostEqual(card.angle_degrees, -2.0, delta=1.0)
        self.assertAlmostEqual(ruler.angle_degrees, -2.0, delta=1.0)

    def test_reference_card_and_printed_ruler_scale_are_aligned(self) -> None:
        from seedvision.calibration import detect_colour_card, detect_ruler

        image = self._synthetic_calibration_image()
        card = detect_colour_card(image)
        ruler = detect_ruler(image)
        self.assertEqual(card.detected_swatch_count, 24)
        bounds = sorted(card.bounds, key=lambda point: point[0] + point[1])
        self.assertAlmostEqual(bounds[0][0], 45.0, delta=6.0)
        self.assertAlmostEqual(bounds[0][1], 35.0, delta=6.0)
        ruler_x = sorted((ruler.endpoint_a[0], ruler.endpoint_b[0]))
        self.assertAlmostEqual(ruler_x[0], 525.0, delta=4.0)
        self.assertAlmostEqual(ruler_x[1], 1125.0, delta=4.0)
        self.assertAlmostEqual(ruler.length_px, 600.0, delta=5.0)

    def test_swatch_grid_snaps_to_shared_printed_edges(self) -> None:
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

    def test_projective_deskew_rectifies_mild_camera_tilt(self) -> None:
        import cv2
        import numpy as np

        from seedvision.calibration import calibrate_image

        image = self._synthetic_calibration_image()
        source = np.float32(((0, 0), (1199, 0), (1199, 799), (0, 799)))
        target = np.float32(((18, 12), (1175, 34), (1190, 775), (5, 748)))
        tilted = cv2.warpPerspective(
            image,
            cv2.getPerspectiveTransform(source, target),
            (1200, 800),
            borderMode=cv2.BORDER_REPLICATE,
        )
        result = calibrate_image(tilted)
        self.assertTrue(result.perspective_corrected)
        self.assertGreater(result.perspective_strength, 0.01)
        bounds = result.transform_points(result.colour_card.bounds)
        sides = [bounds[(index + 1) % 4] - bounds[index] for index in range(4)]
        lengths = [float(np.linalg.norm(side)) for side in sides]
        self.assertAlmostEqual(lengths[0], lengths[2], delta=1.0)
        self.assertAlmostEqual(lengths[1], lengths[3], delta=1.0)
        corner_cosine = abs(float(np.dot(sides[0], sides[1]))) / (
            lengths[0] * lengths[1]
        )
        self.assertLess(corner_cosine, 0.01)

    def test_calibration_corrects_cast_deskews_and_assigns_scale(self) -> None:
        from seedvision.calibration import calibrate_image

        result = calibrate_image(
            self._synthetic_calibration_image(rotation_degrees=2.0)
        )
        self.assertIsNotNone(result.colour_card)
        self.assertIsNotNone(result.ruler)
        self.assertAlmostEqual(result.deskew_degrees, -2.0, delta=1.0)
        blue_gain, _, red_gain = result.channel_gains_bgr
        self.assertLess(blue_gain, 1.0)
        self.assertGreater(red_gain, 1.0)
        self.assertIsNotNone(result.pixels_per_mm)
        self.assertAlmostEqual(result.pixels_per_mm, 4.0, delta=1.0)
        self.assertAlmostEqual(result.ruler.angle_degrees, 0.0, delta=1.0)
        self.assertEqual(result.scale_bar_mm, 50.0)
        self.assertGreater(result.scale_confidence, 0.1)
        self.assertGreaterEqual(result.corrected_bgr.shape[0], 800)
        self.assertGreaterEqual(result.corrected_bgr.shape[1], 1200)
        self.assertEqual(result.corrected_bgr.shape[2], 3)

    def test_missing_references_produce_a_safe_uncalibrated_result(self) -> None:
        import numpy as np

        from seedvision.calibration import calibrate_image

        image = np.full((300, 400, 3), 180, dtype=np.uint8)
        result = calibrate_image(image)
        self.assertIsNone(result.colour_card)
        self.assertIsNone(result.ruler)
        self.assertIsNone(result.pixels_per_mm)
        self.assertEqual(result.deskew_degrees, 0.0)
        self.assertTrue(result.warnings)

    def test_detects_both_concentric_petri_dish_edges(self) -> None:
        import cv2
        import numpy as np

        from seedvision.calibration.geometry import (
            DishDetectionSettings,
            detect_dish,
        )

        image = np.full((800, 1200, 3), 232, dtype=np.uint8)
        center = (700, 350)
        cv2.circle(image, center, 180, (65, 65, 65), 4)
        cv2.circle(image, center, 196, (92, 92, 92), 4)
        dish = detect_dish(
            image,
            DishDetectionSettings(
                min_radius_fraction=0.18,
                max_radius_fraction=0.28,
                expected_center_x_fraction=center[0] / image.shape[1],
                expected_center_y_fraction=center[1] / image.shape[0],
                expected_radius_fraction=180 / image.shape[0],
            ),
        )

        self.assertEqual(dish.vessel_type, "petri_dish")
        self.assertTrue(dish.rim_pair_detected)
        self.assertAlmostEqual(dish.lower_edge_radius, 180, delta=5)
        self.assertAlmostEqual(dish.upper_edge_radius, 196, delta=5)
        self.assertLess(dish.inner_radius, dish.outer_radius)
        self.assertEqual(dish.radius, dish.outer_radius)


if __name__ == "__main__":
    unittest.main()
