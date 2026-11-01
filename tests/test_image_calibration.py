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

    @staticmethod
    def _faint_rim_scene(*, seed_mass: bool = True):
        """A thin, low-contrast glass rim (r 200) around a dark seed mass (r 150)."""

        import cv2
        import numpy as np

        image = np.full((800, 1200, 3), 235, dtype=np.uint8)
        cv2.circle(image, (700, 330), 200, (165, 165, 165), 2, cv2.LINE_AA)
        cv2.circle(image, (700, 330), 192, (205, 205, 205), 2, cv2.LINE_AA)
        if seed_mass:
            cv2.circle(image, (712, 322), 150, (60, 60, 60), -1, cv2.LINE_AA)
        return image

    def check_test_faint_rim_replaces_an_enclosed_seed_mass_with_a_ruler_scale(self) -> None:
        from unittest import mock

        from seedvision.calibration import geometry

        with mock.patch.object(
            geometry, "_search_calibrated_rim", wraps=geometry._search_calibrated_rim
        ) as search:
            dish = geometry.detect_dish(self._faint_rim_scene(), pixels_per_mm=200 / 48.0)
        # The generic bank prefers the much stronger seed-mass edge; only the
        # calibrated search recovers the glass.
        search.assert_called_once()
        self.assertLessEqual(abs(dish.center_x - 700), 2)
        self.assertLessEqual(abs(dish.center_y - 330), 2)
        self.assertGreaterEqual(dish.outer_radius, 198)
        self.assertLessEqual(dish.outer_radius, 204)
        self.assertLess(dish.inner_radius, dish.outer_radius)
        self.assertTrue(dish.rim_pair_detected)

    def check_test_rim_agreeing_with_the_ruler_or_contradicting_it_clearly_is_kept(self) -> None:
        from unittest import mock

        from seedvision.calibration import geometry

        image = self._faint_rim_scene(seed_mass=False)
        uncalibrated = geometry.detect_dish(image)
        for pixels_per_mm, searches in ((200 / 48.0, 0), (230 / 48.0, 1)):
            with self.subTest(pixels_per_mm=pixels_per_mm), mock.patch.object(
                geometry, "_search_calibrated_rim", wraps=geometry._search_calibrated_rim
            ) as search:
                dish = geometry.detect_dish(image, pixels_per_mm=pixels_per_mm)
                # A 230 px ruler radius disagrees with the clear 200 px glass;
                # circles of that size only graze it, so the glass is kept.
                self.assertEqual(search.call_count, searches)
                self.assertEqual(
                    (dish.center_x, dish.center_y, dish.outer_radius),
                    (uncalibrated.center_x, uncalibrated.center_y, uncalibrated.outer_radius),
                )

    def check_test_fixture_rims_match_the_calibrated_96_mm_dish(self) -> None:
        """Fixtures where a seed mass or colour-card arcs outscored faint glass.

        Centres were read on a millimetre grid (about 1 mm accuracy).
        IMG_9632c is a control whose rim the generic search already resolves.
        """

        import cv2
        import numpy as np
        from pathlib import Path
        from unittest import mock

        from seedvision.calibration import geometry
        from seedvision.calibration.image import calibrate_image
        from seedvision.cuda import CudaContext

        reviewed_centres = {
            "IMG_9403.JPG": (4100, 2262),
            "IMG_9636c.JPG": (1934, 842),
            "IMG_9641c.JPG": (1952, 871),
            "IMG_9668c.JPG": (1939, 860),
            "IMG_9719.JPG": (3787, 1669),
            "IMG_9632c.JPG": None,
        }
        root = Path(__file__).resolve().parents[1] / "images"
        if not all((root / name).exists() for name in reviewed_centres):
            self.skipTest("Petri-rim regression fixtures are not present")
        context = CudaContext.resolve()
        for name, centre in reviewed_centres.items():
            calibration = calibrate_image(
                cv2.imread(str(root / name), cv2.IMREAD_COLOR), cuda_context=context
            )
            with self.subTest(path=name), mock.patch.object(
                geometry, "_search_calibrated_rim", wraps=geometry._search_calibrated_rim
            ) as search:
                dish = geometry.detect_dish(
                    calibration.corrected_bgr,
                    cuda_context=context,
                    image_tensor=calibration.gpu_corrected_bgr,
                    pixels_per_mm=calibration.pixels_per_mm,
                )
                pixels_per_mm = calibration.pixels_per_mm
                self.assertIsNotNone(pixels_per_mm)
                self.assertEqual(search.call_count, 0 if centre is None else 1)
                exterior_diameter_mm = sum(dish.outer_axes) / pixels_per_mm
                self.assertGreaterEqual(exterior_diameter_mm, 92.5)
                self.assertLessEqual(exterior_diameter_mm, 99.5)
                self.assertLess(dish.inner_radius, dish.outer_radius)
                self.assertTrue(dish.rim_pair_detected)
                if centre is not None:
                    offset = np.hypot(dish.center_x - centre[0], dish.center_y - centre[1])
                    self.assertLessEqual(offset / pixels_per_mm, 2.5)

    def test_fixture_rims_match_the_calibrated_96_mm_dish(self):
        self.scenario(self.check_test_fixture_rims_match_the_calibrated_96_mm_dish, labelled=False)

    def test_validation_and_recovery(self):
        """2 related ImageCalibrationTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_scale_is_rejected_when_longest_ticks_are_not_major_increments)
        self.scenario(self.check_test_missing_references_produce_a_safe_uncalibrated_result)

    def test_workflow_lifecycle(self):
        """4 related ImageCalibrationTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_swatch_grid_snaps_to_shared_printed_edges)
        self.scenario(self.check_test_mild_elliptical_rim_is_retained_with_conservative_bounds)
        self.scenario(self.check_test_faint_rim_replaces_an_enclosed_seed_mass_with_a_ruler_scale)
        self.scenario(self.check_test_rim_agreeing_with_the_ruler_or_contradicting_it_clearly_is_kept)


if __name__ == "__main__":
    unittest.main()
